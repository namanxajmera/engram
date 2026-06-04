import asyncio
import collections
import hashlib
import logging
import os
import shutil
import struct
import tempfile
import time
import urllib.request

import numpy as np
import onnxruntime as ort
from openai import AsyncOpenAI
from tokenizers import Tokenizer

logger = logging.getLogger(__name__)

EMBEDDING_DIM = 1536
_EMBEDDING_MODEL = "text-embedding-3-small"
_QUESTION_MODEL = os.environ.get("ENGRAM_QUESTION_MODEL", "gpt-5.4-nano")
_DATA_DIR = "/data" if os.path.isdir("/data") else "./data"

_RERANK_DIR = os.path.join(_DATA_DIR, "model", "ms-marco-MiniLM-L-6-v2")
_RERANK_HF = "https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2/resolve/main/onnx"
_RERANK_TOKENIZER_URL = f"{_RERANK_HF}/tokenizer.json"
_RERANK_MAX_LENGTH = 128
_RERANK_BATCH_SIZE = 32  # cap candidates per ONNX pass to bound peak memory

_openai: AsyncOpenAI | None = None
_rerank_session: ort.InferenceSession | None = None
_rerank_tokenizer: Tokenizer | None = None

_CACHE_MAX_SIZE = 1000
_CACHE_TTL = 7 * 24 * 3600  # 7 days
_embedding_cache: collections.OrderedDict[str, tuple[bytes, float]] = collections.OrderedDict()


def _cache_key(text: str) -> str:
    return hashlib.sha256(f"{text.strip().lower()}|{_EMBEDDING_MODEL}".encode()).hexdigest()


def _cache_get(key: str) -> bytes | None:
    entry = _embedding_cache.get(key)
    if entry is None:
        return None
    value, ts = entry
    if time.monotonic() - ts > _CACHE_TTL:
        del _embedding_cache[key]
        return None
    _embedding_cache.move_to_end(key)
    return value


def _cache_put(key: str, value: bytes):
    if key in _embedding_cache:
        _embedding_cache.move_to_end(key)
    elif len(_embedding_cache) >= _CACHE_MAX_SIZE:
        _embedding_cache.popitem(last=False)
    _embedding_cache[key] = (value, time.monotonic())


def _download_if_needed(url: str, path: str):
    if os.path.exists(path):
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    logger.info("Downloading %s", url)
    tmp_fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(path))
    try:
        os.close(tmp_fd)
        urllib.request.urlretrieve(url, tmp_path)
        shutil.move(tmp_path, path)
    except Exception:
        os.unlink(tmp_path)
        raise


def _detect_rerank_model_url() -> str:
    import platform

    arch = platform.machine().lower()
    if arch in ("aarch64", "arm64"):
        logger.info("Detected ARM64 — using qint8_arm64 reranker")
        return f"{_RERANK_HF}/model_qint8_arm64.onnx"
    # /proc/cpuinfo exists on Linux; absent on macOS (falls through to fp32)
    try:
        with open("/proc/cpuinfo") as f:
            flags = f.read()
        if "avx512" in flags:
            logger.info("Detected AVX-512 — using qint8_avx512 reranker")
            return f"{_RERANK_HF}/model_qint8_avx512.onnx"
        if "avx2" in flags:
            logger.info("Detected AVX2 — using quint8_avx2 reranker")
            return f"{_RERANK_HF}/model_quint8_avx2.onnx"
    except OSError:
        pass
    logger.info("No SIMD detected — using optimized fp32 reranker (O2)")
    return f"{_RERANK_HF}/model_O2.onnx"


def load_model():
    global _openai, _rerank_session, _rerank_tokenizer

    if _openai is None:
        _openai = AsyncOpenAI()
        logger.info("OpenAI client initialized (%s, %d dims)", _EMBEDDING_MODEL, EMBEDDING_DIM)

    if _rerank_session is None:
        model_url = _detect_rerank_model_url()
        model_path = os.path.join(_RERANK_DIR, "model.onnx")
        tokenizer_path = os.path.join(_RERANK_DIR, "tokenizer.json")
        marker_path = os.path.join(_RERANK_DIR, ".model_url")
        # Re-download if model variant changed
        if os.path.exists(model_path):
            existing_url = ""
            if os.path.exists(marker_path):
                with open(marker_path) as f:
                    existing_url = f.read().strip()
            if existing_url != model_url:
                logger.info("Reranker model variant changed, re-downloading")
                os.remove(model_path)
        _download_if_needed(model_url, model_path)
        with open(marker_path, "w") as f:
            f.write(model_url)
        _download_if_needed(_RERANK_TOKENIZER_URL, tokenizer_path)
        sess_opts = ort.SessionOptions()
        sess_opts.inter_op_num_threads = 1
        sess_opts.intra_op_num_threads = 2
        sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        _rerank_session = ort.InferenceSession(model_path, sess_options=sess_opts, providers=["CPUExecutionProvider"])
        _rerank_tokenizer = Tokenizer.from_file(tokenizer_path)
        _rerank_tokenizer.enable_padding(length=_RERANK_MAX_LENGTH)
        _rerank_tokenizer.enable_truncation(max_length=_RERANK_MAX_LENGTH)
        logger.info("Reranker loaded (ms-marco-MiniLM-L-6-v2, max_length=%d)", _RERANK_MAX_LENGTH)


def _pack(vec: list[float]) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)


async def get_embedding(text: str) -> bytes:
    assert _openai, "Call load_model() first"
    key = _cache_key(text)
    cached = _cache_get(key)
    if cached is not None:
        return cached
    response = await _openai.embeddings.create(input=text, model=_EMBEDDING_MODEL)
    packed = _pack(response.data[0].embedding)
    _cache_put(key, packed)
    return packed


async def get_embeddings_batch(texts: list[str]) -> list[bytes]:
    if not texts:
        return []
    assert _openai, "Call load_model() first"
    response = await _openai.embeddings.create(input=texts, model=_EMBEDDING_MODEL)
    return [_pack(e.embedding) for e in response.data]


def _rerank_batch(encodings: list) -> list[float]:
    n = len(encodings)
    ml = _RERANK_MAX_LENGTH
    input_ids = np.array([e.ids for e in encodings], dtype=np.int64).reshape(n, ml)
    attention_mask = np.array([e.attention_mask for e in encodings], dtype=np.int64).reshape(n, ml)
    token_type_ids = np.array([e.type_ids for e in encodings], dtype=np.int64).reshape(n, ml)
    outputs = _rerank_session.run(
        None,
        {"input_ids": input_ids, "attention_mask": attention_mask, "token_type_ids": token_type_ids},
    )
    return outputs[0][:, 0].tolist()


def _rerank_sync(query: str, documents: list[str]) -> list[float]:
    assert _rerank_session and _rerank_tokenizer, "Call load_model() first"
    if not documents:
        return []
    encodings = [_rerank_tokenizer.encode(query, doc) for doc in documents]
    scores: list[float] = []
    for start in range(0, len(encodings), _RERANK_BATCH_SIZE):
        scores.extend(_rerank_batch(encodings[start : start + _RERANK_BATCH_SIZE]))
    return scores


async def rerank(query: str, documents: list[str]) -> list[float]:
    return await asyncio.to_thread(_rerank_sync, query, documents)


async def generate_questions(content: str, memory_type: str = "general") -> list[str]:
    assert _openai, "Call load_model() first"
    response = await _openai.chat.completions.create(
        model=_QUESTION_MODEL,
        messages=[
            {
                "role": "system",
                "content": "Generate 3-5 short questions that this fact could answer. Return only the questions, one per line. No numbering or bullets.",
            },
            {"role": "user", "content": f"[{memory_type}] {content}"},
        ],
        max_completion_tokens=200,
        temperature=0.3,
    )
    text = response.choices[0].message.content or ""
    return [q.strip() for q in text.strip().split("\n") if q.strip()]
