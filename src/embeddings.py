import asyncio
import logging
import os
import shutil
import struct
import tempfile
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
_RERANK_MODEL_URL = "https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2/resolve/main/onnx/model_quint8_avx2.onnx"
_RERANK_TOKENIZER_URL = "https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2/resolve/main/tokenizer.json"

_openai: AsyncOpenAI | None = None
_rerank_session: ort.InferenceSession | None = None
_rerank_tokenizer: Tokenizer | None = None


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


def load_model():
    global _openai, _rerank_session, _rerank_tokenizer

    if _openai is None:
        _openai = AsyncOpenAI()
        logger.info("OpenAI client initialized (%s, %d dims)", _EMBEDDING_MODEL, EMBEDDING_DIM)

    if _rerank_session is None:
        model_path = os.path.join(_RERANK_DIR, "model.onnx")
        tokenizer_path = os.path.join(_RERANK_DIR, "tokenizer.json")
        _download_if_needed(_RERANK_MODEL_URL, model_path)
        _download_if_needed(_RERANK_TOKENIZER_URL, tokenizer_path)
        _rerank_session = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        _rerank_tokenizer = Tokenizer.from_file(tokenizer_path)
        _rerank_tokenizer.enable_padding(length=256)
        _rerank_tokenizer.enable_truncation(max_length=256)
        logger.info("Reranker loaded (ms-marco-MiniLM-L-6-v2)")


def _pack(vec: list[float]) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)


async def get_embedding(text: str) -> bytes:
    assert _openai, "Call load_model() first"
    response = await _openai.embeddings.create(input=text, model=_EMBEDDING_MODEL)
    return _pack(response.data[0].embedding)


async def get_embeddings_batch(texts: list[str]) -> list[bytes]:
    if not texts:
        return []
    assert _openai, "Call load_model() first"
    response = await _openai.embeddings.create(input=texts, model=_EMBEDDING_MODEL)
    return [_pack(e.embedding) for e in response.data]


def _rerank_sync(query: str, documents: list[str]) -> list[float]:
    assert _rerank_session and _rerank_tokenizer, "Call load_model() first"
    if not documents:
        return []
    encodings = [_rerank_tokenizer.encode(query, doc) for doc in documents]
    n = len(encodings)
    input_ids = np.array([e.ids for e in encodings], dtype=np.int64).reshape(n, 256)
    attention_mask = np.array([e.attention_mask for e in encodings], dtype=np.int64).reshape(n, 256)
    token_type_ids = np.array([e.type_ids for e in encodings], dtype=np.int64).reshape(n, 256)
    outputs = _rerank_session.run(
        None,
        {"input_ids": input_ids, "attention_mask": attention_mask, "token_type_ids": token_type_ids},
    )
    return outputs[0][:, 0].tolist()


async def rerank(query: str, documents: list[str]) -> list[float]:
    return await asyncio.to_thread(_rerank_sync, query, documents)


async def generate_questions(content: str, memory_type: str = "general") -> list[str]:
    assert _openai, "Call load_model() first"
    response = await _openai.chat.completions.create(
        model=_QUESTION_MODEL,
        messages=[
            {"role": "system", "content": "Generate 3-5 short questions that this fact could answer. Return only the questions, one per line. No numbering or bullets."},
            {"role": "user", "content": f"[{memory_type}] {content}"},
        ],
        max_completion_tokens=200,
        temperature=0.3,
    )
    text = response.choices[0].message.content or ""
    return [q.strip() for q in text.strip().split("\n") if q.strip()]
