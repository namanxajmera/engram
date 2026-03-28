import asyncio
import contextlib
import hashlib
import json
import logging
import os
import time
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime

import aiosqlite
import sqlite_vec

from src.embeddings import _DATA_DIR, EMBEDDING_DIM

logger = logging.getLogger(__name__)

DB_PATH = os.path.join(_DATA_DIR, "memories.db")
VALID_MEMORY_TYPES = {"user", "feedback", "project", "reference", "general"}
MAX_SOURCE_LENGTH = 50
MAX_CONTENT_LENGTH = 10_000
MAX_LIMIT = 100

_TAG_FILTER = " AND EXISTS (SELECT 1 FROM json_each(tags) WHERE json_each.value = ?)"
_COLS = "id, content, tags, memory_type, metadata, source, valid_at, invalid_at, created_at, updated_at"
_MAX_HISTORY = 50
_VEC_SQL = f"""CREATE VIRTUAL TABLE memory_embeddings USING vec0(
    memory_id INTEGER PRIMARY KEY, embedding FLOAT[{EMBEDDING_DIM}] distance_metric=cosine
)"""

_db: aiosqlite.Connection | None = None
_read_pool: list[aiosqlite.Connection] = []
_READ_POOL_SIZE = 3


async def _configure_connection(db: aiosqlite.Connection, readonly: bool = False):
    """Apply performance PRAGMAs and load extensions."""
    db.row_factory = aiosqlite.Row
    await db.enable_load_extension(True)
    await db.load_extension(sqlite_vec.loadable_path())
    await db.enable_load_extension(False)
    if not readonly:
        await db.execute("PRAGMA journal_mode=WAL")
    await db.execute("PRAGMA mmap_size=268435456")  # 256MB mmap
    await db.execute("PRAGMA cache_size=-65536")  # 64MB page cache
    await db.execute("PRAGMA temp_store=MEMORY")


async def init_db() -> aiosqlite.Connection:
    global _db
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    db = await aiosqlite.connect(DB_PATH)
    await _configure_connection(db)
    await _init_tables(db)
    _db = db
    # Read-only pool for parallel search queries
    for _ in range(_READ_POOL_SIZE):
        rconn = await aiosqlite.connect(f"file:{DB_PATH}?mode=ro", uri=True)
        await _configure_connection(rconn, readonly=True)
        _read_pool.append(rconn)
    logger.info("Database initialized at %s (read pool=%d)", DB_PATH, _READ_POOL_SIZE)
    return db


async def get_db() -> aiosqlite.Connection:
    if _db is None:
        raise RuntimeError("Database not initialized")
    return _db


async def close_db():
    global _db
    for rconn in _read_pool:
        await rconn.close()
    _read_pool.clear()
    if _db:
        await _db.close()
        _db = None


async def _init_tables(db: aiosqlite.Connection):
    await db.executescript("""
        CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content_hash TEXT UNIQUE NOT NULL,
            content TEXT NOT NULL,
            tags TEXT DEFAULT '[]',
            memory_type TEXT DEFAULT 'general',
            metadata TEXT DEFAULT '{}',
            source TEXT DEFAULT 'unknown',
            valid_at TEXT,
            invalid_at TEXT DEFAULT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            deleted_at TEXT DEFAULT NULL
        );
        CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
            content, content='memories', content_rowid='id', tokenize='porter'
        );
        CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN
            INSERT INTO memory_fts(rowid, content) VALUES (new.id, new.content);
        END;
        CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN
            INSERT INTO memory_fts(memory_fts, rowid, content) VALUES('delete', old.id, old.content);
        END;
        CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN
            INSERT INTO memory_fts(memory_fts, rowid, content) VALUES('delete', old.id, old.content);
            INSERT INTO memory_fts(rowid, content) VALUES (new.id, new.content);
        END;
    """)
    try:
        await db.execute(_VEC_SQL)
    except aiosqlite.OperationalError:
        try:
            dummy = b"\x00" * (EMBEDDING_DIM * 4)
            await db.execute("INSERT INTO memory_embeddings (memory_id, embedding) VALUES (-1, ?)", (dummy,))
            await db.execute("DELETE FROM memory_embeddings WHERE memory_id = -1")
        except Exception:
            logger.warning("Embedding dimension changed, rebuilding vector table")
            await db.execute("DROP TABLE memory_embeddings")
            await db.execute(_VEC_SQL)
    with contextlib.suppress(aiosqlite.OperationalError):
        await db.execute(f"""CREATE VIRTUAL TABLE memory_question_embeddings USING vec0(
            rowid INTEGER PRIMARY KEY, embedding FLOAT[{EMBEDDING_DIM}] distance_metric=cosine
        )""")
    await db.executescript("""
        CREATE TABLE IF NOT EXISTS memory_questions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            memory_id INTEGER NOT NULL,
            question TEXT NOT NULL,
            FOREIGN KEY (memory_id) REFERENCES memories(id)
        );
        CREATE INDEX IF NOT EXISTS idx_mq_mid ON memory_questions(memory_id);
    """)
    await db.commit()
    logger.info("Schema initialized at %s", DB_PATH)


def _hash(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()


def _now() -> str:
    return datetime.now(UTC).isoformat()


def _escape_fts(text: str) -> str:
    return " ".join('"' + t.replace('"', '""') + '"' for t in text.split() if t)


def _row_to_dict(row) -> dict:
    d = {
        k: row[k]
        for k in ("id", "content", "memory_type", "source", "valid_at", "invalid_at", "created_at", "updated_at")
    }
    d["tags"] = json.loads(row["tags"])
    meta = json.loads(row["metadata"])
    if meta and meta.get("history"):
        d["history"] = meta["history"]
    return d


async def clear_questions(db: aiosqlite.Connection, memory_id: int):
    await db.execute(
        "DELETE FROM memory_question_embeddings WHERE rowid IN (SELECT id FROM memory_questions WHERE memory_id = ?)",
        (memory_id,),
    )
    await db.execute("DELETE FROM memory_questions WHERE memory_id = ?", (memory_id,))


async def store_questions(db: aiosqlite.Connection, memory_id: int, questions: list[tuple[str, bytes]]):
    for text, emb in questions:
        cur = await db.execute("INSERT INTO memory_questions (memory_id, question) VALUES (?, ?)", (memory_id, text))
        await db.execute(
            "INSERT INTO memory_question_embeddings (rowid, embedding) VALUES (?, ?)", (cur.lastrowid, emb)
        )


async def check_duplicate(db: aiosqlite.Connection, content: str) -> int | None:
    rows = await db.execute_fetchall(
        "SELECT id FROM memories WHERE content_hash = ? AND deleted_at IS NULL", (_hash(content),)
    )
    return rows[0][0] if rows else None


async def add_memory(
    db: aiosqlite.Connection,
    content: str,
    embedding: bytes,
    tags: list[str] | None = None,
    memory_type: str = "general",
    source: str | None = None,
    valid_at: str | None = None,
    questions: list[tuple[str, bytes]] | None = None,
) -> dict:
    now = _now()
    ch = _hash(content)
    try:
        cursor = await db.execute(
            """INSERT INTO memories (content_hash, content, tags, memory_type, source, valid_at, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (ch, content, json.dumps(tags or []), memory_type, source or "unknown", valid_at, now, now),
        )
        mid = cursor.lastrowid
        await db.execute("INSERT INTO memory_embeddings (memory_id, embedding) VALUES (?, ?)", (mid, embedding))
        if questions:
            await store_questions(db, mid, questions)
        await db.commit()
        return {"id": mid, "content_hash": ch, "status": "created"}
    except aiosqlite.IntegrityError:
        return {"status": "duplicate"}


def _rrf_score(ranks: dict[int, list[int]], k: int = 60) -> list[tuple[int, float]]:
    scores: dict[int, float] = {}
    for mid, positions in ranks.items():
        for rank in positions:
            scores[mid] = scores.get(mid, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


async def _vec_search(conn: aiosqlite.Connection, embedding: bytes, limit: int) -> list:
    return await conn.execute_fetchall(
        "SELECT memory_id, distance FROM memory_embeddings WHERE embedding MATCH ? ORDER BY distance LIMIT ?",
        (embedding, limit),
    )


async def _hyde_search(conn: aiosqlite.Connection, embedding: bytes, limit: int) -> list[tuple[int, float]]:
    hyde_raw = await conn.execute_fetchall(
        "SELECT rowid, distance FROM memory_question_embeddings WHERE embedding MATCH ? ORDER BY distance LIMIT ?",
        (embedding, limit),
    )
    if not hyde_raw:
        return []
    q_ids = [r[0] for r in hyde_raw]
    ph = ",".join("?" * len(q_ids))
    q_map = {
        r[0]: r[1]
        for r in await conn.execute_fetchall(f"SELECT id, memory_id FROM memory_questions WHERE id IN ({ph})", q_ids)
    }
    return [(q_map[r[0]], r[1]) for r in hyde_raw if r[0] in q_map]


async def _bm25_search(conn: aiosqlite.Connection, query_text: str, limit: int) -> list:
    escaped = _escape_fts(query_text)
    if not escaped:
        return []
    return await conn.execute_fetchall(
        "SELECT rowid, rank FROM memory_fts WHERE memory_fts MATCH ? ORDER BY rank LIMIT ?",
        (escaped, limit),
    )


async def search_memories(
    db: aiosqlite.Connection,
    query_embedding: bytes,
    query_text: str,
    tag_filter: str | None = None,
    limit: int = 5,
    rerank_fn: Callable[[str, list[str]], Awaitable[list[float]]] | None = None,
) -> list[dict]:
    limit = min(limit, MAX_LIMIT)
    fetch_limit = max(limit * 5, 20)

    t0 = time.perf_counter()

    # Run all 3 retrieval queries in parallel on separate read connections
    if len(_read_pool) >= 3:
        vec_results, hyde_results, bm25_results = await asyncio.gather(
            _vec_search(_read_pool[0], query_embedding, fetch_limit),
            _hyde_search(_read_pool[1], query_embedding, fetch_limit),
            _bm25_search(_read_pool[2], query_text, fetch_limit),
        )
    else:
        vec_results = await _vec_search(db, query_embedding, fetch_limit)
        hyde_results = await _hyde_search(db, query_embedding, fetch_limit)
        bm25_results = await _bm25_search(db, query_text, fetch_limit)
    t1 = time.perf_counter()

    ranks: dict[int, list[int]] = {}
    for i, (mid, _) in enumerate(vec_results):
        ranks.setdefault(mid, []).append(i + 1)
    for i, (mid, _) in enumerate(hyde_results):
        ranks.setdefault(mid, []).append(i + 1)
    for i, (rid, _) in enumerate(bm25_results):
        ranks.setdefault(rid, []).append(i + 1)

    rrf_ranked = _rrf_score(ranks)
    candidate_ids = [mid for mid, _ in rrf_ranked[:fetch_limit]]
    if not candidate_ids:
        return []

    ph = ",".join("?" * len(candidate_ids))
    params: list = list(candidate_ids)
    tag_clause = ""
    if tag_filter:
        tag_clause = _TAG_FILTER
        params.append(tag_filter)

    rows = await db.execute_fetchall(
        f"SELECT {_COLS} FROM memories WHERE id IN ({ph}) AND deleted_at IS NULL{tag_clause}", params
    )
    row_map = {row["id"]: row for row in rows}
    candidates = [(mid, s, row_map[mid]) for mid, s in rrf_ranked if mid in row_map]

    t2 = time.perf_counter()
    if rerank_fn and len(candidates) > limit:
        scores = await rerank_fn(query_text, [row["content"] for _, _, row in candidates])
        scored = sorted(zip(candidates, scores, strict=True), key=lambda x: x[1], reverse=True)
        candidates = [(mid, score, row) for (mid, _, row), score in scored]
    t3 = time.perf_counter()

    logger.info(
        "SEARCH: retrieve=%.3fs(vec=%d hyde=%d bm25=%d) fetch=%.3fs rerank=%.3fs(%d) total=%.3fs",
        t1 - t0, len(vec_results), len(hyde_results), len(bm25_results),
        t2 - t1, t3 - t2, len(candidates), t3 - t0,
    )

    return [{**_row_to_dict(row), "score": round(s, 4)} for _, s, row in candidates[:limit]]


async def get_memory(db: aiosqlite.Connection, memory_id: int) -> dict | None:
    rows = await db.execute_fetchall(f"SELECT {_COLS} FROM memories WHERE id = ? AND deleted_at IS NULL", (memory_id,))
    return _row_to_dict(rows[0]) if rows else None


async def get_profile_memories(db: aiosqlite.Connection, limit: int = 70) -> dict[str, list[str]]:
    rows = await db.execute_fetchall(
        """SELECT content, memory_type FROM memories
           WHERE deleted_at IS NULL AND memory_type IN ('user', 'feedback')
           ORDER BY memory_type, updated_at DESC LIMIT ?""",
        (min(limit, MAX_LIMIT),),
    )
    result: dict[str, list[str]] = {"user": [], "feedback": []}
    for r in rows:
        result[r["memory_type"]].append(r["content"])
    return result


async def list_memories(
    db: aiosqlite.Connection,
    memory_type: str | None = None,
    tag: str | None = None,
    offset: int = 0,
    limit: int = 20,
) -> list[dict]:
    limit = min(limit, MAX_LIMIT)
    query = f"SELECT {_COLS} FROM memories WHERE deleted_at IS NULL"
    params: list = []
    if memory_type:
        query += " AND memory_type = ?"
        params.append(memory_type)
    if tag:
        query += _TAG_FILTER
        params.append(tag)
    query += " ORDER BY updated_at DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])
    return [_row_to_dict(row) for row in await db.execute_fetchall(query, params)]


async def update_memory(
    db: aiosqlite.Connection,
    memory_id: int,
    content: str | None = None,
    tags: list[str] | None = None,
    memory_type: str | None = None,
    new_embedding: bytes | None = None,
    valid_at: str | None = None,
    source: str | None = None,
    questions: list[tuple[str, bytes]] | None = None,
) -> dict | None:
    now = _now()
    existing = await get_memory(db, memory_id)
    if not existing:
        return None

    updates, params = [], []

    if content is not None:
        history = existing.get("history", [])[-_MAX_HISTORY:]
        history.append(
            {
                "content": existing["content"],
                "source": existing.get("source"),
                "valid_at": existing.get("valid_at"),
                "invalid_at": now,
                "updated_at": existing["updated_at"],
            }
        )
        updates.extend(["content = ?", "content_hash = ?", "metadata = ?"])
        params.extend([content, _hash(content), json.dumps({"history": history})])
    if valid_at is not None:
        updates.append("valid_at = ?")
        params.append(valid_at)
    if tags is not None:
        updates.append("tags = ?")
        params.append(json.dumps(tags))
    if memory_type is not None:
        updates.append("memory_type = ?")
        params.append(memory_type)
    if source is not None:
        updates.append("source = ?")
        params.append(source)

    if updates:
        updates.append("updated_at = ?")
        params.append(now)
        params.append(memory_id)
        await db.execute(f"UPDATE memories SET {', '.join(updates)} WHERE id = ?", params)
    if new_embedding is not None:
        await db.execute("DELETE FROM memory_embeddings WHERE memory_id = ?", (memory_id,))
        await db.execute(
            "INSERT INTO memory_embeddings (memory_id, embedding) VALUES (?, ?)", (memory_id, new_embedding)
        )
    if questions is not None:
        await clear_questions(db, memory_id)
        await store_questions(db, memory_id, questions)

    await db.commit()
    return await get_memory(db, memory_id)


async def delete_memory(db: aiosqlite.Connection, memory_id: int) -> bool:
    cursor = await db.execute(
        "UPDATE memories SET deleted_at = ? WHERE id = ? AND deleted_at IS NULL", (_now(), memory_id)
    )
    await db.commit()
    return cursor.rowcount > 0
