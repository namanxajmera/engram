import asyncio
import json
import logging
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings

from src.auth import APIKeyMiddleware
from src.embeddings import generate_questions, get_embedding, get_embeddings_batch, load_model, rerank
from src.storage import (
    MAX_CONTENT_LENGTH,
    MAX_SOURCE_LENGTH,
    VALID_MEMORY_TYPES,
    add_memory,
    check_duplicate,
    close_db,
    delete_memory,
    get_db,
    get_memory,
    get_profile_memories,
    init_db,
    list_memories,
    search_memories,
    update_memory,
)

logger = logging.getLogger(__name__)

INSTRUCTIONS = """Engram is the user's personal memory service — a shared context layer across all their AI tools.
You MUST use these tools proactively. Do not wait for the user to ask you to remember or recall things.

RESOURCES (loaded automatically on connect):
- engram://profile — user identity, preferences, working style
- engram://recent — 10 most recently updated memories
- engram://projects — all project context

─── SEARCH (search_memories_tool) ───
Use PROACTIVELY at the START of every conversation to load context about the user and their current work.
Also use when:
- The user references past work, decisions, or preferences
- You need context about who the user is, what they've built, or how they like to work
- Before making assumptions — check memory first

Tips: Use descriptive queries ("user's current job and role"), not single words ("work").
If the first search misses, rephrase with different terms. Run multiple searches if needed.

─── STORE (add_memory_tool) ───
Use PROACTIVELY whenever you learn something worth remembering across sessions:
- User shares personal info, preferences, or corrections to your behavior
- Significant work completed (architecture decisions, key features, deployments)
- User explicitly asks you to remember something
- Any fact that would help another AI tool serve this user better

Format: ONE concise fact per memory. Not paragraphs.
  Good: "Adopted a golden retriever named Max in April 2025"
  Bad: "The user got a dog, they were thinking about it for a while..."

─── UPDATE (update_memory_tool) ───
Use when a stored fact has changed. ALWAYS search first to find the existing memory.
Old versions are automatically preserved in history with timestamps and source attribution.

Patterns:
- State change: "Revenue target was $50K/mo, revised to $20K/mo in Q2 2025"
- Stopped activity: "Ran a weekly podcast in 2024, stopped March 2025 to focus on product"
- Contradiction: search first, then update to capture both states with dates
- Redundant: merge into one cleaner version with the most complete wording

NEVER delete a memory — history has value even after facts change.

─── MEMORY TYPES (required on store) ───
- user: who the user is, preferences, role, personal details
- feedback: how the user wants you to work (tone, approach, things to avoid)
- project: project context, decisions, status, tech stack
- reference: pointers to external resources, URLs, docs
- general: anything that doesn't fit the above

─── TAGS ───
Short lowercase tags for categorization. Use consistent terms like: preference, decision, seo, tech-stack, metrics, communication, workflow.
Reuse existing tags when possible. Add new ones when these don't fit. Prefer 1-3 tags per memory.

─── REQUIRED FIELDS ───
When storing (add_memory_tool): content, source, memory_type, valid_at, tags (all required).
When updating (update_memory_tool): memory_id and source (always required), rest optional.
- source: lowercase identifier for your tool (e.g. "claude-code", "poke", "cursor")
- valid_at: when the fact became true (ISO date, e.g. "2026-03-21"), NOT when stored

─── SEARCH BEFORE STORE ───
ALWAYS search before creating a new memory. If a related memory exists, update it instead.
Duplicates waste space and create confusion across tools."""

mcp = FastMCP(
    "Engram",
    host="0.0.0.0",
    instructions=INSTRUCTIONS,
    transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
)


async def _embed_content(content: str, memory_type: str) -> tuple[bytes, list[tuple[str, bytes]]]:
    async def build_questions():
        texts = await generate_questions(content, memory_type)
        if not texts:
            return []
        embeddings = await get_embeddings_batch(texts)
        return list(zip(texts, embeddings, strict=True))

    return await asyncio.gather(get_embedding(content), build_questions())


@mcp.resource("engram://profile")
async def profile_resource() -> str:
    db = await get_db()
    grouped = await get_profile_memories(db)
    if not grouped["user"] and not grouped["feedback"]:
        return "No profile data yet."
    lines = []
    if grouped["user"]:
        lines.append("## About the user")
        lines.extend(f"- {c}" for c in grouped["user"])
    if grouped["feedback"]:
        lines.append("\n## Working preferences")
        lines.extend(f"- {c}" for c in grouped["feedback"])
    return "\n".join(lines)


@mcp.resource("engram://recent")
async def recent_resource() -> str:
    db = await get_db()
    memories = await list_memories(db, limit=10)
    if not memories:
        return "No memories yet."
    return "\n".join(
        f"[{m['memory_type']}] ({', '.join(m['tags']) or 'untagged'}) {m['content']}" for m in memories
    )


@mcp.resource("engram://projects")
async def projects_resource() -> str:
    db = await get_db()
    memories = await list_memories(db, memory_type="project", limit=50)
    if not memories:
        return "No project memories yet."
    return "\n".join(
        ["## Projects"] + [f"- **[{', '.join(m['tags']) or 'untagged'}]** {m['content']}" for m in memories]
    )


@mcp.tool()
async def add_memory_tool(
    content: str, source: str, memory_type: str, valid_at: str,
    tags: list[str] = [],
) -> str:
    """Store a single fact as a memory. Use PROACTIVELY when the user shares info worth remembering across sessions.

    All fields required:
    - content: One concise fact. Not a paragraph — one sentence, one fact.
    - source: Lowercase identifier for your tool (e.g. 'claude-code', 'poke', 'cursor').
    - memory_type: One of: user, feedback, project, reference, general.
    - valid_at: When this fact became true (ISO date, e.g. '2026-03-21'), NOT when you're storing it.
    - tags: Short lowercase tags for categorization (e.g. ['seo', 'whop-trends']). Use consistent terms. Can be empty [].

    IMPORTANT: Search before storing to avoid duplicates. If a related memory exists, use update_memory_tool instead."""
    if not source or len(source) > MAX_SOURCE_LENGTH:
        return json.dumps({"error": f"source is required and must be <= {MAX_SOURCE_LENGTH} chars"})
    if memory_type not in VALID_MEMORY_TYPES:
        return json.dumps({"error": f"invalid memory_type '{memory_type}'"})
    if len(content) > MAX_CONTENT_LENGTH:
        return json.dumps({"error": f"content exceeds {MAX_CONTENT_LENGTH} characters"})
    db = await get_db()
    existing_id = await check_duplicate(db, content)
    if existing_id is not None:
        return json.dumps({"id": existing_id, "status": "duplicate"})
    embedding, questions = await _embed_content(content, memory_type)
    return json.dumps(await add_memory(db, content, embedding, tags, memory_type, source, valid_at, questions))


@mcp.tool()
async def search_memories_tool(query: str, tag_filter: str | None = None, limit: int = 5) -> str:
    """Search memories by semantic similarity. Use PROACTIVELY at the start of every conversation and whenever context might exist.

    - query: Descriptive natural language. "user's current job and tech stack" beats "work".
    - tag_filter: Optional. Filter to a specific tag (e.g. 'whop-trends').
    - limit: Number of results (default 5, max 100).

    If the first search misses, rephrase and try again with different terms."""
    db = await get_db()
    embedding = await get_embedding(query)
    return json.dumps(await search_memories(db, embedding, query, tag_filter, limit, rerank_fn=rerank))


@mcp.tool()
async def get_memory_tool(memory_id: int) -> str:
    """Retrieve a specific memory by ID, including its full edit history. Use when you need the complete audit trail of a memory — who changed it, when, and what the previous values were."""
    db = await get_db()
    result = await get_memory(db, memory_id)
    return json.dumps(result or {"error": "not found"})


@mcp.tool()
async def list_memories_tool(
    memory_type: str | None = None, tag: str | None = None, offset: int = 0, limit: int = 20,
) -> str:
    """Browse memories with optional filters. Use when listing/browsing rather than searching by meaning.

    - memory_type: Filter by type (user, feedback, project, reference, general).
    - tag: Filter by a specific tag.
    - offset/limit: Pagination (default 20 per page, max 100)."""
    db = await get_db()
    return json.dumps(await list_memories(db, memory_type, tag, offset, limit))


@mcp.tool()
async def update_memory_tool(
    memory_id: int, source: str, content: str | None = None, tags: list[str] | None = None,
    memory_type: str | None = None, valid_at: str | None = None,
) -> str:
    """Update an existing memory. Use when a fact has changed — ALWAYS search first to find the memory ID.
    The old version is automatically saved in history with source attribution.

    Required:
    - memory_id: ID of the memory to update.
    - source: Lowercase identifier for your tool (e.g. 'claude-code', 'poke', 'cursor').

    Optional (only include what changed):
    - content: New fact text. Write temporal narratives: "Was X, changed to Y in March 2026".
    - tags: Replacement tag list.
    - memory_type: New type if miscategorized.
    - valid_at: When the NEW fact became true (ISO date), not when you're updating."""
    if not source or len(source) > MAX_SOURCE_LENGTH:
        return json.dumps({"error": f"source is required and must be <= {MAX_SOURCE_LENGTH} chars"})
    if memory_type is not None and memory_type not in VALID_MEMORY_TYPES:
        return json.dumps({"error": f"invalid memory_type '{memory_type}'"})
    if content is not None and len(content) > MAX_CONTENT_LENGTH:
        return json.dumps({"error": f"content exceeds {MAX_CONTENT_LENGTH} characters"})
    db = await get_db()
    new_embedding, questions = (await _embed_content(content, memory_type or "general")) if content else (None, None)
    result = await update_memory(db, memory_id, content, tags, memory_type, new_embedding, valid_at, source, questions)
    return json.dumps(result or {"error": "not found"})


@mcp.tool()
async def delete_memory_tool(memory_id: int) -> str:
    """Soft-delete a memory by ID. Use sparingly — prefer updating over deleting. Deleted memories are hidden but not destroyed."""
    db = await get_db()
    return json.dumps({"deleted": await delete_memory(db, memory_id)})


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Route INFO/DEBUG to stdout, WARNING+ to stderr (Railway shows stderr as red)
    for handler, level, filt in [
        (logging.StreamHandler(sys.stdout), logging.DEBUG, lambda r: r.levelno < logging.WARNING),
        (logging.StreamHandler(sys.stderr), logging.WARNING, None),
    ]:
        handler.setLevel(level)
        if filt:
            handler.addFilter(filt)
        logging.root.addHandler(handler)
    logging.root.setLevel(logging.INFO)
    await init_db()
    load_model()
    logger.info("Memory service started")
    async with mcp.session_manager.run():
        yield
    await close_db()
    logger.info("Memory service stopped")


def create_app() -> FastAPI:
    app = FastAPI(lifespan=lifespan)
    app.add_middleware(APIKeyMiddleware)

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    app.mount("/", mcp.streamable_http_app())
    return app


app = create_app()
