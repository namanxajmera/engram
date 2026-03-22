# Engram

Personal memory service — a shared context layer across AI tools (Claude Code, Poke, Cursor, etc).

Stores, searches, and retrieves memories via Streamable HTTP MCP. Uses Reverse HyDE for high-quality retrieval on vague queries. Temporal versioning tracks how facts change over time. Deploy anywhere.

## Connect

**Claude Code:**
```bash
claude mcp add engram https://your-engram-url.up.railway.app/mcp \
  -t http -s user \
  -H "Authorization: Bearer YOUR_API_KEY"
```

**Poke:**
```bash
npx poke@latest mcp add https://your-engram-url.up.railway.app/mcp \
  -n "Engram" \
  -k "YOUR_API_KEY"
```

**Any MCP client:**
Point it at your `/mcp` endpoint with a Bearer token in the Authorization header.

## MCP Resources (passive context)

Resources are read-only data that clients can pull automatically — no tool call needed. The model gets context before the user even asks.

| Resource | Description |
|---|---|
| `engram://profile` | User profile — who they are, preferences, working style |
| `engram://recent` | 10 most recently added/updated memories |
| `engram://projects` | Overview of all projects the user has worked on |

## MCP Tools (model-controlled)

Tools the AI model can call to read and write memories. Descriptions include trigger conditions that guide agents to use them proactively.

| Tool | Required Fields | Description |
|---|---|---|
| `add_memory_tool` | content, source, memory_type, valid_at, tags | Store a fact. Agents are prompted to use this proactively. |
| `search_memories_tool` | query | Hybrid search. Agents are prompted to search at conversation start. |
| `get_memory_tool` | memory_id | Get memory by ID with full edit history and source attribution. |
| `list_memories_tool` | _(none)_ | Browse/filter memories by type or tag. |
| `update_memory_tool` | memory_id, source | Update a memory. Old version auto-saved to history with source. |
| `delete_memory_tool` | memory_id | Soft delete. Agents are guided to prefer updates over deletes. |

**Memory types:** `user`, `feedback`, `project`, `reference`, `general`
**Source:** Required on writes/updates. Lowercase tool identifier (e.g. `claude-code`, `poke`, `cursor`). Any string accepted — not restricted to a fixed list.

## Memory Schema

Each memory stores:

| Field | Description |
|---|---|
| `content` | The fact itself — one concise fact per memory |
| `tags` | JSON array of tags for filtering |
| `memory_type` | Category (user, feedback, project, reference, general) |
| `source` | Which tool stored it (claude-code, poke, etc) |
| `valid_at` | When this fact became true (not when it was stored) |
| `invalid_at` | When it was superseded (null = still current) |
| `history` | Previous versions with source, content, and date ranges (auto-tracked on update) |

### Temporal Versioning

Engram never deletes facts. When a memory is updated, the old version is automatically pushed to a history array with its `source`, `valid_at`, and `invalid_at` dates. This preserves the full arc of how facts change over time — including which AI tool made each change.

Example: updating "Financial goal is $10M" to "Financial goal revised to $5M" keeps both versions — so any tool can see the user originally targeted $10M and when they changed their mind.

## Search Pipeline

Engram uses a 4-stage search pipeline for high-quality retrieval:

1. **Triple retrieval** — BM25 keyword search (FTS5) + content vector search + Reverse HyDE question vector search, all in parallel
2. **Reciprocal Rank Fusion** — combines all three ranked lists into a single ordering
3. **Cross-encoder reranking** — top candidates re-scored by a cross-encoder that reads query + document together
4. **Reverse HyDE** — at storage time, generates 3-5 hypothetical questions each memory could answer and embeds them. A vague query like "work" matches the pre-generated "What does the user do for work?" embedding directly

This means vague queries find the right memory even when there's no keyword or semantic overlap with the stored fact.

## Stack

- Python 3.12 / FastAPI / uvicorn
- MCP SDK (streamable-http transport)
- SQLite + sqlite-vec (vector search) with WAL mode
- FTS5 (BM25 keyword search)
- OpenAI `text-embedding-3-small` (1536-dim embeddings, ~$0.02/M tokens)
- ONNX Runtime `ms-marco-MiniLM-L-6-v2` (local cross-encoder reranker, ~23MB quantized)
- Railway (persistent volume at `/data`)

## Setup

### Local
```bash
uv venv && uv pip install -r requirements.txt
MEMORY_API_KEY=your-secret-key uvicorn src.server:app --reload
```

The reranker ONNX model (~23MB) downloads automatically on first run to `./data/model/`.

### Railway
```bash
railway init
railway up
```
Add a persistent volume mounted at `/data` and set `MEMORY_API_KEY` + `OPENAI_API_KEY` in your service variables.

### Docker
```bash
docker build -t engram .
docker run -p 8000:8000 -e MEMORY_API_KEY=your-secret-key -e OPENAI_API_KEY=your-openai-key -v engram-data:/data engram
```

### Lint & Format
```bash
uvx ruff check src/     # lint
uvx ruff format src/    # format
```

### Env Vars

| Variable | Required | Description |
|---|---|---|
| `MEMORY_API_KEY` | Yes | Bearer token for auth. If unset, service returns 503 (fail-closed). |
| `OPENAI_API_KEY` | Yes | For embeddings (`text-embedding-3-small`) and question generation. |
| `ENGRAM_QUESTION_MODEL` | No | LLM for Reverse HyDE question generation (default: `gpt-5.4-nano`). |
| `PORT` | No | Server port (default: 8000) |

## Agent Behavior

Engram's MCP server includes detailed instructions and tool descriptions that guide AI agents to:

- **Search proactively** at the start of every conversation for user context
- **Store proactively** when the user shares facts, preferences, or completes significant work
- **Update over delete** — preserve temporal history of how facts change
- **Search before storing** to avoid duplicates
- **Use consistent tags** with lowercase terms (e.g. `seo`, `tech-stack`, `metrics`)
- **Always attribute** writes/updates with their tool's `source` identifier

These behaviors are embedded in the server instructions and each tool's description. Most MCP clients will follow them natively. For best results, also reinforce in your client's system prompt:

```markdown
You have access to Engram, a shared memory service via MCP. Use it proactively:
- Search at the start of every conversation for context
- Store when you learn something worth remembering across sessions
- Always set source to "your-tool-name" and valid_at to when the fact became true
- Search before storing to avoid duplicates — update existing memories instead
```
