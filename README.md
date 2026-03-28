# Engram

Personal memory service for AI tools. Stores facts about you and retrieves them when your AI needs context — across Claude Code, Cursor, Poke, Windsurf, or any MCP client.

Built on [Model Context Protocol (MCP)](https://modelcontextprotocol.io/) with hybrid search (BM25 + vector + Reverse HyDE), cross-encoder reranking, and temporal versioning.

## Prerequisites

- **OpenAI API key** — used for embeddings (`text-embedding-3-small`) and question generation
- **Python 3.12+** (for local dev) or **Docker**
- **Railway account** (optional, for deployment)

## Quick Start

```bash
git clone https://github.com/namanxajmera/engram.git
cd engram
uv venv && uv pip install -r requirements.txt
MEMORY_API_KEY=your-secret-key OPENAI_API_KEY=sk-... uvicorn src.server:app --reload
```

The reranker model (~23MB) downloads automatically on first run.

## Connect Your AI Tools

**Claude Code:**
```bash
claude mcp add engram https://your-engram-url/mcp \
  -t http -s user \
  -H "Authorization: Bearer YOUR_API_KEY"
```

**Poke:**
```bash
npx poke@latest mcp add https://your-engram-url/mcp \
  -n "Engram" \
  -k "YOUR_API_KEY"
```

**Any MCP client:** Point at your `/mcp` endpoint with a Bearer token in the Authorization header.

## How It Works

### Writes (add/update)

```
Content → OpenAI embedding → Store in SQLite → Return immediately
                                   ↓ (background)
                            LLM generates 3-5 questions this fact answers
                            → Embed questions → Store for Reverse HyDE search
```

Writes return in ~1-2s (just the content embedding). Question indexing happens async in the background.

### Reads (search)

```
Query → OpenAI embedding → 3 parallel searches:
                             ├─ Vector similarity (content embeddings)
                             ├─ BM25 keyword match (FTS5)
                             └─ Reverse HyDE (question embeddings)
                           → Reciprocal Rank Fusion
                           → Cross-encoder reranking
                           → Top results
```

Repeated queries hit the embedding cache (1000 entries, 7-day TTL) and skip the OpenAI round-trip.

## MCP Resources (passive context)

Clients pull these automatically — no tool call needed.

| Resource | Description |
|---|---|
| `engram://profile` | User identity, preferences, working style |
| `engram://recent` | 10 most recently updated memories |
| `engram://projects` | All project context |

## MCP Tools

| Tool | Required Fields | Description |
|---|---|---|
| `add_memory_tool` | content, source, memory_type, valid_at, tags | Store a fact |
| `search_memories_tool` | query | Hybrid search with reranking |
| `get_memory_tool` | memory_id | Get memory by ID with full edit history |
| `list_memories_tool` | _(none)_ | Browse/filter by type or tag |
| `update_memory_tool` | memory_id, source | Update a memory (old version saved to history) |
| `delete_memory_tool` | memory_id | Soft delete |

**Memory types:** `user`, `feedback`, `project`, `reference`, `general`

**Source:** Required on writes. Lowercase tool identifier (e.g. `claude-code`, `cursor`, `poke`).

## Memory Schema

| Field | Description |
|---|---|
| `content` | One concise fact per memory |
| `tags` | JSON array for filtering |
| `memory_type` | Category (user, feedback, project, reference, general) |
| `source` | Which tool stored it |
| `valid_at` | When this fact became true (not when stored) |
| `invalid_at` | When superseded (null = still current) |
| `history` | Previous versions with source, content, and date ranges |

### Temporal Versioning

Updates push the old version to a history array with source attribution and date ranges. Nothing is lost — any tool can see what changed and when.

## Deploy

### Railway

```bash
railway init
railway up
```

Then in the Railway dashboard:
1. Add a **persistent volume** mounted at `/data`
2. Set environment variables: `MEMORY_API_KEY`, `OPENAI_API_KEY`
3. Optionally set a custom domain

### Docker

```bash
docker build -t engram .
docker run -p 8000:8000 \
  -e MEMORY_API_KEY=your-secret-key \
  -e OPENAI_API_KEY=sk-... \
  -v engram-data:/data \
  engram
```

### Env Vars

| Variable | Required | Description |
|---|---|---|
| `MEMORY_API_KEY` | Yes | Bearer token for auth. Unset = 503 (fail-closed). |
| `OPENAI_API_KEY` | Yes | For embeddings and question generation. |
| `ENGRAM_QUESTION_MODEL` | No | LLM for Reverse HyDE questions (default: `gpt-5.4-nano`). |
| `PORT` | No | Server port (default: 8000). |

## Stack

- Python 3.12 / FastAPI / uvicorn
- MCP SDK (Streamable HTTP transport)
- SQLite + sqlite-vec (vector search) + FTS5 (BM25) + WAL mode
- OpenAI `text-embedding-3-small` (1536-dim, ~$0.02/M tokens)
- ONNX Runtime `ms-marco-MiniLM-L-6-v2` (local cross-encoder reranker, ~23MB)

## Development

```bash
uvx ruff check src/     # lint
uvx ruff format src/    # format
```

## Agent Behavior

Engram's tool descriptions guide AI agents to:

- **Search proactively** at conversation start
- **Store proactively** when they learn facts worth remembering
- **Update over delete** to preserve history
- **Search before storing** to avoid duplicates
- **Attribute** every write with their `source` identifier

For best results, reinforce in your client's system prompt:

```markdown
You have access to Engram, a shared memory service via MCP. Use it proactively:
- Search at the start of every conversation for context
- Store when you learn something worth remembering across sessions
- Always set source to "your-tool-name" and valid_at to when the fact became true
- Search before storing to avoid duplicates — update existing memories instead
```

## License

MIT
