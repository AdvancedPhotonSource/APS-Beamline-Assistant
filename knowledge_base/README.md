# APEXA Knowledge Base (RAG)

Citation-aware retrieval that backs the `KnowledgeAgent` and the
`query_hedm_knowledge` MCP tool. Content is embedded with Nomic
(`nomic-ai/nomic-embed-text-v1.5`, cosine) into a local ChromaDB collection
(`hedm_knowledge`).

## What ships in git

| Path | Tracked | Notes |
|---|---|---|
| `index_knowledge.py`, `fetch_materials_from_mp.py`, `sync_from_zotero.py` | ✅ | indexer + helpers |
| `data/*.json` | ✅ | index stats, materials/params templates |
| `papers/*.bib` | ✅ | citation sidecars (the `*.pdf` are **not** shipped — copyrighted) |
| `logbooks/*.md` (the 15 open MIDAS docs) | ✅ | FF/NF handbooks + lab notebooks, `Reconstruction_Reports`, and the analysis/calibration docs — un-ignored explicitly in `.gitignore` |
| `logbooks/*.txt`, other `logbooks/*.md` | ❌ | drop-zone for **private** beamtime notes — gitignored by default |
| `papers/*.pdf`, `books/*.pdf` | ❌ | copyrighted / large |
| `chroma_db/` | ❌ | the built index — **rebuilt on deploy** (see below) |

## Deploy step (run once per machine, after `git pull`)

The index (`chroma_db/`) is not committed, so build it after checkout:

```bash
uv run python knowledge_base/index_knowledge.py
```

First run downloads the Nomic embedding model (~once, cached) and embeds all
`papers/` + `logbooks/` + `books/` content — a few minutes on CPU. Re-run it any
time you add or change a source document. On success it prints
`Indexing complete` and writes `data/index_stats.json`.

To use a different embedding model, set `APEXA_EMBED_MODEL` before indexing **and**
at query time (both must match, or retrieval returns garbage):

```bash
export APEXA_EMBED_MODEL=nvidia/llama-embed-nemotron-8b
```

## Adding open MIDAS docs

Copy the `.md` into `logbooks/`, add a matching `!knowledge_base/logbooks/<name>.md`
negation line in `.gitignore`, then re-run the indexer. Keep proprietary beamtime
notes out of git — leave them as plain `logbooks/*.md`/`*.txt` (already ignored).
