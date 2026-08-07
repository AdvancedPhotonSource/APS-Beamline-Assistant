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
| `chroma_db/` | ✅ | the **prebuilt index (~15 MB) ships in git** — ready to query on checkout, no re-index needed. Only the active segment + `chroma.sqlite3` are tracked. |

## Deploy step (after `git pull`)

The prebuilt index now ships in git, so **online machines need no deploy step** —
the RAG works on checkout (first query downloads + caches the ~523 MB Nomic model).

**Re-index only when you change a source document:**

```bash
uv run python knowledge_base/index_knowledge.py   # then commit the updated chroma_db/
```

On success it prints `Indexing complete` and writes `data/index_stats.json`.

**Air-gapped / no-internet machine:** the embedding model must be pre-staged and
`APEXA_OFFLINE=1` set (both index- and query-time load only from the local cache).
See [`../docs/OFFLINE_DEPLOYMENT.md`](../docs/OFFLINE_DEPLOYMENT.md).

To use a different embedding model, set `APEXA_EMBED_MODEL` before indexing **and**
at query time (both must match, or retrieval returns garbage):

```bash
export APEXA_EMBED_MODEL=nvidia/llama-embed-nemotron-8b
```

## Adding open MIDAS docs

Copy the `.md` into `logbooks/`, add a matching `!knowledge_base/logbooks/<name>.md`
negation line in `.gitignore`, then re-run the indexer. Keep proprietary beamtime
notes out of git — leave them as plain `logbooks/*.md`/`*.txt` (already ignored).
