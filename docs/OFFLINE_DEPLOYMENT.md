# Offline / air-gapped beamline deployment

For a beamline machine with **no internet access** (or a locked-down network that
can't reach `huggingface.co`). Covers the two things that don't "just work" on a
fresh checkout: the **knowledge base (RAG)** embedding model, and **data that
lives on another machine**.

A machine *with* internet needs none of this — the normal flow
(`git pull` → `uv sync` → `index_knowledge.py` on first run) downloads the model
once and caches it. This doc is only for the offline case.

---

## 0. Dependencies & network tier — the base install is offline-clean

`uv sync` **with no extras has no public-internet dependency.** Everything needed
for data reduction and RAG installs and runs on an air-gapped host:

- The RAG stack (`chromadb` + `sentence-transformers`) is a **base** dependency, so
  retrieval works offline — it only needs the network for the *first* embedder
  download, which you pre-stage (§1). It is not gated behind an extra.
- **ChromaDB telemetry is disabled** before `chromadb` is imported, on every code
  path (runtime `get_knowledge_base`, the `index_knowledge.py` re-index script, and
  `apexa_network.apply_offline_env`). So no anonymised PostHog call is attempted —
  one fewer thing that can block or leak from a beamline host.
- The **only** online-only feature is Materials Project CIF fetch
  (`fetch_cif_from_mp`). Its `mp-api` package (and the large pymatgen tree it drags
  in) lives in a **separate optional extra**, deliberately kept out of the base:

  ```bash
  uv sync              # offline-clean: NO mp-api, NO public-internet dep
  uv sync --extra mp   # ONLY on a web-capable host — adds Materials Project CIF fetch
  ```

  On an internal/offline host, omit `--extra mp`. The tool is hidden below the
  `web` tier and, if a non-standard client reaches it anyway, returns a structured
  "unavailable at this tier" error — it never hangs on the network.

- **tiktoken is tier-gated too.** `tiktoken.get_encoding` downloads its BPE vocab
  from a public URL on first use; on an air-gapped host that download BLOCKS (a
  hang, not an exception — it wedged APEXA before its first LLM call). Below the
  `web` tier APEXA now skips it and estimates token counts as `len//4`
  (instrumentation-only, never load-bearing). If you have staged the tiktoken
  cache (or set `TIKTOKEN_CACHE_DIR`), set `APEXA_TIKTOKEN_OK=1` to re-enable exact
  counts.

Set the tier so the runtime never *tries* the public internet (a network hang is
not catchable — the tier prevents the attempt):

```bash
APEXA_NETWORK=internal   # DEFAULT — ANL-internal only; HuggingFace forced offline
# APEXA_OFFLINE=1 remains supported as a legacy alias and caps the tier at internal.
```

`docs/setup_user.sh` probes reachability and writes `APEXA_NETWORK` into `.env` for
you. See the CLAUDE.md "Network tiers" section for the full model.

---

## 1. Knowledge base (FF/NF handbooks + notebooks)

### What ships in git ✅
| Artifact | In git | Notes |
|---|---|---|
| Source docs (15) | ✅ | all FF/NF handbooks + lab notebooks + analysis/calibration docs, under `knowledge_base/logbooks/*.md` |
| **Prebuilt index** (`knowledge_base/chroma_db/`, ~15 MB) | ✅ | 522 chunks, ready to query on checkout — **no re-index step needed** |
| Indexer + helpers | ✅ | `index_knowledge.py`, index stats |

### The one thing NOT in git ❌ — the embedding model
Retrieval embeds **every query** at runtime with Nomic
(`nomic-ai/nomic-embed-text-v1.5`, ~523 MB). The prebuilt index removes the
*re-index* step, but the model is still needed **at query time** — there is no way
around having it present locally. It is not in git (too large, and it's an upstream
artifact). On an air-gapped machine you must **pre-stage** it.

### Pre-stage the model (do this once, from an internet-connected machine)

The model lives in the HuggingFace cache. Copy the whole cache tree to the beamline
machine's home directory:

```bash
# On a machine that HAS internet (e.g. your laptop), ensure the model is cached:
uv run python -c "from sentence_transformers import SentenceTransformer as S; S('nomic-ai/nomic-embed-text-v1.5', trust_remote_code=True)"

# Copy these three cache entries to the beamline machine's ~/.cache/huggingface/:
#   hub/models--nomic-ai--nomic-embed-text-v1.5     (~523 MB)
#   hub/models--nomic-ai--nomic-bert-2048           (~112 KB — the config/tokenizer repo)
#   modules/transformers_modules                    (~236 KB — trust_remote_code code)
rsync -a --info=progress2 \
  ~/.cache/huggingface/hub/models--nomic-ai--nomic-embed-text-v1.5 \
  ~/.cache/huggingface/hub/models--nomic-ai--nomic-bert-2048 \
  <beamline-host>:~/.cache/huggingface/hub/

rsync -a ~/.cache/huggingface/modules/transformers_modules \
  <beamline-host>:~/.cache/huggingface/modules/
```

(No ssh between the two? Copy to a USB drive / mounted share instead — the target
layout is all that matters: `~/.cache/huggingface/hub/models--nomic-ai--...`.)

### Turn on offline mode

Add to `.env` (or export before launching APEXA):

```bash
APEXA_OFFLINE=1
```

This makes APEXA set `HF_HUB_OFFLINE=1` + `TRANSFORMERS_OFFLINE=1` and load the
model with `local_files_only=True`, so it reads the pre-staged cache and **never
touches the network** — no hang, no failed HEAD request. Applies to both query time
(`midas_comprehensive_server.py`) and re-index (`index_knowledge.py`).

### Verify

```bash
APEXA_OFFLINE=1 uv run python knowledge_base/index_knowledge.py   # if re-indexing
# or just run APEXA and ask a MIDAS question:
#   "How do I set RingThresh for an FF-HEDM run?"
# → should cite FF_HEDM_Handbook.md
```

If you see a network/HEAD error, the model isn't fully staged — re-check the three
cache paths above.

### Using a different embed model
Set `APEXA_EMBED_MODEL` **at both index and query time** (they must match, or
retrieval returns garbage), pre-stage that model's cache the same way, and re-index:
`APEXA_OFFLINE=1 APEXA_EMBED_MODEL=<model> uv run python knowledge_base/index_knowledge.py`.

---

## 2. Data on another machine (e.g. `copland`)

There are two ways to work with data that lives on another host (e.g. data on
`copland` while APEXA runs on `chiltepin`):

### Run analysis remotely over SSH — `run_remote_command` (recommended for copland)

APEXA can run commands **on the remote host where the data already is**, via the
`run_remote_command` core tool — no copy needed:

```
ssh copland 'cd /gdata/dm/1ID/2026/pokharel_jul26 && ff_MIDAS.py -paramFile ff.txt'
```

- Set `APEXA_ANALYSIS_HOST=copland` in `.env` (or pass `host=` per call).
- **Requires key-based SSH** from the APEXA host: `ssh-copy-id copland`, then
  confirm `ssh copland true` returns with no password prompt. APEXA runs
  non-interactively, so a password prompt fails fast (rc=255) with a hint rather
  than hanging.
- Runs through a remote **login shell** so the remote MIDAS env is sourced; the
  same command allowlist as local `run_command` applies.
- NOTE: `run_remote_command` is the agent driving the MIDAS CLI **by hand** on the
  remote host (generic, unguarded — the handbook lint gate does not run on it). A
  few *typed* MIDAS tools are now themselves SSH-routed to the host that owns the
  data (`calibrate_ring_thresholds`, `run_ff_hedm_full_workflow`) — prefer the typed
  tool when one exists, so the lint gate + output verifier still apply. Most typed
  tools still execute locally. See the CLAUDE.md "Remote execution / topology"
  section and `remote_hosts.example.json`.

### Or make the data local — mount / stage

To use the typed MIDAS tools against remote data, make it appear as a **local
path**, then point APEXA at that path:

**Option A — mount it (preferred; no copy, no extra disk):**
```bash
# NFS (if copland exports the share and this host can mount it):
sudo mount -t nfs copland:/data/pokharel_jul26 /mnt/copland_data

# or sshfs (userspace, needs ssh access + macFUSE/fuse):
sshfs user@copland:/data/pokharel_jul26 /mnt/copland_data
```
Then in APEXA: `integrate /mnt/copland_data/...`.

**Option B — stage it locally (when there's no mount, or for speed):**
```bash
rsync -a --info=progress2 user@copland:/data/pokharel_jul26/ /scratch/pokharel_jul26/
```
This matches the **data-locality** rule in `COMPUTE_DISPATCH.md`: put the compute
where it can already see the data (`/scratch`), rather than reaching across the
network per file.

> A built-in remote-data capability (an APEXA-managed mount/stage helper) is being
> scoped — see the remote-data feature notes. Until it lands, use the mount/stage
> workflow above.

---

## Quick checklist for an air-gapped node

- [ ] `git pull` — brings source docs **and** the prebuilt `chroma_db/`
- [ ] `uv sync` — base install is offline-clean (do **not** add `--extra mp`; §0)
- [ ] Pre-stage the Nomic model cache (§1) — the only large offline artifact
- [ ] `APEXA_NETWORK=internal` (or `APEXA_OFFLINE=1`) in `.env`
- [ ] Mount or stage any off-machine data to a local path — or SSH-route it (§2)
- [ ] Smoke-test: one MIDAS knowledge question + one file listing on the data path
