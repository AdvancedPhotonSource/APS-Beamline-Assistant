#!/usr/bin/env python3
"""
Hybrid dense + sparse retrieval over the APEXA ChromaDB knowledge base.

Dense (embedding) retrieval is strong on paraphrase and concept, weak on exact
tokens (a specific flag name, a filename, a parameter key). Sparse BM25 is the
mirror image. Fusing their rankings with Reciprocal Rank Fusion (RRF) recovers
both, and — crucially for a beamline host — the sparse side needs **no model and
no network**: it is built directly from the document texts already stored in
Chroma (``collection.get``), tokenized with a stdlib regex. The only third-party
dependency is ``rank-bm25`` (pure-Python, requires just numpy, which is already a
base dep), so this module keeps ``uv sync`` offline-clean.

Design contract:
  * FAIL-OPEN. Any failure (rank-bm25 absent, empty corpus, Chroma quirk) raises
    ``HybridUnavailable``; the caller falls back to the plain dense path, which is
    byte-for-byte the pre-existing behaviour.
  * The ``where`` filter (type / technique scoping) is applied to BOTH candidate
    sets *before* the merge, so hybrid composes cleanly with technique-scoped
    retrieval — it never widens the result set past the filter.
  * A sparse-only hit has no embedding distance, so its ``distance`` is ``None``
    (surfaced as ``similarity: null`` upstream) — never a fabricated 0.0.
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional

# RRF constant. 60 is the value from the original Cormack et al. (2009) paper and
# the community default; large enough that the tail ranks still contribute, small
# enough that the head dominates. Ablation may revisit it.
RRF_K = 60

_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


class HybridUnavailable(RuntimeError):
    """Raised when hybrid retrieval cannot run; caller falls back to dense."""


def _tokenize(text: str) -> List[str]:
    """Lowercase alphanumeric/underscore tokens. Keeps ``ff_midas`` and
    ``RingThresh`` retrievable as whole tokens (underscores kept, case folded)."""
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


def hybrid_search(
    collection,
    query_embedding: List[float],
    query_text: str,
    n_results: int = 3,
    where_filter: Optional[dict] = None,
    k: int = RRF_K,
    candidate_pool: Optional[int] = None,
) -> List[Dict]:
    """Return up to ``n_results`` fused hits, most-relevant first.

    Each hit: ``{"document", "metadata", "distance"|None, "dense_rank"|None,
    "sparse_rank"|None, "rrf"}``. ``distance`` is the dense cosine distance when
    the doc was a dense hit, else ``None`` (sparse-only).

    Raises ``HybridUnavailable`` on any condition that should trigger the dense
    fallback (missing rank-bm25, empty corpus, Chroma error).
    """
    try:
        from rank_bm25 import BM25Okapi
    except ImportError as e:  # pragma: no cover - environment-dependent
        raise HybridUnavailable(f"rank-bm25 not installed: {e}") from e

    if n_results < 1:
        return []

    # Pull the sparse corpus: every doc matching the same filter as the dense
    # side. No embedder, no network — just the stored text.
    try:
        corpus = collection.get(
            where=where_filter,
            include=["documents", "metadatas"],
        )
    except Exception as e:
        raise HybridUnavailable(f"collection.get failed: {e}") from e

    corpus_ids = corpus.get("ids") or []
    corpus_docs = corpus.get("documents") or []
    corpus_metas = corpus.get("metadatas") or []
    if not corpus_ids or not corpus_docs:
        raise HybridUnavailable("empty corpus for the given filter")

    # Candidate pool from the dense side: over-fetch so the fusion has room to
    # re-rank. Bounded by the corpus size.
    pool = candidate_pool or max(n_results * 10, 50)
    pool = min(pool, len(corpus_ids))

    try:
        dense = collection.query(
            query_embeddings=[query_embedding],
            n_results=pool,
            where=where_filter,
        )
    except Exception as e:
        raise HybridUnavailable(f"dense query failed: {e}") from e

    dense_ids = (dense.get("ids") or [[]])[0]
    dense_docs = (dense.get("documents") or [[]])[0]
    dense_metas = (dense.get("metadatas") or [[]])[0]
    dense_dists = (dense.get("distances") or [[]])[0]

    # id -> unified record. Seed from the corpus so sparse-only hits have text.
    records: Dict[str, Dict] = {}
    for cid, doc, meta in zip(corpus_ids, corpus_docs, corpus_metas):
        records[cid] = {
            "id": cid, "document": doc, "metadata": meta,
            "distance": None, "dense_rank": None, "sparse_rank": None,
        }

    # Dense ranks (1-indexed).
    for rank, (cid, doc, meta, dist) in enumerate(
        zip(dense_ids, dense_docs, dense_metas, dense_dists), start=1
    ):
        rec = records.get(cid)
        if rec is None:  # dense returned an id not in the corpus.get set (rare); add it
            rec = {"id": cid, "document": doc, "metadata": meta,
                   "distance": None, "dense_rank": None, "sparse_rank": None}
            records[cid] = rec
        rec["dense_rank"] = rank
        rec["distance"] = dist

    # Sparse ranks (1-indexed) over the whole filtered corpus.
    tokenized_corpus = [_tokenize(d) for d in corpus_docs]
    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(_tokenize(query_text))
    order = sorted(range(len(corpus_ids)), key=lambda i: scores[i], reverse=True)
    for rank, idx in enumerate(order[:pool], start=1):
        # Only count as a sparse hit if the score is positive (query shares a
        # token with the doc); a 0 score is not evidence and would just add noise.
        if scores[idx] <= 0:
            continue
        records[corpus_ids[idx]]["sparse_rank"] = rank

    # RRF fuse.
    fused = []
    for rec in records.values():
        if rec["dense_rank"] is None and rec["sparse_rank"] is None:
            continue
        rrf = 0.0
        if rec["dense_rank"] is not None:
            rrf += 1.0 / (k + rec["dense_rank"])
        if rec["sparse_rank"] is not None:
            rrf += 1.0 / (k + rec["sparse_rank"])
        rec["rrf"] = rrf
        fused.append(rec)

    if not fused:
        raise HybridUnavailable("no candidates from either retriever")

    # Sort by RRF desc; tie-break by dense rank (embedding relevance) then id for
    # determinism.
    fused.sort(key=lambda r: (
        -r["rrf"],
        r["dense_rank"] if r["dense_rank"] is not None else 1e9,
        r["id"],
    ))
    return fused[:n_results]
