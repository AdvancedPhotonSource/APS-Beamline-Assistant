#!/usr/bin/env python3
"""Retrieval ablation: dense vs sparse (BM25) vs hybrid (RRF) on the kno_* set.

Decides whether ``retrieval_mode="hybrid"`` should become the default for
``query_hedm_knowledge``. Ship hybrid as the default ONLY if it is >= max(dense,
sparse) on mean hit@k over the knowledge questions; otherwise keep ``dense``.

Scoring is keyword-recall: a question is a "hit@k" if any of the top-k retrieved
excerpts contains any of the question's ``expected_keywords_any`` (case-insensitive).
This is a coarse but honest proxy — it measures whether the right material was
surfaced, independent of the LLM's phrasing.

Usage:
    APEXA_OFFLINE=1 uv run python scripts/retrieval_ablation.py [--k 3]

Offline-clean: uses only the local Chroma DB and the (pre-staged) embedder; no
network. Requires the KB to be indexed first (knowledge_base/index_knowledge.py).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "knowledge_base"))

os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
os.environ.setdefault("CHROMA_TELEMETRY_ENABLED", "False")


def _load_kno_tasks() -> list:
    tasks_file = _REPO / "benchmark" / "benchmark_tasks.json"
    data = json.loads(tasks_file.read_text())
    tasks = data["tasks"] if isinstance(data, dict) else data
    return [t for t in tasks if str(t.get("id", "")).startswith("kno_")]


def _hit(excerpt_texts: list, keywords: list) -> bool:
    blob = " ".join(excerpt_texts).lower()
    return any(kw.lower() in blob for kw in (keywords or []))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=3, help="top-k excerpts to score (default 3)")
    args = ap.parse_args()
    k = args.k

    try:
        import chromadb
        import index_knowledge as ik
        import hybrid_retrieval as hyb
    except Exception as e:
        print(f"❌ dependencies unavailable: {e}")
        return 2

    chroma_path = _REPO / "knowledge_base" / "chroma_db"
    if not chroma_path.exists():
        print("❌ knowledge base not indexed. Run: uv run python knowledge_base/index_knowledge.py")
        return 2

    embedder = ik._load_embedder()
    client = chromadb.PersistentClient(path=str(chroma_path))
    collection = client.get_collection(name="hedm_knowledge")

    # Sparse-only: BM25 over the full corpus (no filter), ranked by score.
    # BM25Okapi is vendored in hybrid_retrieval (numpy-only) — no PyPI dep.
    from hybrid_retrieval import BM25Okapi
    corpus = collection.get(include=["documents", "metadatas"])
    corpus_docs = corpus["documents"]
    bm25 = BM25Okapi([hyb._tokenize(d) for d in corpus_docs])

    tasks = _load_kno_tasks()
    modes = {"dense": 0, "sparse": 0, "hybrid": 0}
    per_task = []

    for t in tasks:
        q = t["query"]
        kws = t.get("expected_keywords_any", [])
        emb = ik.embed_query(embedder, q)

        # dense
        d = collection.query(query_embeddings=[emb], n_results=k)
        dense_texts = d["documents"][0]

        # sparse
        scores = bm25.get_scores(hyb._tokenize(q))
        order = sorted(range(len(corpus_docs)), key=lambda i: scores[i], reverse=True)[:k]
        sparse_texts = [corpus_docs[i] for i in order]

        # hybrid
        try:
            fused = hyb.hybrid_search(collection, emb, q, n_results=k)
            hybrid_texts = [r["document"] for r in fused]
        except hyb.HybridUnavailable:
            hybrid_texts = dense_texts

        row = {
            "id": t["id"],
            "dense": _hit(dense_texts, kws),
            "sparse": _hit(sparse_texts, kws),
            "hybrid": _hit(hybrid_texts, kws),
        }
        for m in modes:
            modes[m] += 1 if row[m] else 0
        per_task.append(row)

    n = len(tasks)
    print(f"\nRetrieval ablation over {n} knowledge questions (hit@{k}):\n")
    print(f"{'task':10} {'dense':>7} {'sparse':>7} {'hybrid':>7}")
    for row in per_task:
        print(f"{row['id']:10} {str(row['dense']):>7} {str(row['sparse']):>7} {str(row['hybrid']):>7}")
    print("-" * 34)
    means = {m: modes[m] / n for m in modes}
    print(f"{'mean':10} {means['dense']:>7.2f} {means['sparse']:>7.2f} {means['hybrid']:>7.2f}")

    # Recommendation: hybrid only if it does not lose to the best single retriever.
    best_single = max(means["dense"], means["sparse"])
    if means["hybrid"] >= best_single and means["hybrid"] >= means["dense"]:
        rec = "hybrid"
    else:
        rec = "dense"
    print(f"\nRecommended default retrieval_mode: {rec}")
    print("(ship hybrid as default only if it does not lose to dense/sparse on this set)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
