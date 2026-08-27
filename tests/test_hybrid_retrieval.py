"""Phase 3c — hybrid dense+sparse retrieval (`knowledge_base/hybrid_retrieval.py`).

Uses a fake Chroma-like collection (no embedder, no network) so the RRF fusion,
technique/type filter composition, sparse-only null-distance handling, and
fail-open contract are all exercised deterministically.
"""
import importlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "knowledge_base"))
h = importlib.import_module("hybrid_retrieval")


class FakeCollection:
    """Minimal stand-in for a Chroma collection.

    - .get(where, include) returns all docs matching a simple equality `where`.
    - .query(query_embeddings, n_results, where) ranks by a caller-supplied
      dense order (ids listed in `dense_order`, best first).
    """

    def __init__(self, docs, dense_order):
        # docs: list of (id, text, meta)
        self._docs = docs
        self._dense_order = dense_order

    def _match(self, meta, where):
        if not where:
            return True
        if "$and" in where:
            return all(self._match(meta, c) for c in where["$and"])
        (k, v), = where.items()
        return meta.get(k) == v

    def get(self, where=None, include=None):
        ids, documents, metadatas = [], [], []
        for _id, text, meta in self._docs:
            if self._match(meta, where):
                ids.append(_id); documents.append(text); metadatas.append(meta)
        return {"ids": ids, "documents": documents, "metadatas": metadatas}

    def query(self, query_embeddings=None, n_results=3, where=None):
        by_id = {d[0]: d for d in self._docs}
        ordered = [by_id[i] for i in self._dense_order
                   if i in by_id and self._match(by_id[i][2], where)]
        ordered = ordered[:n_results]
        return {
            "ids": [[d[0] for d in ordered]],
            "documents": [[d[1] for d in ordered]],
            "metadatas": [[d[2] for d in ordered]],
            "distances": [[0.1 * (r + 1) for r in range(len(ordered))]],
        }


def _sample_docs():
    return [
        ("a", "grain indexing completeness confidence tolerance", {"type": "capsule", "technique": "ff-hedm"}),
        ("b", "RingThresh per-ring threshold recommendation", {"type": "capsule", "technique": "ff-hedm"}),
        ("c", "near-field forward simulation mic reconstruction", {"type": "capsule", "technique": "nf-hedm"}),
        ("d", "powder calibration ceria ring pattern", {"type": "paper", "technique": ""}),
    ]


def test_hybrid_fuses_and_returns_records():
    coll = FakeCollection(_sample_docs(), dense_order=["a", "b", "c", "d"])
    out = h.hybrid_search(coll, [0.0], "RingThresh threshold", n_results=3)
    assert out, "expected fused hits"
    # 'b' mentions RingThresh/threshold → sparse should surface it strongly.
    ids = [r["id"] for r in out]
    assert "b" in ids
    for r in out:
        assert "rrf" in r
        assert r["dense_rank"] is not None or r["sparse_rank"] is not None


def test_technique_filter_scopes_both_retrievers():
    coll = FakeCollection(_sample_docs(), dense_order=["a", "b", "c", "d"])
    out = h.hybrid_search(
        coll, [0.0], "reconstruction", n_results=5,
        where_filter={"technique": "nf-hedm"},
    )
    assert out
    assert all(r["metadata"]["technique"] == "nf-hedm" for r in out)


def test_and_filter_composes():
    coll = FakeCollection(_sample_docs(), dense_order=["a", "b", "c", "d"])
    out = h.hybrid_search(
        coll, [0.0], "grain", n_results=5,
        where_filter={"$and": [{"type": "capsule"}, {"technique": "ff-hedm"}]},
    )
    assert out
    for r in out:
        assert r["metadata"]["type"] == "capsule"
        assert r["metadata"]["technique"] == "ff-hedm"


def test_sparse_only_hit_has_null_distance():
    # Query token appears only in a doc the dense side omits → sparse-only, so its
    # distance must be None (never a fabricated 0.0).
    coll = FakeCollection(_sample_docs(), dense_order=["a"])  # dense returns only 'a'
    out = h.hybrid_search(coll, [0.0], "powder ceria", n_results=5)
    sparse_only = [r for r in out if r["dense_rank"] is None]
    assert sparse_only, "expected at least one sparse-only hit"
    assert all(r["distance"] is None for r in sparse_only)


def test_empty_corpus_raises_unavailable():
    coll = FakeCollection([], dense_order=[])
    with pytest.raises(h.HybridUnavailable):
        h.hybrid_search(coll, [0.0], "anything", n_results=3)


def test_no_matching_filter_raises_unavailable():
    coll = FakeCollection(_sample_docs(), dense_order=["a", "b", "c", "d"])
    with pytest.raises(h.HybridUnavailable):
        h.hybrid_search(coll, [0.0], "x", n_results=3,
                        where_filter={"technique": "no-such-technique"})
