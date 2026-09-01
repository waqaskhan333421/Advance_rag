"""Unit tests for Reciprocal Rank Fusion."""

import pytest

from src.retrieval import RetrievedChunk, reciprocal_rank_fusion


def make_chunk(cid: str, score: float, source: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=cid,
        text=f"text-{cid}",
        score=score,
        metadata={"page": 1},
        source=source,
    )


def test_rrf_basic_fusion():
    dense = [
        make_chunk("a", 0.9, "dense"),
        make_chunk("b", 0.8, "dense"),
        make_chunk("c", 0.7, "dense"),
    ]
    sparse = [
        make_chunk("b", 0.85, "sparse"),
        make_chunk("d", 0.75, "sparse"),
        make_chunk("a", 0.6, "sparse"),
    ]

    fused = reciprocal_rank_fusion(dense, sparse, k=60, top_k=10)

    ids = [f.chunk_id for f in fused]
    assert "b" in ids  # appears in both lists
    assert "a" in ids
    assert "d" in ids
    assert "c" in ids

    # b should rank highest (appears in both)
    assert fused[0].chunk_id == "b"


def test_rrf_deduplication():
    dense = [make_chunk("x", 0.9, "dense")]
    sparse = [make_chunk("x", 0.8, "sparse")]

    fused = reciprocal_rank_fusion(dense, sparse, k=60, top_k=10)
    assert len(fused) == 1
    assert fused[0].chunk_id == "x"
    # Score should be higher than single list due to both contributions
    assert fused[0].score > 1 / 61


def test_rrf_empty_input():
    assert reciprocal_rank_fusion([], [], k=60, top_k=10) == []
    assert len(reciprocal_rank_fusion([make_chunk("a", 0.5, "dense")], [], k=60)) == 1
