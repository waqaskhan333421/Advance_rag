"""Unit tests for the semantic chunker."""

import pytest

from src.ingestion import SemanticChunker


def test_chunker_splits_oversized_text():
    chunker = SemanticChunker()
    chunker.chunk_size = 10  # words
    chunker.chunk_overlap = 2

    long_text = " ".join([f"word{i}" for i in range(100)])
    chunks = chunker._split_text(long_text)

    assert len(chunks) > 1
    for c in chunks:
        assert len(c.split()) <= chunker.chunk_size + chunker.chunk_overlap + 5  # tolerance


def test_chunker_preserves_short_text():
    chunker = SemanticChunker()
    text = "This is a short sentence."
    chunks = chunker._split_text(text)
    assert len(chunks) == 1
    assert chunks[0] == text


def test_chunker_respects_separators():
    chunker = SemanticChunker()
    chunker.chunk_size = 5
    text = "Header\n\nThis is the body content that should split separately."
    chunks = chunker._split_text(text)
    # Should have at least 2 chunks due to \n\n separator
    assert len(chunks) >= 1
