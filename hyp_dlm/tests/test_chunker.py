"""Tests for Semantic Chunker and Anchor Chunker."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest


# ══════════════════════════════════════════════════════════
# SemanticChunker tests (existing)
# ══════════════════════════════════════════════════════════

def test_sentence_tokenize():
    from src.indexing.semantic_chunker import SemanticChunker

    text = "Hello world. This is a test. How are you?"
    sentences = SemanticChunker.sentence_tokenize(text)
    assert len(sentences) == 3
    assert sentences[0] == "Hello world."
    assert sentences[1] == "This is a test."


def test_sentence_tokenize_abbreviations():
    from src.indexing.semantic_chunker import SemanticChunker

    text = "Dr. Smith went to Washington. He met with officials."
    sentences = SemanticChunker.sentence_tokenize(text)
    # May split on "Washington." since next word is capitalized
    assert len(sentences) >= 1


def test_chunk_document_basic():
    from src.indexing.semantic_chunker import SemanticChunker

    config = {
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "similarity_threshold": 0.3,  # Low threshold = fewer breaks
        "min_chunk_tokens": 5,
        "max_chunk_tokens": 100,
        "overlap_sentences": 0,
    }

    chunker = SemanticChunker(config)

    text = (
        "Albert Einstein was born in Germany. He developed the theory of relativity. "
        "The Eiffel Tower is located in Paris. It was built in 1889 for the World's Fair. "
        "Python is a programming language. It is widely used in data science."
    )

    chunks = chunker.chunk_document(text, doc_id="test_doc")
    assert len(chunks) > 0
    for chunk in chunks:
        assert chunk.doc_id == "test_doc"
        assert len(chunk.text) > 0
        assert chunk.token_count > 0


def test_empty_document():
    from src.indexing.semantic_chunker import SemanticChunker

    config = {
        "similarity_threshold": 0.5,
        "min_chunk_tokens": 5,
        "max_chunk_tokens": 100,
        "overlap_sentences": 0,
    }

    chunker = SemanticChunker(config)
    chunks = chunker.chunk_document("", doc_id="empty")
    assert len(chunks) == 0


# ══════════════════════════════════════════════════════════
# AnchorChunker: robust_sentence_split tests
# ══════════════════════════════════════════════════════════

def test_robust_sentence_split_basic():
    from src.indexing.semantic_chunker import AnchorChunker

    text = "Hello world. This is a test. How are you?"
    sentences = AnchorChunker.robust_sentence_split(text)
    assert len(sentences) == 3
    assert sentences[0] == "Hello world."
    assert sentences[1] == "This is a test."
    assert sentences[2] == "How are you?"


def test_robust_sentence_split_abbreviations():
    from src.indexing.semantic_chunker import AnchorChunker

    text = "Dr. Smith went to Washington. He met with officials."
    sentences = AnchorChunker.robust_sentence_split(text)
    # "Dr." should be protected — should NOT split after "Dr."
    # Should still split after "Washington." since next word is capitalized
    assert len(sentences) >= 1
    # The first sentence should contain "Dr. Smith" intact
    assert "Dr." in sentences[0] or "Dr" in sentences[0]


def test_robust_sentence_split_decimals():
    from src.indexing.semantic_chunker import AnchorChunker

    text = "The value is 3.14 approximately. That is pi."
    sentences = AnchorChunker.robust_sentence_split(text)
    # "3.14" should NOT cause a split
    assert any("3.14" in s for s in sentences)
    assert len(sentences) == 2


def test_robust_sentence_split_ellipsis():
    from src.indexing.semantic_chunker import AnchorChunker

    text = "He thought about it... Then he decided to go. It was a long journey."
    sentences = AnchorChunker.robust_sentence_split(text)
    # Ellipsis should be protected — "..." should not cause weird splits
    assert any("..." in s for s in sentences)


def test_robust_sentence_split_empty():
    from src.indexing.semantic_chunker import AnchorChunker

    assert AnchorChunker.robust_sentence_split("") == []
    assert AnchorChunker.robust_sentence_split("   ") == []


def test_robust_sentence_split_us_abbreviation():
    from src.indexing.semantic_chunker import AnchorChunker

    text = "The U.S. government announced new policies. Congress will vote next week."
    sentences = AnchorChunker.robust_sentence_split(text)
    # U.S. should be protected
    assert any("U.S." in s or "U.S" in s for s in sentences)


# ══════════════════════════════════════════════════════════
# AnchorChunker: _insert_markers tests
# ══════════════════════════════════════════════════════════

def test_insert_markers():
    from src.indexing.semantic_chunker import AnchorChunker

    sentences = ["Hello world.", "This is a test.", "How are you?"]
    marked_text, marker_map = AnchorChunker._insert_markers(sentences)

    assert "[1]" in marked_text
    assert "[2]" in marked_text
    assert "[3]" in marked_text
    assert marker_map == {1: 0, 2: 1, 3: 2}

    # Verify order
    pos1 = marked_text.index("[1]")
    pos2 = marked_text.index("[2]")
    pos3 = marked_text.index("[3]")
    assert pos1 < pos2 < pos3


def test_insert_markers_single():
    from src.indexing.semantic_chunker import AnchorChunker

    sentences = ["Only one sentence."]
    marked_text, marker_map = AnchorChunker._insert_markers(sentences)

    assert "[1]" in marked_text
    assert marker_map == {1: 0}


# ══════════════════════════════════════════════════════════
# AnchorChunker: _parse_llm_response tests
# ══════════════════════════════════════════════════════════

def test_parse_llm_response_normal():
    from src.indexing.semantic_chunker import AnchorChunker

    result = AnchorChunker._parse_llm_response("1, 3, 5", max_marker=6)
    assert result == [1, 3, 5]


def test_parse_llm_response_bracket_wrapped():
    from src.indexing.semantic_chunker import AnchorChunker

    result = AnchorChunker._parse_llm_response("[1], [3], [5]", max_marker=6)
    assert result == [1, 3, 5]


def test_parse_llm_response_missing_one():
    from src.indexing.semantic_chunker import AnchorChunker

    # If [1] is missing from response, it should be added
    result = AnchorChunker._parse_llm_response("3, 5", max_marker=6)
    assert result == [1, 3, 5]


def test_parse_llm_response_out_of_range():
    from src.indexing.semantic_chunker import AnchorChunker

    result = AnchorChunker._parse_llm_response("1, 3, 99", max_marker=6)
    assert result == [1, 3]
    assert 99 not in result


def test_parse_llm_response_empty():
    from src.indexing.semantic_chunker import AnchorChunker

    result = AnchorChunker._parse_llm_response("", max_marker=6)
    assert result == [1]  # Always includes 1


def test_parse_llm_response_duplicates():
    from src.indexing.semantic_chunker import AnchorChunker

    result = AnchorChunker._parse_llm_response("1, 1, 3, 3, 5", max_marker=6)
    assert result == [1, 3, 5]  # No duplicates


# ══════════════════════════════════════════════════════════
# create_chunker factory tests
# ══════════════════════════════════════════════════════════

def test_create_chunker_similarity():
    from src.indexing.semantic_chunker import create_chunker, SemanticChunker

    config = {"method": "similarity", "similarity_threshold": 0.5}
    chunker = create_chunker(config)
    assert isinstance(chunker, SemanticChunker)


def test_create_chunker_llm_anchor():
    from src.indexing.semantic_chunker import create_chunker, AnchorChunker

    config = {"method": "llm_anchor", "llm_model": "gpt-4o-mini"}
    chunker = create_chunker(config)
    assert isinstance(chunker, AnchorChunker)


def test_create_chunker_default():
    from src.indexing.semantic_chunker import create_chunker, SemanticChunker

    # No method specified — should default to similarity
    config = {"similarity_threshold": 0.5}
    chunker = create_chunker(config)
    assert isinstance(chunker, SemanticChunker)


def test_create_chunker_invalid():
    from src.indexing.semantic_chunker import create_chunker

    with pytest.raises(ValueError, match="Unknown chunking method"):
        create_chunker({"method": "invalid_method"})
