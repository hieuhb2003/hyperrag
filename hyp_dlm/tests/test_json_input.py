"""Tests for JSON/JSONL input loading and debug mode."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest


# ══════════════════════════════════════════════════════════
# JSON/JSONL loading tests
# ══════════════════════════════════════════════════════════

def test_validate_document_entry_valid():
    from scripts.index_corpus import _validate_document_entry

    entry = {"id": "doc1", "text": "Hello world", "path": "/tmp/doc1.txt", "metadata": {"source": "web"}}
    result = _validate_document_entry(entry, 0)
    assert result["id"] == "doc1"
    assert result["text"] == "Hello world"
    assert result["path"] == "/tmp/doc1.txt"
    assert result["metadata"] == {"source": "web"}


def test_validate_document_entry_minimal():
    from scripts.index_corpus import _validate_document_entry

    entry = {"id": "doc1", "text": "Hello world"}
    result = _validate_document_entry(entry, 0)
    assert result["path"] == ""
    assert result["metadata"] is None


def test_validate_document_entry_missing_id():
    from scripts.index_corpus import _validate_document_entry

    with pytest.raises(ValueError, match="missing required field 'id'"):
        _validate_document_entry({"text": "hello"}, 0)


def test_validate_document_entry_missing_text():
    from scripts.index_corpus import _validate_document_entry

    with pytest.raises(ValueError, match="missing required field 'text'"):
        _validate_document_entry({"id": "doc1"}, 0)


def test_validate_document_entry_wrong_type():
    from scripts.index_corpus import _validate_document_entry

    with pytest.raises(ValueError, match="'id' must be string"):
        _validate_document_entry({"id": 123, "text": "hello"}, 0)


def test_validate_document_entry_empty_text():
    from scripts.index_corpus import _validate_document_entry

    with pytest.raises(ValueError, match="'text' is empty"):
        _validate_document_entry({"id": "doc1", "text": "   "}, 0)


def test_validate_document_entry_not_dict():
    from scripts.index_corpus import _validate_document_entry

    with pytest.raises(ValueError, match="expected dict"):
        _validate_document_entry("not a dict", 0)


def test_load_json_file(tmp_path):
    from scripts.index_corpus import load_documents_json

    docs = [
        {"id": "doc1", "text": "First document content"},
        {"id": "doc2", "text": "Second document content"},
    ]
    json_file = tmp_path / "test.json"
    json_file.write_text(json.dumps(docs))
    result = load_documents_json(str(json_file))

    assert len(result) == 2
    assert result[0]["id"] == "doc1"
    assert result[1]["id"] == "doc2"


def test_load_jsonl_file(tmp_path):
    from scripts.index_corpus import load_documents_json

    jsonl_file = tmp_path / "test.jsonl"
    lines = [
        json.dumps({"id": "doc1", "text": "First"}),
        json.dumps({"id": "doc2", "text": "Second"}),
        "",  # empty line should be skipped
        json.dumps({"id": "doc3", "text": "Third"}),
    ]
    jsonl_file.write_text("\n".join(lines) + "\n")
    result = load_documents_json(str(jsonl_file))

    assert len(result) == 3
    assert result[2]["id"] == "doc3"


def test_load_json_not_array(tmp_path):
    from scripts.index_corpus import load_documents_json

    json_file = tmp_path / "bad.json"
    json_file.write_text(json.dumps({"id": "doc1", "text": "hello"}))
    with pytest.raises(ValueError, match="top-level array"):
        load_documents_json(str(json_file))


def test_load_json_file_not_found():
    from scripts.index_corpus import load_documents_json

    with pytest.raises(FileNotFoundError):
        load_documents_json("/nonexistent/path.json")


def test_load_jsonl_invalid_line(tmp_path):
    from scripts.index_corpus import load_documents_json

    jsonl_file = tmp_path / "bad.jsonl"
    jsonl_file.write_text(
        json.dumps({"id": "doc1", "text": "valid"}) + "\nnot valid json\n"
    )
    with pytest.raises(ValueError, match="JSONL parse error"):
        load_documents_json(str(jsonl_file))


def test_load_json_with_metadata(tmp_path):
    from scripts.index_corpus import load_documents_json

    docs = [{"id": "doc1", "text": "Content", "metadata": {"author": "Jane"}}]
    json_file = tmp_path / "meta.json"
    json_file.write_text(json.dumps(docs))
    result = load_documents_json(str(json_file))

    assert result[0]["metadata"] == {"author": "Jane"}


# ══════════════════════════════════════════════════════════
# Chunk metadata field test
# ══════════════════════════════════════════════════════════

def test_chunk_metadata_field():
    from src.indexing.semantic_chunker import Chunk

    # Default: None
    chunk = Chunk(text="hello", doc_id="d1", chunk_id=0, start_char=0, end_char=5)
    assert chunk.metadata is None

    # With metadata
    chunk2 = Chunk(text="hello", doc_id="d1", chunk_id=0, start_char=0, end_char=5,
                   metadata={"source": "wiki"})
    assert chunk2.metadata == {"source": "wiki"}


# ══════════════════════════════════════════════════════════
# Debug mode logger tests
# ══════════════════════════════════════════════════════════

def test_debug_mode_disabled_by_default():
    from src.utils.logger import get_logger

    logger = get_logger("test_debug_default")
    assert logger.is_debug is False


def test_debug_mode_enable(tmp_path):
    from src.utils.logger import get_logger

    logger = get_logger("test_debug_enable")
    logger.enable_debug(str(tmp_path), max_samples=5)
    assert logger.is_debug is True
    assert logger._debug_max_samples == 5
    assert logger._debug_output_dir == str(tmp_path)


def test_debug_sample_noop_when_disabled():
    from src.utils.logger import get_logger

    logger = get_logger("test_debug_noop")
    logger._debug_enabled = False
    # Should not raise
    logger.debug_sample("Test", ["item1", "item2"])


def test_debug_checkpoint_creates_file(tmp_path):
    from src.utils.logger import get_logger

    logger = get_logger("test_debug_checkpoint")
    logger.enable_debug(str(tmp_path))
    logger.debug_checkpoint("test_step", {"key": "value", "count": 42})

    json_path = tmp_path / "test_step.json"
    assert json_path.exists()
    data = json.loads(json_path.read_text())
    assert data["key"] == "value"
    assert data["count"] == 42


def test_debug_checkpoint_with_timing(tmp_path):
    from src.utils.logger import get_logger

    logger = get_logger("test_debug_timing")
    logger.enable_debug(str(tmp_path))
    logger.debug_checkpoint("timed_step", {"data": "test"}, step_time=1.234)

    assert len(logger._debug_timings) == 1
    assert logger._debug_timings[0]["step"] == "timed_step"
    assert logger._debug_timings[0]["time_s"] == 1.234


def test_debug_report_creates_file(tmp_path):
    from src.utils.logger import get_logger

    logger = get_logger("test_debug_report")
    logger.enable_debug(str(tmp_path))
    logger._debug_timings = [{"step": "s1", "time_s": 1.5}, {"step": "s2", "time_s": 2.0}]
    logger.debug_report(extra={"pipeline": "test"})

    report_path = tmp_path / "debug_report.json"
    assert report_path.exists()
    data = json.loads(report_path.read_text())
    assert data["total_time_s"] == 3.5
    assert data["steps"] == 2
    assert data["pipeline"] == "test"


def test_stop_timer_returns_float():
    from src.utils.logger import get_logger

    logger = get_logger("test_stop_timer")
    logger.start_timer("test_label")
    elapsed = logger.stop_timer("test_label")
    assert isinstance(elapsed, float)
    assert elapsed >= 0


def test_stop_timer_not_found():
    from src.utils.logger import get_logger

    logger = get_logger("test_stop_timer_miss")
    elapsed = logger.stop_timer("nonexistent")
    assert elapsed == -1.0
