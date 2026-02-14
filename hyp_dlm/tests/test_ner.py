"""Tests for NER Extractor."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest


def test_spacy_extraction():
    from src.indexing.ner_extractor import NERExtractor

    config = {
        "model": "spacy",
        "spacy_model": "en_core_web_sm",
        "min_entity_length": 2,
        "exclude_types": ["CARDINAL", "ORDINAL", "QUANTITY"],
    }

    ner = NERExtractor(config)
    entities = ner.extract(
        "Barack Obama was the 44th President of the United States. "
        "He was born in Honolulu, Hawaii."
    )

    # Should find at least Barack Obama and United States
    names = [e.name for e in entities]
    assert len(entities) > 0
    # Check at least one entity was found
    assert any("Obama" in n or "Barack" in n for n in names)


def test_entity_normalization():
    from src.indexing.ner_extractor import Entity

    ent = Entity(name="  barack   obama  ", entity_type="PERSON")
    assert ent.normalized_name == "Barack Obama"


def test_deduplication():
    from src.indexing.ner_extractor import Entity, deduplicate_entities
    from src.indexing.semantic_chunker import Chunk

    # Create mock chunks with overlapping entities
    chunk1 = Chunk(text="test1", doc_id="d1", chunk_id=0, start_char=0, end_char=5)
    chunk1.entities = [
        Entity(name="Barack Obama", entity_type="PERSON"),
        Entity(name="New York", entity_type="LOC"),
    ]
    chunk2 = Chunk(text="test2", doc_id="d1", chunk_id=1, start_char=5, end_char=10)
    chunk2.entities = [
        Entity(name="Barack Obama", entity_type="PERSON"),
        Entity(name="Washington", entity_type="LOC"),
    ]

    global_entities = deduplicate_entities([chunk1, chunk2])
    names = [e.name for e in global_entities]
    assert len(global_entities) == 3  # Obama, New York, Washington
    assert "Barack Obama" in names


def test_empty_text():
    from src.indexing.ner_extractor import NERExtractor

    config = {"model": "spacy", "spacy_model": "en_core_web_sm", "min_entity_length": 2}
    ner = NERExtractor(config)
    entities = ner.extract("")
    assert len(entities) == 0
