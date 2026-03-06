"""spaCy NER implementation. Traditional NER with fixed entity types."""

from __future__ import annotations

import importlib

from src.config import Config

_base = importlib.import_module("src.indexing.entity-extractor-base")
EntityExtractorBase = _base.EntityExtractorBase

_logging = importlib.import_module("src.logging-setup")
logger = _logging.get_logger("ner.spacy")


class SpacyEntityExtractor(EntityExtractorBase):
    """NER using spaCy (en_core_web_sm or configured model)."""

    def __init__(self, config: Config):
        super().__init__(config)
        self.nlp = None

    def load_model(self):
        import spacy
        self.nlp = spacy.load(self.config.spacy_model)
        logger.info(f"Loaded spaCy model: {self.config.spacy_model}")

    def extract_from_chunk(self, chunk) -> list[tuple[str, str]]:
        """Extract entities from a single chunk using spaCy."""
        if self.nlp is None:
            self.load_model()

        doc = self.nlp(chunk.text)
        result = [(ent.text, ent.label_) for ent in doc.ents]
        logger.debug(f"Chunk {chunk.id}: {len(result)} entities extracted")
        return result
