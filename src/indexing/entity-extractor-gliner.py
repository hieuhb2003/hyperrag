"""GLiNER2 NER implementation. Zero-shot entity extraction with custom types."""

from __future__ import annotations

import importlib

from src.config import Config

_base = importlib.import_module("src.indexing.entity-extractor-base")
EntityExtractorBase = _base.EntityExtractorBase

_logging = importlib.import_module("src.logging-setup")
logger = _logging.get_logger("ner.gliner")


class GLiNEREntityExtractor(EntityExtractorBase):
    """NER using GLiNER2 (fastino/gliner2-base-v1)."""

    def __init__(self, config: Config):
        super().__init__(config)
        self.model = None

    def load_model(self):
        from gliner import GLiNER
        self.model = GLiNER.from_pretrained(self.config.gliner_model)
        # Move to device if GPU
        if self.config.device != "cpu":
            self.model = self.model.to(self.config.device)
        logger.info(f"Loaded GLiNER2: {self.config.gliner_model} on {self.config.device}")

    def extract_from_chunk(self, chunk) -> list[tuple[str, str]]:
        """Extract entities from a single chunk using GLiNER2."""
        if self.model is None:
            self.load_model()

        entities = self.model.predict_entities(
            chunk.text,
            labels=self.config.ner_entity_types,
            threshold=0.5,
        )
        result = [(e["text"], e["label"]) for e in entities]
        logger.debug(f"Chunk {chunk.id}: {len(result)} entities extracted")
        return result
