"""Abstract NER interface + Entity dataclass + factory function."""

from __future__ import annotations

import importlib
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from src.config import Config

_logging = importlib.import_module("src.logging-setup")
get_logger = _logging.get_logger

logger = get_logger("ner")


@dataclass
class Entity:
    id: str  # "ent_{idx}"
    text: str  # normalized (lowercased, stripped)
    type: str  # entity type
    chunk_ids: list[str] = field(default_factory=list)  # chunks where entity appears


class EntityExtractorBase(ABC):
    """Abstract base for NER extractors."""

    def __init__(self, config: Config):
        self.config = config

    @abstractmethod
    def load_model(self):
        pass

    @abstractmethod
    def extract_from_chunk(self, chunk) -> list[tuple[str, str]]:
        """Return list of (entity_text, entity_type) from a chunk."""
        pass

    def extract_all(self, chunks: list) -> list[Entity]:
        """Extract entities from all chunks, deduplicate globally."""
        from tqdm import tqdm

        registry: dict[str, Entity] = {}  # normalized_text -> Entity
        ent_counter = 0
        filtered_count = 0

        for chunk in tqdm(chunks, desc="NER"):
            raw_entities = self.extract_from_chunk(chunk)
            for text, etype in raw_entities:
                normalized = self._normalize(text)
                if not self._passes_filter(normalized):
                    filtered_count += 1
                    continue
                if normalized in registry:
                    if chunk.id not in registry[normalized].chunk_ids:
                        registry[normalized].chunk_ids.append(chunk.id)
                else:
                    registry[normalized] = Entity(
                        id=f"ent_{ent_counter}",
                        text=normalized,
                        type=etype,
                        chunk_ids=[chunk.id],
                    )
                    ent_counter += 1

        entities = list(registry.values())
        logger.info(f"NER: {len(entities)} unique entities from {len(chunks)} chunks "
                    f"({filtered_count} filtered)")
        return entities

    def _normalize(self, text: str) -> str:
        """Normalize entity text: strip, lowercase."""
        return text.strip().lower()

    def _passes_filter(self, text: str) -> bool:
        """Post-filter: min length, no pure numbers, no score patterns."""
        if len(text) < self.config.ner_min_entity_len:
            return False
        if text.replace(" ", "").replace("-", "").isdigit():
            return False
        # Score patterns like "3-1", "2:0"
        if re.match(r'^\d+[-:]\d+$', text):
            return False
        return True


def create_entity_extractor(config: Config) -> EntityExtractorBase:
    """Factory function: create NER extractor based on config."""
    if config.ner_backend == "gliner":
        _mod = importlib.import_module("src.indexing.entity-extractor-gliner")
        return _mod.GLiNEREntityExtractor(config)
    elif config.ner_backend == "spacy":
        _mod = importlib.import_module("src.indexing.entity-extractor-spacy")
        return _mod.SpacyEntityExtractor(config)
    else:
        raise ValueError(f"Unknown NER backend: {config.ner_backend}")
