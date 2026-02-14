"""
NER-based entity extraction from chunks.

Supports SpaCy (fast, well-tested) and GLiNER (zero-shot, custom types).
All extraction is local — no LLM calls.
"""

import re
from dataclasses import dataclass
from typing import Optional

from src.utils.logger import get_logger, get_progress

logger = get_logger(__name__)


@dataclass
class Entity:
    """A named entity extracted from a chunk."""
    name: str
    entity_type: str
    span_start: int = 0
    span_end: int = 0

    @property
    def normalized_name(self) -> str:
        """Normalize: strip, collapse whitespace, title case."""
        text = re.sub(r'\s+', ' ', self.name.strip())
        return text.title()


class NERExtractor:
    """Extract named entities using local NER models."""

    def __init__(self, config: dict):
        self.model_type = config.get("model", "spacy")
        self.entity_types = config.get(
            "entity_types",
            ["person", "organization", "location", "date", "event", "product"],
        )
        self.min_entity_length = config.get("min_entity_length", 2)
        self.exclude_types = set(
            config.get("exclude_types", ["CARDINAL", "ORDINAL", "QUANTITY"])
        )

        self._spacy_model_name = config.get("spacy_model", "en_core_web_sm")
        self._gliner_model_name = config.get(
            "gliner_model", "urchade/gliner_multi-v2.1"
        )
        self._model = None

        logger.step(
            "NERExtractor",
            f"Initialized with backend='{self.model_type}'",
            min_length=self.min_entity_length,
        )

    def _load_model(self) -> None:
        """Lazy load the NER model."""
        logger.start_timer("ner_model_load")

        if self.model_type == "spacy":
            import spacy
            try:
                self._model = spacy.load(self._spacy_model_name)
            except OSError:
                logger.warning(
                    f"SpaCy model '{self._spacy_model_name}' not found. "
                    "Downloading..."
                )
                from spacy.cli import download
                download(self._spacy_model_name)
                self._model = spacy.load(self._spacy_model_name)

        elif self.model_type == "gliner":
            from gliner import GLiNER
            self._model = GLiNER.from_pretrained(self._gliner_model_name)

        else:
            raise ValueError(f"Unknown NER model type: {self.model_type}")

        elapsed = logger.stop_timer("ner_model_load")
        logger.step("NERExtractor", f"Model loaded ({elapsed})")

    @property
    def model(self):
        if self._model is None:
            self._load_model()
        return self._model

    def extract(self, text: str) -> list[Entity]:
        """
        Extract entities from a single text.

        Returns deduplicated list of Entity objects.
        """
        if self.model_type == "spacy":
            return self._extract_spacy(text)
        elif self.model_type == "gliner":
            return self._extract_gliner(text)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def _extract_spacy(self, text: str) -> list[Entity]:
        """Extract entities using SpaCy."""
        doc = self.model(text)
        entities = []
        seen = set()

        for ent in doc.ents:
            if ent.label_ in self.exclude_types:
                continue
            name = re.sub(r'\s+', ' ', ent.text.strip())
            if len(name) < self.min_entity_length:
                continue
            normalized = name.title()
            if normalized in seen:
                continue
            seen.add(normalized)
            entities.append(
                Entity(
                    name=normalized,
                    entity_type=ent.label_,
                    span_start=ent.start_char,
                    span_end=ent.end_char,
                )
            )

        return entities

    def _extract_gliner(self, text: str) -> list[Entity]:
        """Extract entities using GLiNER (zero-shot)."""
        raw_entities = self.model.predict_entities(
            text, labels=self.entity_types, threshold=0.5
        )
        entities = []
        seen = set()

        for ent in raw_entities:
            name = re.sub(r'\s+', ' ', ent["text"].strip())
            if len(name) < self.min_entity_length:
                continue
            normalized = name.title()
            if normalized in seen:
                continue
            seen.add(normalized)
            entities.append(
                Entity(
                    name=normalized,
                    entity_type=ent["label"],
                    span_start=ent.get("start", 0),
                    span_end=ent.get("end", 0),
                )
            )

        return entities

    def extract_from_chunks(self, chunks: list) -> list:
        """
        Extract entities from all chunks, mutating chunk.entities in-place.

        Returns: list of all unique Entity objects across all chunks.
        """
        logger.start_timer("ner_all_chunks")
        total_entities = 0

        with get_progress() as progress:
            task = progress.add_task("NER extraction...", total=len(chunks))

            for chunk in chunks:
                chunk.entities = self.extract(chunk.text)
                total_entities += len(chunk.entities)
                progress.advance(task)

        elapsed = logger.stop_timer("ner_all_chunks")

        # Per-chunk stats
        entity_counts = [len(c.entities) for c in chunks]
        import numpy as np
        logger.step(
            "NERExtractor",
            f"Extracted entities from {len(chunks)} chunks",
            total_entities=total_entities,
            avg_per_chunk=f"{np.mean(entity_counts):.1f}" if entity_counts else "0",
            max_per_chunk=max(entity_counts) if entity_counts else 0,
            time=elapsed,
        )

        # Log entity type distribution
        type_dist: dict[str, int] = {}
        for chunk in chunks:
            for ent in chunk.entities:
                type_dist[ent.entity_type] = type_dist.get(ent.entity_type, 0) + 1
        logger.debug(f"  Entity type distribution: {type_dist}")

        return chunks

    def extract_from_chunks_batch(self, chunks: list, batch_size: int = 50) -> list:
        """
        Extract entities from all chunks using batch processing.

        For SpaCy: uses nlp.pipe() which handles internal batching/multi-threading.
        For GLiNER: falls back to sequential extraction.

        Args:
            chunks: list of Chunk objects
            batch_size: number of texts to process per batch (SpaCy only)

        Returns:
            chunks with .entities populated in-place
        """
        if self.model_type != "spacy":
            logger.debug("Batch NER not supported for GLiNER, falling back to sequential")
            return self.extract_from_chunks(chunks)

        logger.start_timer("ner_batch")
        total_entities = 0

        texts = [chunk.text for chunk in chunks]

        with get_progress() as progress:
            task = progress.add_task("Batch NER extraction...", total=len(chunks))

            for i, doc in enumerate(self.model.pipe(texts, batch_size=batch_size)):
                entities = []
                seen = set()
                for ent in doc.ents:
                    if ent.label_ in self.exclude_types:
                        continue
                    name = re.sub(r'\s+', ' ', ent.text.strip())
                    if len(name) < self.min_entity_length:
                        continue
                    normalized = name.title()
                    if normalized in seen:
                        continue
                    seen.add(normalized)
                    entities.append(
                        Entity(
                            name=normalized,
                            entity_type=ent.label_,
                            span_start=ent.start_char,
                            span_end=ent.end_char,
                        )
                    )
                chunks[i].entities = entities
                total_entities += len(entities)
                progress.advance(task)

        elapsed = logger.stop_timer("ner_batch")

        entity_counts = [len(c.entities) for c in chunks]
        import numpy as np
        logger.step(
            "NERExtractor",
            f"Batch extracted entities from {len(chunks)} chunks",
            total_entities=total_entities,
            avg_per_chunk=f"{np.mean(entity_counts):.1f}" if entity_counts else "0",
            max_per_chunk=max(entity_counts) if entity_counts else 0,
            batch_size=batch_size,
            time=elapsed,
        )

        type_dist: dict[str, int] = {}
        for chunk in chunks:
            for ent in chunk.entities:
                type_dist[ent.entity_type] = type_dist.get(ent.entity_type, 0) + 1
        logger.debug(f"  Entity type distribution: {type_dist}")

        return chunks


def deduplicate_entities(chunks: list) -> list[Entity]:
    """
    Build a global deduplicated entity list from all chunks.
    Deduplication is by normalized name.
    """
    entity_map: dict[str, Entity] = {}
    for chunk in chunks:
        for ent in chunk.entities:
            key = ent.normalized_name
            if key not in entity_map:
                entity_map[key] = Entity(
                    name=key,
                    entity_type=ent.entity_type,
                )

    global_entities = list(entity_map.values())
    logger.step(
        "Entity Dedup",
        f"Global unique entities: {len(global_entities)}",
        from_chunks=len(chunks),
    )

    # Log type breakdown
    type_counts: dict[str, int] = {}
    for ent in global_entities:
        type_counts[ent.entity_type] = type_counts.get(ent.entity_type, 0) + 1
    logger.debug(f"  Global entity types: {type_counts}")

    return global_entities
