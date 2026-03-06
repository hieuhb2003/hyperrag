"""Coreference resolution via FastCoref LingMessCoref.
Replaces pronouns/anaphoric refs with canonical entity forms."""

from __future__ import annotations

import re
import importlib
from dataclasses import dataclass
from typing import Optional

from src.config import Config

_logging = importlib.import_module("src.logging-setup")
get_logger = _logging.get_logger

logger = get_logger("coreference")


class CoreferenceResolver:
    """Document-level coreference resolution using FastCoref."""

    def __init__(self, config: Config):
        self.config = config
        self.model = None

    def load_model(self):
        """Load FastCoref LingMessCoref model."""
        from fastcoref import LingMessCoref
        self.model = LingMessCoref(device=self.config.device)
        logger.info(f"Loaded LingMessCoref on {self.config.device}")

    def resolve(self, documents: list) -> list:
        """Resolve coreferences in all documents. Returns new Document list."""
        from tqdm import tqdm
        _dl = importlib.import_module("src.data-loader")

        if self.model is None:
            self.load_model()

        resolved = []
        total_replacements = 0

        for doc in tqdm(documents, desc="Coreference"):
            resolved_text, n_replacements = self._resolve_single(doc.text)
            resolved.append(_dl.Document(id=doc.id, text=resolved_text))
            total_replacements += n_replacements
            logger.debug(f"Doc {doc.id}: {n_replacements} replacements")

        logger.info(f"Coreference resolved: {len(documents)} docs, "
                    f"{total_replacements} total replacements")
        return resolved

    def resolve_batch(self, documents: list) -> list:
        """Batch coreference resolution for efficiency."""
        from tqdm import tqdm
        _dl = importlib.import_module("src.data-loader")

        if self.model is None:
            self.load_model()

        texts = [doc.text for doc in documents]
        # Process in batches
        batch_size = self.config.coref_batch_size
        resolved = []
        total_replacements = 0

        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            batch_docs = documents[i:i + batch_size]

            preds = self.model.predict(texts=batch_texts)
            for doc, pred in zip(batch_docs, preds):
                resolved_text, n_rep = self._apply_clusters(
                    doc.text, pred.get_clusters(as_strings=False)
                )
                resolved.append(_dl.Document(id=doc.id, text=resolved_text))
                total_replacements += n_rep

            logger.debug(f"Batch {i // batch_size + 1}: processed {len(batch_texts)} docs")

        logger.info(f"Coreference resolved (batch): {len(documents)} docs, "
                    f"{total_replacements} total replacements")
        return resolved

    def _resolve_single(self, text: str) -> tuple[str, int]:
        """Resolve coreferences in a single text. Returns (resolved_text, n_replacements)."""
        preds = self.model.predict(texts=[text])
        clusters = preds[0].get_clusters(as_strings=False)
        return self._apply_clusters(text, clusters)

    def _apply_clusters(self, text: str, clusters: list[list[tuple[int, int]]]) -> tuple[str, int]:
        """Apply coreference clusters to text by replacing mentions with canonical form."""
        if not clusters:
            return text, 0

        # Collect all replacements: (start, end, canonical_text)
        replacements = []
        for cluster in clusters:
            # Extract mention texts
            mentions = [(start, end, text[start:end]) for start, end in cluster]
            canonical = self._select_canonical([m[2] for m in mentions])

            for start, end, mention_text in mentions:
                if mention_text.strip() != canonical.strip():
                    replacements.append((start, end, canonical))

        # Sort by start position descending (replace right-to-left to preserve offsets)
        replacements.sort(key=lambda x: x[0], reverse=True)

        # Remove overlapping spans (keep first = rightmost)
        filtered = []
        last_start = len(text)
        for start, end, repl in replacements:
            if end <= last_start:
                filtered.append((start, end, repl))
                last_start = start

        # Apply replacements
        result = text
        for start, end, repl in filtered:
            result = result[:start] + repl + result[end:]

        return result, len(filtered)

    def _select_canonical(self, mentions: list[str]) -> str:
        """Select canonical mention: proper noun > common noun > pronoun, longer preferred."""
        if not mentions:
            return ""

        scored = []
        for m in mentions:
            stripped = m.strip()
            # Score: proper noun (capitalized, multi-word preferred)
            is_proper = bool(re.match(r'^[A-Z]', stripped)) and len(stripped) > 2
            # Penalize pronouns
            is_pronoun = stripped.lower() in {
                "he", "she", "it", "they", "him", "her", "his", "its",
                "their", "them", "we", "us", "our", "who", "whom",
                "this", "that", "these", "those",
            }
            score = (
                0 if is_pronoun else (2 if is_proper else 1),  # type priority
                len(stripped),  # length tie-break
            )
            scored.append((score, stripped))

        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0][1]
