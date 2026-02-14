"""
Embedding Storage — compute and store embeddings for entities and hyperedges.
"""

import os
import pickle
from pathlib import Path

import numpy as np

from src.utils.logger import get_logger
from src.utils.embedding import EmbeddingModel

logger = get_logger(__name__)


class EmbeddingStore:
    """Compute, store, and load entity/hyperedge embeddings."""

    def __init__(self, config: dict):
        self.model_name = config.get("model", "sentence-transformers/all-MiniLM-L6-v2")
        self.batch_size = config.get("batch_size", 256)
        self.normalize = config.get("normalize", True)
        self._embedder: EmbeddingModel | None = None

        self.entity_embeddings: np.ndarray | None = None
        self.hyperedge_embeddings: np.ndarray | None = None

    @property
    def embedder(self) -> EmbeddingModel:
        if self._embedder is None:
            self._embedder = EmbeddingModel(
                model_name=self.model_name,
                batch_size=self.batch_size,
                normalize=self.normalize,
            )
        return self._embedder

    def set_embedder(self, embedder: EmbeddingModel) -> None:
        self._embedder = embedder

    def compute_entity_embeddings(self, entities: list) -> np.ndarray:
        """Compute embeddings for all entities."""
        names = [e.name for e in entities]
        logger.debug(f"Computing entity embeddings for {len(names)} entities")
        self.entity_embeddings = self.embedder.encode(names, show_progress=True)
        logger.step(
            "EmbeddingStore",
            f"Entity embeddings: {self.entity_embeddings.shape}",
        )
        return self.entity_embeddings

    def compute_hyperedge_embeddings(self, chunks: list) -> np.ndarray:
        """Compute embeddings for all hyperedges (chunks)."""
        texts = [c.text for c in chunks]
        logger.debug(f"Computing hyperedge embeddings for {len(texts)} chunks")
        self.hyperedge_embeddings = self.embedder.encode(texts, show_progress=True)
        logger.step(
            "EmbeddingStore",
            f"Hyperedge embeddings: {self.hyperedge_embeddings.shape}",
        )
        return self.hyperedge_embeddings

    def save(self, output_dir: str) -> None:
        """Save embeddings to disk."""
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)

        if self.entity_embeddings is not None:
            np.save(path / "entity_embeddings.npy", self.entity_embeddings)
        if self.hyperedge_embeddings is not None:
            np.save(path / "hyperedge_embeddings.npy", self.hyperedge_embeddings)

        logger.debug(f"Embeddings saved to {output_dir}")

    def load(self, input_dir: str) -> None:
        """Load embeddings from disk."""
        path = Path(input_dir)

        ent_path = path / "entity_embeddings.npy"
        he_path = path / "hyperedge_embeddings.npy"

        if ent_path.exists():
            self.entity_embeddings = np.load(ent_path)
            logger.debug(f"Loaded entity embeddings: {self.entity_embeddings.shape}")
        if he_path.exists():
            self.hyperedge_embeddings = np.load(he_path)
            logger.debug(
                f"Loaded hyperedge embeddings: {self.hyperedge_embeddings.shape}"
            )
