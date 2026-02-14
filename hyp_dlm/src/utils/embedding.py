"""
Embedding model wrapper for HyP-DLM.

Wraps sentence-transformers for encoding entities, chunks, and queries
into the same vector space.
"""

import numpy as np
from typing import Optional

from src.utils.logger import get_logger

logger = get_logger(__name__)


class EmbeddingModel:
    """Wrapper around sentence-transformers for consistent embedding."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        batch_size: int = 256,
        normalize: bool = True,
        device: Optional[str] = None,
    ):
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize = normalize

        logger.start_timer("embedding_model_load")
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, device=device)
        self.dim = self.model.get_sentence_embedding_dimension()
        elapsed = logger.stop_timer("embedding_model_load")
        logger.step(
            "Embedding",
            f"Loaded model '{model_name}'",
            dim=self.dim,
            load_time=elapsed,
        )

    def encode(
        self,
        texts: list[str],
        show_progress: bool = False,
    ) -> np.ndarray:
        """
        Encode a list of texts into embeddings.

        Returns:
            np.ndarray of shape (len(texts), dim)
        """
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)

        logger.debug(f"Encoding {len(texts)} texts (batch_size={self.batch_size})")
        logger.start_timer("encoding")

        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=show_progress,
            normalize_embeddings=self.normalize,
            convert_to_numpy=True,
        )

        elapsed = logger.stop_timer("encoding")
        logger.debug(
            f"Encoded {len(texts)} texts → shape {embeddings.shape} ({elapsed})"
        )
        return embeddings.astype(np.float32)

    def encode_single(self, text: str) -> np.ndarray:
        """Encode a single text string. Returns shape (dim,)."""
        return self.encode([text])[0]
