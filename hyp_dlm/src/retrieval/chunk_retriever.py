"""
Hybrid Chunk Retriever — dense retrieval as fallback/complement to graph retrieval.

Used for:
  1. "direct" route: only dense retrieval, no graph
  2. Hybrid fusion: combine with graph retrieval results
"""

import numpy as np

from src.utils.logger import get_logger
from src.utils.sparse_ops import cosine_sim_vec

logger = get_logger(__name__)


class ChunkRetriever:
    """Standard dense retrieval for chunks/hyperedges."""

    def __init__(self, config: dict):
        self.top_k = config.get("top_k", 5)
        logger.step("ChunkRetriever", "Initialized", top_k=self.top_k)

    def retrieve(
        self,
        query_embedding: np.ndarray,
        hyperedge_embeddings: np.ndarray,
        top_k: int | None = None,
    ) -> list[dict]:
        """
        Retrieve top-K chunks by cosine similarity.

        Returns:
            List of {"hyperedge_id": int, "score": float}
        """
        logger.start_timer("dense_retrieval")
        k = top_k or self.top_k

        scores = cosine_sim_vec(query_embedding, hyperedge_embeddings)
        top_ids = np.argsort(scores)[::-1][:k]

        results = []
        for he_id in top_ids:
            results.append({
                "hyperedge_id": int(he_id),
                "score": float(scores[he_id]),
            })

        elapsed = logger.stop_timer("dense_retrieval")
        logger.step(
            "ChunkRetriever",
            f"Retrieved {len(results)} chunks via dense search",
            top_score=f"{results[0]['score']:.4f}" if results else "N/A",
            time=elapsed,
        )

        return results
