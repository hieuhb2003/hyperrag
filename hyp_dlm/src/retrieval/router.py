"""
Familiarity Router — decides retrieval strategy based on query familiarity.

Routes:
  - "direct": Answer from top-K chunks directly (skip graph)
  - "graph": Full DAG-guided graph propagation
  - "hybrid": Light propagation + dense retrieval
"""

import numpy as np

from src.utils.logger import get_logger
from src.utils.embedding import EmbeddingModel
from src.utils.sparse_ops import cosine_sim_vec

logger = get_logger(__name__)


class FamiliarityRouter:
    """Route queries based on familiarity signals."""

    def __init__(self, config: dict):
        self.probe_k = config.get("probe_k", 10)
        self.temperature = config.get("temperature", 1.0)
        self.threshold_high = config.get("threshold_high", 0.75)
        self.threshold_low = config.get("threshold_low", 0.45)
        self.entropy_low = config.get("entropy_low", 1.5)
        self.entropy_high = config.get("entropy_high", 2.5)

        logger.step(
            "Router",
            "Initialized",
            probe_k=self.probe_k,
            threshold_high=self.threshold_high,
            threshold_low=self.threshold_low,
        )

    def route(
        self,
        query_embedding: np.ndarray,
        hyperedge_embeddings: np.ndarray,
    ) -> dict:
        """
        Decide retrieval route for a query.

        Returns:
            {
                "route": "direct" | "graph" | "hybrid",
                "mean_score": float,
                "entropy": float,
                "top_k_scores": list[float],
            }
        """
        logger.start_timer("routing")

        # Probe: find top-K nearest hyperedges
        all_scores = cosine_sim_vec(query_embedding, hyperedge_embeddings)
        top_k_indices = np.argsort(all_scores)[-self.probe_k:]
        top_k_scores = all_scores[top_k_indices]

        mean_score = float(np.mean(top_k_scores))

        # Compute entropy of softmax probabilities
        logits = top_k_scores / self.temperature
        logits = logits - logits.max()  # Numerical stability
        probs = np.exp(logits) / np.sum(np.exp(logits))
        entropy = float(-np.sum(probs * np.log(probs + 1e-10)))

        # Route decision
        if mean_score > self.threshold_high and entropy < self.entropy_low:
            route = "direct"
        elif mean_score < self.threshold_low or entropy > self.entropy_high:
            route = "graph"
        else:
            route = "hybrid"

        elapsed = logger.stop_timer("routing")

        result = {
            "route": route,
            "mean_score": mean_score,
            "entropy": entropy,
            "top_k_scores": top_k_scores.tolist(),
        }

        logger.step(
            "Router",
            f"Route decision: '{route}'",
            mean_score=f"{mean_score:.4f}",
            entropy=f"{entropy:.4f}",
            time=elapsed,
        )
        logger.debug(
            f"  Thresholds: high={self.threshold_high}, low={self.threshold_low}, "
            f"entropy_low={self.entropy_low}, entropy_high={self.entropy_high}"
        )

        return result
