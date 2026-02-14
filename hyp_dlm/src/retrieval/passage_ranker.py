"""
PPR-based Passage Ranking — rank passages using Personalized PageRank
on the entity-passage bipartite graph.
"""

import numpy as np
from scipy import sparse

from src.utils.logger import get_logger
from src.utils.sparse_ops import cosine_sim_vec

logger = get_logger(__name__)


class PassageRanker:
    """Rank passages via PPR on entity-passage bipartite graph."""

    def __init__(self, config: dict):
        self.damping_factor = config.get("damping_factor", 0.85)
        self.iterations = config.get("iterations", 20)
        self.top_k = config.get("top_k_passages", 10)
        self.query_weight = config.get("query_weight", 0.3)

        logger.step(
            "PassageRanker",
            "Initialized",
            damping=self.damping_factor,
            iterations=self.iterations,
            top_k=self.top_k,
        )

    def rank(
        self,
        entity_scores: np.ndarray,
        H: sparse.csr_matrix,
        query_embedding: np.ndarray,
        hyperedge_embeddings: np.ndarray,
    ) -> list[dict]:
        """
        Rank passages using PPR on bipartite entity-passage graph.

        Args:
            entity_scores: activation scores from propagation (N,)
            H: incidence matrix (N x M)
            query_embedding: query embedding (dim,)
            hyperedge_embeddings: all hyperedge embeddings (M x dim)

        Returns:
            List of {"hyperedge_id": int, "score": float} sorted by score desc
        """
        logger.start_timer("ppr_ranking")

        N, M = H.shape
        total_nodes = N + M

        # Build bipartite adjacency matrix
        # [0, H; H^T, 0] but as a combined system
        # Entity indices: 0..N-1
        # Passage indices: N..N+M-1

        # Initial scores
        scores = np.zeros(total_nodes)

        # Entity initial scores from propagation
        scores[:N] = entity_scores

        # Passage initial scores from direct query-passage similarity
        passage_sims = cosine_sim_vec(query_embedding, hyperedge_embeddings)
        passage_sims = (passage_sims + 1.0) / 2.0  # Normalize to [0,1]
        scores[N:] = self.query_weight * passage_sims

        # Normalize initial scores
        total = scores.sum()
        if total > 0:
            initial_scores = scores / total
        else:
            initial_scores = np.ones(total_nodes) / total_nodes

        # Build transition matrix for PPR
        # Entity -> Passage: H[i,j] > 0
        # Passage -> Entity: H[i,j] > 0 (transpose)
        H_float = H.astype(float)

        # Run PPR iterations
        current_scores = initial_scores.copy()
        d = self.damping_factor

        logger.debug(
            f"  PPR: {total_nodes} nodes ({N} entities + {M} passages), "
            f"{self.iterations} iterations"
        )

        for iteration in range(self.iterations):
            new_scores = np.zeros(total_nodes)

            # Entity scores from passage neighbors
            # For each entity i: sum over passages j where H[i,j]=1
            passage_to_entity = H_float @ current_scores[N:]  # N,
            entity_degrees = np.array(H_float.sum(axis=1)).flatten()
            entity_degrees[entity_degrees == 0] = 1.0

            # Passage scores from entity neighbors
            entity_to_passage = H_float.T @ current_scores[:N]  # M,
            passage_degrees = np.array(H_float.sum(axis=0)).flatten()
            passage_degrees[passage_degrees == 0] = 1.0

            new_scores[:N] = (
                (1 - d) * initial_scores[:N]
                + d * passage_to_entity / entity_degrees
            )
            new_scores[N:] = (
                (1 - d) * initial_scores[N:]
                + d * entity_to_passage / passage_degrees
            )

            # Normalize
            total_new = new_scores.sum()
            if total_new > 0:
                new_scores = new_scores / total_new

            current_scores = new_scores

        # Extract passage scores
        passage_scores = current_scores[N:]

        # Rank
        top_ids = np.argsort(passage_scores)[::-1][: self.top_k]

        results = []
        for he_id in top_ids:
            results.append({
                "hyperedge_id": int(he_id),
                "score": float(passage_scores[he_id]),
            })

        elapsed = logger.stop_timer("ppr_ranking")
        logger.step(
            "PassageRanker",
            f"Ranked {len(results)} passages via PPR",
            top_score=f"{results[0]['score']:.6f}" if results else "N/A",
            time=elapsed,
        )

        return results
