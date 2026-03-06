"""Multi-granular scoring: composite of hyperedge + entity scores.
s_comp = w_h * s_hyper_norm + w_e * s_entity_norm"""

from __future__ import annotations

import importlib

import numpy as np
import scipy.sparse as sp

from src.config import Config

_logging = importlib.import_module("src.logging-setup")
logger = _logging.get_logger("scorer")


class MultiGranularScorer:
    """Compute composite chunk scores from entity states."""

    def __init__(self, config: Config):
        self.config = config

    def score(
        self,
        entity_states: dict[str, np.ndarray],
        H_norm: sp.csr_matrix,
        chunks: list,
        entities: list,
        chunk_embeddings: np.ndarray,
        query_embedding: np.ndarray,
    ) -> list[tuple[str, float]]:
        """Compute multi-granular composite scores.

        Returns: list of (chunk_id, score) sorted descending, top-N.
        """
        N = len(entities)
        M = len(chunks)

        # Aggregate entity states across sub-questions (element-wise max)
        s_entity = np.zeros(N, dtype=np.float64)
        for sq_id, state in entity_states.items():
            s_entity = np.maximum(s_entity, state)

        # --- Hyperedge score: semantic relevance of chunk to query ---
        q_norm = np.linalg.norm(query_embedding)
        if q_norm > 0:
            q_normed = query_embedding / q_norm
        else:
            q_normed = query_embedding

        c_norms = np.linalg.norm(chunk_embeddings, axis=1)
        c_norms = np.where(c_norms == 0, 1, c_norms)
        c_normed = chunk_embeddings / c_norms[:, np.newaxis]
        s_hyper = np.maximum(0, c_normed @ q_normed)  # [M], ReLU cosine

        # Min-max normalize hyperedge scores
        s_hyper_norm = self._minmax(s_hyper)

        # --- Entity score per chunk: max entity activation ---
        s_ent_per_chunk = np.zeros(M, dtype=np.float64)
        # Use the raw incidence matrix (binarized from H_norm)
        H_binary = (H_norm != 0).astype(np.float64)
        for j in range(M):
            col = H_binary[:, j]
            ent_indices = col.nonzero()[0]
            if len(ent_indices) > 0:
                s_ent_per_chunk[j] = s_entity[ent_indices].max()

        # Min-max normalize entity scores
        s_ent_norm = self._minmax(s_ent_per_chunk)

        # --- Composite score ---
        w_h = self.config.score_weight_hyper
        w_e = self.config.score_weight_entity
        s_comp = w_h * s_hyper_norm + w_e * s_ent_norm

        # Top-N
        top_n = self.config.score_top_n_final
        top_indices = np.argsort(s_comp)[::-1][:top_n]
        results = [(chunks[i].id, float(s_comp[i])) for i in top_indices if s_comp[i] > 0]

        logger.info(f"Scored {M} chunks, top-{top_n} selected")
        for rank, (cid, score) in enumerate(results[:5]):
            logger.debug(f"  Rank {rank + 1}: {cid} (score={score:.4f})")

        return results

    def _minmax(self, arr: np.ndarray) -> np.ndarray:
        """Min-max normalize array to [0, 1]."""
        mn, mx = arr.min(), arr.max()
        if mx - mn < 1e-10:
            return np.zeros_like(arr)
        return (arr - mn) / (mx - mn)
