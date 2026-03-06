"""8-stage synonym matrix pipeline. Builds S_conf [N x N] confidence matrix.
Stages: TF-IDF sparse → dense encoding → Qdrant hybrid → RRF → context validation."""

from __future__ import annotations

import importlib
from typing import Optional

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine

from src.config import Config

_logging = importlib.import_module("src.logging-setup")
logger = _logging.get_logger("synonym")


class SynonymMatrixBuilder:
    """Build entity-entity synonym confidence matrix via 8-stage pipeline."""

    def __init__(self, config: Config):
        self.config = config
        self.embed_model = None  # shared with chunker

    def set_embed_model(self, model):
        """Reuse embedding model from chunker (DRY)."""
        self.embed_model = model

    def build(self, entities: list, chunks: list,
              entity_to_idx: dict[str, int]) -> sp.csr_matrix:
        """Build S_conf [N x N] synonym confidence matrix."""
        N = len(entities)
        if N < 2:
            logger.info("Fewer than 2 entities, returning empty synonym matrix")
            return sp.csr_matrix((N, N), dtype=np.float32)

        entity_texts = [e.text for e in entities]

        # Stage 1: TF-IDF char n-gram encoding
        tfidf = TfidfVectorizer(
            analyzer="char",
            ngram_range=(self.config.synonym_char_ngram_min, self.config.synonym_char_ngram_max),
        )
        sparse_vecs = tfidf.fit_transform(entity_texts)
        logger.info(f"Stage 1: TF-IDF char n-grams for {N} entities")

        # Stage 2: Sparse candidate search
        sparse_candidates = self._sparse_search(sparse_vecs, top_k=50)
        logger.info(f"Stage 2: {sum(len(v) for v in sparse_candidates.values())} sparse candidates")

        # Stage 3: Dense entity embeddings
        dense_vecs = self.embed_model.encode(entity_texts, show_progress_bar=False, convert_to_numpy=True)
        logger.info("Stage 3: Dense embeddings computed")

        # Stage 4: Dense candidate search
        dense_candidates = self._dense_search(dense_vecs, top_k=50)
        logger.info(f"Stage 4: {sum(len(v) for v in dense_candidates.values())} dense candidates")

        # Stage 5: RRF fusion
        fused = self._rrf_fusion(sparse_candidates, dense_candidates, k=self.config.synonym_rrf_k)
        logger.info(f"Stage 5: {len(fused)} fused candidate pairs")

        # Stage 6: Build entity context windows
        contexts = self._build_contexts(entities, chunks)
        logger.info("Stage 6: Context windows built")

        # Stage 7: Context validation (AND gate, MAX cosine)
        validated = self._context_validate(fused, contexts)
        logger.info(f"Stage 7: {len(validated)} validated synonym pairs")

        # Stage 8: Build S_conf sparse matrix
        S_conf = self._build_matrix(validated, N)
        logger.info(f"Stage 8: S_conf built, shape={S_conf.shape}, nnz={S_conf.nnz}")

        return S_conf

    def _sparse_search(self, sparse_vecs, top_k: int = 50) -> dict[int, list[tuple[int, float]]]:
        """For each entity, find top-K by TF-IDF cosine similarity."""
        N = sparse_vecs.shape[0]
        candidates: dict[int, list[tuple[int, float]]] = {}

        # Process in batches to avoid memory blow-up
        batch_size = 500
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            batch = sparse_vecs[start:end]
            sims = sklearn_cosine(batch, sparse_vecs)  # [batch x N]

            for i_local in range(sims.shape[0]):
                i_global = start + i_local
                row = sims[i_local]
                row[i_global] = -1  # exclude self
                top_idx = np.argsort(row)[-top_k:][::-1]
                candidates[i_global] = [
                    (int(j), float(row[j])) for j in top_idx if row[j] > 0
                ]

        return candidates

    def _dense_search(self, dense_vecs: np.ndarray, top_k: int = 50) -> dict[int, list[tuple[int, float]]]:
        """Dense nearest neighbor search using numpy (avoids Qdrant dependency for simplicity)."""
        N = dense_vecs.shape[0]
        candidates: dict[int, list[tuple[int, float]]] = {}

        # Normalize for cosine
        norms = np.linalg.norm(dense_vecs, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        normed = dense_vecs / norms

        # Process in batches
        batch_size = 500
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            batch = normed[start:end]
            sims = batch @ normed.T  # [batch x N]

            for i_local in range(sims.shape[0]):
                i_global = start + i_local
                row = sims[i_local]
                row[i_global] = -1  # exclude self
                top_idx = np.argsort(row)[-top_k:][::-1]
                candidates[i_global] = [
                    (int(j), float(row[j])) for j in top_idx if row[j] > 0
                ]

        return candidates

    def _rrf_fusion(
        self,
        sparse_cands: dict[int, list[tuple[int, float]]],
        dense_cands: dict[int, list[tuple[int, float]]],
        k: int = 60,
    ) -> dict[tuple[int, int], float]:
        """Reciprocal Rank Fusion of sparse and dense candidate lists."""
        fused: dict[tuple[int, int], float] = {}

        all_entities = set(sparse_cands.keys()) | set(dense_cands.keys())
        for i in all_entities:
            # Build rank maps per source
            scores: dict[int, float] = {}
            for source_cands in [sparse_cands.get(i, []), dense_cands.get(i, [])]:
                for rank, (j, _score) in enumerate(source_cands):
                    if j == i:
                        continue
                    rrf_contrib = 1.0 / (k + rank + 1)
                    scores[j] = scores.get(j, 0.0) + rrf_contrib

            for j, rrf_score in scores.items():
                if rrf_score >= self.config.synonym_rrf_threshold:
                    pair = (min(i, j), max(i, j))  # canonical order
                    fused[pair] = max(fused.get(pair, 0), rrf_score)

        return fused

    def _build_contexts(self, entities: list, chunks: list) -> dict[int, list[np.ndarray]]:
        """For each entity, collect chunk embeddings where it appears."""
        chunk_embed_map = {c.id: c.embedding for c in chunks}
        contexts: dict[int, list[np.ndarray]] = {}

        for i, entity in enumerate(entities):
            embs = [chunk_embed_map[cid] for cid in entity.chunk_ids if cid in chunk_embed_map]
            contexts[i] = embs

        return contexts

    def _context_validate(
        self,
        fused: dict[tuple[int, int], float],
        contexts: dict[int, list[np.ndarray]],
    ) -> dict[tuple[int, int], float]:
        """AND gate + MAX cosine context validation."""
        validated: dict[tuple[int, int], float] = {}

        for (i, j), rrf_score in fused.items():
            ctx_i = contexts.get(i, [])
            ctx_j = contexts.get(j, [])
            if not ctx_i or not ctx_j:
                continue  # AND gate: both must have context

            # MAX cosine between any context pair
            max_cos = 0.0
            for ci in ctx_i:
                for cj in ctx_j:
                    norm_i = np.linalg.norm(ci)
                    norm_j = np.linalg.norm(cj)
                    if norm_i > 0 and norm_j > 0:
                        cos = float(np.dot(ci, cj) / (norm_i * norm_j))
                        max_cos = max(max_cos, cos)

            # AND gate: context sim must also pass threshold
            if max_cos < self.config.synonym_ctx_threshold:
                continue

            confidence = rrf_score * max_cos
            if confidence > 0:
                validated[(i, j)] = confidence

        return validated

    def _build_matrix(self, validated: dict[tuple[int, int], float], N: int) -> sp.csr_matrix:
        """Build symmetric S_conf matrix with diagonal = 1."""
        rows, cols, data = [], [], []

        # Symmetric off-diagonal entries
        for (i, j), conf in validated.items():
            rows.extend([i, j])
            cols.extend([j, i])
            data.extend([conf, conf])

        # Diagonal = 1
        for i in range(N):
            rows.append(i)
            cols.append(i)
            data.append(1.0)

        return sp.csr_matrix((data, (rows, cols)), shape=(N, N), dtype=np.float32)

    def save(self, S_conf: sp.csr_matrix, storage_path: str):
        """Save synonym matrix to disk."""
        from pathlib import Path
        path = Path(storage_path)
        path.mkdir(parents=True, exist_ok=True)
        sp.save_npz(str(path / "synonym_conf.npz"), S_conf)
        logger.info(f"Saved S_conf to {path / 'synonym_conf.npz'}")

    def load(self, storage_path: str) -> sp.csr_matrix:
        """Load synonym matrix from disk."""
        from pathlib import Path
        return sp.load_npz(str(Path(storage_path) / "synonym_conf.npz"))
