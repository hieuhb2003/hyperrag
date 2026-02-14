"""
Entity Linker — detects synonyms across chunks.

Builds a soft-link synonym matrix S (N x N) using:
  - Jaro-Winkler lexical similarity (fast pre-filter)
  - Cosine semantic similarity (embedding-based)
"""

import numpy as np
from scipy import sparse
from typing import Optional

from src.utils.logger import get_logger, get_progress
from src.utils.embedding import EmbeddingModel

logger = get_logger(__name__)


class EntityLinker:
    """Build synonym matrix S for entity soft-linking."""

    def __init__(self, config: dict, embedder: Optional[EmbeddingModel] = None):
        self.lexical_prefilter = config.get("lexical_prefilter", 0.7)
        self.synonym_threshold = config.get("synonym_threshold", 0.9)
        self.lexical_weight = config.get("lexical_weight", 0.3)
        self.semantic_weight = config.get("semantic_weight", 0.7)
        self.short_entity_max_len = config.get("short_entity_max_len", 5)
        self.short_entity_threshold = config.get("short_entity_threshold", 0.85)
        self.exclude_numeric_dates = config.get("exclude_numeric_dates", True)
        self._embedder = embedder

        logger.step(
            "EntityLinker",
            "Initialized",
            lex_prefilter=self.lexical_prefilter,
            synonym_thresh=self.synonym_threshold,
            short_max_len=self.short_entity_max_len,
            short_thresh=self.short_entity_threshold,
        )

    def set_embedder(self, embedder: EmbeddingModel) -> None:
        self._embedder = embedder

    @staticmethod
    def _is_numeric_or_date(name: str) -> bool:
        """Check if entity is purely numeric or a date-like pattern."""
        import re
        stripped = name.strip().replace(",", "").replace(".", "")
        if stripped.isdigit():
            return True
        if re.match(r'^\d{1,4}[-/]\d{1,2}[-/]\d{1,4}$', name.strip()):
            return True
        return False

    def build_synonym_matrix(
        self, entities: list, entity_embeddings: Optional[np.ndarray] = None
    ) -> sparse.csr_matrix:
        """
        Build synonym matrix S (N x N).

        Args:
            entities: list of Entity objects with .name and .entity_type
            entity_embeddings: precomputed embeddings (N x dim). If None, will compute.

        Returns:
            S: sparse synonym matrix (N x N)
        """
        import jellyfish

        N = len(entities)
        logger.start_timer("synonym_matrix")
        logger.debug(f"Building synonym matrix for {N} entities")

        if N == 0:
            return sparse.csr_matrix((0, 0))

        # Compute entity embeddings if not provided
        if entity_embeddings is None:
            if self._embedder is None:
                raise ValueError("No embedder provided and no precomputed embeddings")
            entity_names = [e.name for e in entities]
            entity_embeddings = self._embedder.encode(entity_names)

        # Group entities by type for within-type comparison
        type_groups: dict[str, list[int]] = {}
        for idx, ent in enumerate(entities):
            t = ent.entity_type
            if t not in type_groups:
                type_groups[t] = []
            type_groups[t].append(idx)

        logger.debug(f"  Entity type groups: {{{', '.join(f'{k}: {len(v)}' for k, v in type_groups.items())}}}")

        # Build S using lil_matrix (efficient for construction)
        S = sparse.lil_matrix((N, N), dtype=np.float64)

        # Set diagonal
        for i in range(N):
            S[i, i] = 1.0

        pair_count = 0
        synonym_count = 0

        with get_progress() as progress:
            total_groups = len(type_groups)
            task = progress.add_task("Entity linking...", total=total_groups)

            for entity_type, indices in type_groups.items():
                if len(indices) < 2:
                    progress.advance(task)
                    continue

                for i_pos in range(len(indices)):
                    idx_i = indices[i_pos]
                    ent_i = entities[idx_i]

                    # Skip numeric/date entities
                    if self.exclude_numeric_dates and self._is_numeric_or_date(ent_i.name):
                        continue

                    for j_pos in range(i_pos + 1, len(indices)):
                        idx_j = indices[j_pos]
                        ent_j = entities[idx_j]

                        if self.exclude_numeric_dates and self._is_numeric_or_date(ent_j.name):
                            continue

                        pair_count += 1

                        len_min = min(len(ent_i.name), len(ent_j.name))

                        if len_min <= self.short_entity_max_len:
                            # SHORT ENTITY PATH: bypass lexical pre-filter,
                            # rely entirely on semantic similarity.
                            # Catches acronyms like WHO/World Health Organization
                            # where JW score is too low to pass lexical pre-filter.
                            sem_sim = float(
                                np.dot(entity_embeddings[idx_i], entity_embeddings[idx_j])
                            )
                            if sem_sim > self.short_entity_threshold:
                                S[idx_i, idx_j] = sem_sim
                                S[idx_j, idx_i] = sem_sim
                                synonym_count += 1
                                logger.debug(
                                    f"  Synonym found (short-entity): '{ent_i.name}' ↔ '{ent_j.name}' "
                                    f"(sem={sem_sim:.3f})"
                                )
                        else:
                            # STANDARD PATH: lexical pre-filter + combined score
                            lex_sim = jellyfish.jaro_winkler_similarity(
                                ent_i.name.lower(), ent_j.name.lower()
                            )
                            if lex_sim < self.lexical_prefilter:
                                continue

                            sem_sim = float(
                                np.dot(entity_embeddings[idx_i], entity_embeddings[idx_j])
                            )
                            combined = (
                                self.lexical_weight * lex_sim
                                + self.semantic_weight * sem_sim
                            )

                            if combined > self.synonym_threshold:
                                S[idx_i, idx_j] = combined
                                S[idx_j, idx_i] = combined
                                synonym_count += 1
                                logger.debug(
                                    f"  Synonym found: '{ent_i.name}' ↔ '{ent_j.name}' "
                                    f"(lex={lex_sim:.3f}, sem={sem_sim:.3f}, combined={combined:.3f})"
                                )

                progress.advance(task)

        S = S.tocsr()
        elapsed = logger.stop_timer("synonym_matrix")
        logger.step(
            "EntityLinker",
            f"Synonym matrix built: {synonym_count} synonym pairs from {pair_count} comparisons",
            shape=S.shape,
            nnz=S.nnz,
            time=elapsed,
        )
        logger.matrix("Synonym Matrix S", S)

        return S

    def expand_synonym_matrix(
        self,
        old_S: sparse.csr_matrix,
        old_entities: list,
        new_entities: list,
        old_entity_embs: np.ndarray,
        new_entity_embs: np.ndarray,
    ) -> sparse.csr_matrix:
        """
        Expand synonym matrix S for incremental updates.

        Grows S from (N_old, N_old) to (N_old + N_new, N_old + N_new):
        - Top-left block: old S (preserved exactly)
        - Bottom-right block: new-to-new entity links
        - Off-diagonal blocks: new-to-old entity links (cross-links)

        Args:
            old_S: existing synonym matrix (N_old x N_old)
            old_entities: list of existing Entity objects
            new_entities: list of new Entity objects
            old_entity_embs: embeddings for old entities (N_old x dim)
            new_entity_embs: embeddings for new entities (N_new x dim)

        Returns:
            Expanded S: (N_old + N_new, N_old + N_new) sparse matrix
        """
        import jellyfish

        N_old = len(old_entities)
        N_new = len(new_entities)
        N_total = N_old + N_new

        logger.start_timer("expand_synonym")
        logger.debug(f"Expanding synonym matrix: {N_old} + {N_new} = {N_total} entities")

        # Start with expanded matrix, copy old S into top-left
        S = sparse.lil_matrix((N_total, N_total), dtype=np.float64)

        # Copy old S
        old_S_coo = old_S.tocoo()
        for row, col, val in zip(old_S_coo.row, old_S_coo.col, old_S_coo.data):
            S[row, col] = val

        # Set diagonal for new entities
        for i in range(N_old, N_total):
            S[i, i] = 1.0

        # Group new entities by type
        new_type_groups: dict[str, list[int]] = {}
        for idx, ent in enumerate(new_entities):
            t = ent.entity_type
            if t not in new_type_groups:
                new_type_groups[t] = []
            new_type_groups[t].append(idx)

        # Group old entities by type
        old_type_groups: dict[str, list[int]] = {}
        for idx, ent in enumerate(old_entities):
            t = ent.entity_type
            if t not in old_type_groups:
                old_type_groups[t] = []
            old_type_groups[t].append(idx)

        synonym_count = 0

        # New-to-new links (within new entities, same type)
        for entity_type, new_indices in new_type_groups.items():
            if len(new_indices) < 2:
                continue
            for i_pos in range(len(new_indices)):
                ni = new_indices[i_pos]
                ent_i = new_entities[ni]
                if self.exclude_numeric_dates and self._is_numeric_or_date(ent_i.name):
                    continue
                for j_pos in range(i_pos + 1, len(new_indices)):
                    nj = new_indices[j_pos]
                    ent_j = new_entities[nj]
                    if self.exclude_numeric_dates and self._is_numeric_or_date(ent_j.name):
                        continue

                    len_min = min(len(ent_i.name), len(ent_j.name))
                    gi = N_old + ni
                    gj = N_old + nj

                    if len_min <= self.short_entity_max_len:
                        sem_sim = float(np.dot(new_entity_embs[ni], new_entity_embs[nj]))
                        if sem_sim > self.short_entity_threshold:
                            S[gi, gj] = sem_sim
                            S[gj, gi] = sem_sim
                            synonym_count += 1
                    else:
                        lex_sim = jellyfish.jaro_winkler_similarity(
                            ent_i.name.lower(), ent_j.name.lower()
                        )
                        if lex_sim < self.lexical_prefilter:
                            continue

                        sem_sim = float(np.dot(new_entity_embs[ni], new_entity_embs[nj]))
                        combined = self.lexical_weight * lex_sim + self.semantic_weight * sem_sim

                        if combined > self.synonym_threshold:
                            S[gi, gj] = combined
                            S[gj, gi] = combined
                            synonym_count += 1

        # New-to-old links (cross new/old, same type)
        for entity_type, new_indices in new_type_groups.items():
            old_indices = old_type_groups.get(entity_type, [])
            if not old_indices:
                continue
            for ni in new_indices:
                ent_new = new_entities[ni]
                if self.exclude_numeric_dates and self._is_numeric_or_date(ent_new.name):
                    continue
                for oi in old_indices:
                    ent_old = old_entities[oi]
                    if self.exclude_numeric_dates and self._is_numeric_or_date(ent_old.name):
                        continue

                    len_min = min(len(ent_new.name), len(ent_old.name))
                    gi = N_old + ni

                    if len_min <= self.short_entity_max_len:
                        sem_sim = float(np.dot(new_entity_embs[ni], old_entity_embs[oi]))
                        if sem_sim > self.short_entity_threshold:
                            S[gi, oi] = sem_sim
                            S[oi, gi] = sem_sim
                            synonym_count += 1
                    else:
                        lex_sim = jellyfish.jaro_winkler_similarity(
                            ent_new.name.lower(), ent_old.name.lower()
                        )
                        if lex_sim < self.lexical_prefilter:
                            continue

                        sem_sim = float(np.dot(new_entity_embs[ni], old_entity_embs[oi]))
                        combined = self.lexical_weight * lex_sim + self.semantic_weight * sem_sim

                        if combined > self.synonym_threshold:
                            S[gi, oi] = combined
                            S[oi, gi] = combined
                            synonym_count += 1

        S = S.tocsr()
        elapsed = logger.stop_timer("expand_synonym")
        logger.step(
            "EntityLinker",
            f"Synonym matrix expanded: {synonym_count} new synonym pairs",
            shape=S.shape,
            nnz=S.nnz,
            time=elapsed,
        )
        logger.matrix("Expanded Synonym Matrix S", S)

        return S
