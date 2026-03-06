"""State vector initialization for DAG propagation.
NER-based seeding with fallback to direct query-entity matching."""

from __future__ import annotations

import importlib

import numpy as np

from src.config import Config

_logging = importlib.import_module("src.logging-setup")
logger = _logging.get_logger("state-init")


class StateInitializer:
    """Initialize entity state vectors for sub-questions."""

    def __init__(self, config: Config):
        self.config = config
        self.embed_model = None
        self.ner_extractor = None  # reuse from indexing

    def set_embed_model(self, model):
        self.embed_model = model

    def set_ner_extractor(self, extractor):
        """Reuse NER extractor from indexing for query entity extraction."""
        self.ner_extractor = extractor

    def initialize(
        self,
        sub_question_text: str,
        entity_embeddings: np.ndarray,
        parent_state: np.ndarray | None = None,
    ) -> np.ndarray:
        """Initialize state vector s_0 for a sub-question.

        Args:
            sub_question_text: resolved sub-question text
            entity_embeddings: [N x dim] entity embeddings
            parent_state: parent sub-question's final state (if non-root)

        Returns:
            s_0: [N] initial state vector
        """
        N = entity_embeddings.shape[0]
        s_0 = np.zeros(N, dtype=np.float64)
        tau = self.config.ppr_tau_init

        # Step 1: NER-based seeding
        sq_entities = self._extract_entities(sub_question_text)

        if sq_entities:
            sq_embs = self.embed_model.encode(sq_entities, show_progress_bar=False, convert_to_numpy=True)
            # Normalize entity embeddings for cosine
            ent_norms = np.linalg.norm(entity_embeddings, axis=1, keepdims=True)
            ent_norms = np.where(ent_norms == 0, 1, ent_norms)
            ent_normed = entity_embeddings / ent_norms

            for sq_emb in sq_embs:
                sq_norm = np.linalg.norm(sq_emb)
                if sq_norm == 0:
                    continue
                sq_emb_normed = sq_emb / sq_norm
                sims = ent_normed @ sq_emb_normed  # [N]
                mask = sims >= tau
                s_0 = np.maximum(s_0, sims * mask)

        # Step 2: Fallback — direct query-entity matching
        if s_0.sum() == 0:
            sq_emb = self.embed_model.encode([sub_question_text], show_progress_bar=False, convert_to_numpy=True)[0]
            sq_norm = np.linalg.norm(sq_emb)
            if sq_norm > 0:
                ent_norms = np.linalg.norm(entity_embeddings, axis=1)
                ent_norms = np.where(ent_norms == 0, 1, ent_norms)
                sims = (entity_embeddings @ sq_emb) / (ent_norms * sq_norm)
                mask = sims >= tau
                s_0 = np.maximum(0, sims * mask)

        # Step 3: Parent seeding for non-root nodes
        if parent_state is not None:
            alpha_p = self.config.ppr_alpha_parent
            s_0 = np.maximum(s_0, alpha_p * parent_state)

        # Normalize
        max_val = s_0.max()
        if max_val > 0:
            s_0 = s_0 / max_val

        active = int((s_0 > 0).sum())
        logger.debug(f"State init: {active}/{N} active entities "
                    f"(NER found {len(sq_entities)} query entities)")
        return s_0

    def _extract_entities(self, text: str) -> list[str]:
        """Quick NER on sub-question text. Returns list of entity strings."""
        if self.ner_extractor is None:
            return []

        # Create a minimal chunk-like object for the NER extractor
        class _FakeChunk:
            def __init__(self, text):
                self.id = "_query_"
                self.text = text

        try:
            raw = self.ner_extractor.extract_from_chunk(_FakeChunk(text))
            return [ent_text for ent_text, _ in raw]
        except Exception as e:
            logger.debug(f"NER extraction failed for query text: {e}")
            return []
