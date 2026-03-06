"""DAG-guided matrix propagation with dynamic logic modulation.
Core retrieval algorithm: damped PPR on hypergraph per sub-question."""

from __future__ import annotations

import importlib

import numpy as np
import scipy.sparse as sp

from src.config import Config

_logging = importlib.import_module("src.logging-setup")
_tt = importlib.import_module("src.token-tracker")
logger = _logging.get_logger("propagator")


class DAGPropagator:
    """Run DAG-guided propagation retrieval."""

    def __init__(self, config: Config, token_tracker: _tt.TokenTracker):
        self.config = config
        self.token_tracker = token_tracker
        self.embed_model = None

    def set_embed_model(self, model):
        self.embed_model = model

    def propagate(
        self,
        dag,  # QueryDAG
        H_norm: sp.csr_matrix,
        S_conf: sp.csr_matrix,
        semantic_mask,  # SemanticMask
        chunks: list,
        entities: list,
        entity_embeddings: np.ndarray,
        state_initializer,
        query_decomposer,
        query_id: str,
    ) -> tuple[dict, dict]:
        """Run DAG-guided propagation. Returns (entity_states, intermediate_answers)."""
        results: dict[str, np.ndarray] = {}
        intermediate_answers: dict[str, str] = {}
        chunk_embeddings = np.array([c.embedding for c in chunks])

        for sq_id in dag.topological_order:
            sq = next(s for s in dag.sub_questions if s.id == sq_id)

            # Resolve placeholders with intermediate answers
            resolved_text = query_decomposer.resolve_placeholders(sq, intermediate_answers)
            logger.info(f"Processing sub-question '{sq_id}': {resolved_text[:80]}...")

            # Get parent state
            parent_state = None
            if sq.depends_on:
                parent_states = [results[pid] for pid in sq.depends_on if pid in results]
                if parent_states:
                    parent_state = np.mean(parent_states, axis=0)

            # Initialize state
            s_0 = state_initializer.initialize(resolved_text, entity_embeddings, parent_state)

            # Run propagation
            s_final = self._propagate_single(
                resolved_text, s_0, H_norm, S_conf, semantic_mask,
                chunks, chunk_embeddings,
            )
            results[sq_id] = s_final

            # Generate intermediate answer for non-leaf nodes
            if self._has_children(sq_id, dag):
                answer = self._generate_intermediate_answer(
                    resolved_text, s_final, chunks, entities, query_id,
                )
                intermediate_answers[sq_id] = answer

        return results, intermediate_answers

    def _propagate_single(
        self,
        sq_text: str,
        s_0: np.ndarray,
        H_norm: sp.csr_matrix,
        S_conf: sp.csr_matrix,
        semantic_mask,
        chunks: list,
        chunk_embeddings: np.ndarray,
    ) -> np.ndarray:
        """Damped PPR with dynamic modulation for one sub-question."""
        alpha = self.config.ppr_alpha
        M = H_norm.shape[1]

        # Guidance vector
        sq_emb = self.embed_model.encode([sq_text], show_progress_bar=False, convert_to_numpy=True)[0]
        sq_emb_norm = sq_emb / (np.linalg.norm(sq_emb) + 1e-8)

        # Semantic masking — get active clusters
        active_clusters = self._get_active_clusters(sq_emb_norm, semantic_mask)
        active_chunk_ids = set()
        for k in active_clusters:
            active_chunk_ids.update(semantic_mask.cluster_chunk_map.get(k, []))
        # Also include noise points
        for cid, lbl in semantic_mask.chunk_labels.items():
            if lbl == -1:
                active_chunk_ids.add(cid)

        # Build dynamic modulation D_i
        d_diag = np.zeros(M, dtype=np.float64)
        chunk_norms = np.linalg.norm(chunk_embeddings, axis=1)
        chunk_norms = np.where(chunk_norms == 0, 1, chunk_norms)
        chunk_normed = chunk_embeddings / chunk_norms[:, np.newaxis]

        # Vectorized cosine
        cos_sims = chunk_normed @ sq_emb_norm  # [M]
        cos_sims = np.maximum(0, cos_sims)  # ReLU

        # Apply mask
        for j, chunk in enumerate(chunks):
            if chunk.id in active_chunk_ids:
                d_diag[j] = cos_sims[j]

        D_i = sp.diags(d_diag)

        # Propagation matrix: A_i = H_norm * D_i * H_norm^T + S_conf
        A_i = H_norm @ D_i @ H_norm.T + S_conf

        # Damped PPR iteration
        s_t = s_0.copy()
        n_iters = 0
        for t in range(self.config.ppr_max_iterations):
            As = A_i @ s_t
            max_val = np.abs(As).max()
            if max_val > 0:
                As = As / max_val

            s_new = (1 - alpha) * s_0 + alpha * As

            # Dynamic pruning
            s_new[s_new < self.config.ppr_delta_pruning] = 0.0

            # Convergence check
            prev_norm = np.linalg.norm(s_t) + 1e-10
            diff = np.linalg.norm(s_new - s_t) / prev_norm
            logger.debug(f"  Iter {t + 1}: diff={diff:.6f}, active={np.count_nonzero(s_new)}")

            s_t = s_new
            n_iters = t + 1
            if diff < self.config.ppr_epsilon_conv:
                logger.debug(f"  Converged at iter {n_iters}")
                break

        logger.info(f"  Propagation: {n_iters} iters, {np.count_nonzero(s_t)} active entities")
        return s_t

    def _get_active_clusters(self, query_emb_norm: np.ndarray, semantic_mask) -> list[int]:
        """Get top-P clusters most similar to query."""
        centroids = semantic_mask.cluster_centroids  # [K x dim]
        sims = centroids @ query_emb_norm  # [K]
        top_p = min(self.config.ppr_top_p_clusters, len(sims))
        top_indices = np.argsort(sims)[-top_p:][::-1]
        return top_indices.tolist()

    def _has_children(self, sq_id: str, dag) -> bool:
        return any(sq_id in sq.depends_on for sq in dag.sub_questions)

    def _generate_intermediate_answer(
        self, sq_text: str, s_final: np.ndarray,
        chunks: list, entities: list, query_id: str,
    ) -> str:
        """Generate intermediate answer for non-leaf sub-questions."""
        import litellm

        # Get top activated entities
        top_ent_idx = np.argsort(s_final)[-5:][::-1]
        top_ents = [entities[i].text for i in top_ent_idx if s_final[i] > 0]

        # Get top chunks by entity activation
        chunk_scores = {}
        for ent_idx in top_ent_idx:
            if s_final[ent_idx] == 0:
                continue
            for cid in entities[ent_idx].chunk_ids:
                chunk_scores[cid] = max(chunk_scores.get(cid, 0), s_final[ent_idx])

        top_chunk_ids = sorted(chunk_scores, key=chunk_scores.get, reverse=True)[:3]
        chunk_map = {c.id: c for c in chunks}
        context = "\n".join(chunk_map[cid].text for cid in top_chunk_ids if cid in chunk_map)

        prompt = f"""Answer this question briefly based on the context.

Context: {context}

Key entities: {', '.join(top_ents)}

Question: {sq_text}

Answer in one sentence:"""

        response = litellm.completion(
            model=self.config.llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.config.llm_temperature,
            max_tokens=100,
        )

        self.token_tracker.record(
            phase="retrieval",
            doc_or_query_id=query_id,
            model=self.config.llm_model,
            input_tokens=response.usage.prompt_tokens,
            output_tokens=response.usage.completion_tokens,
        )

        answer = response.choices[0].message.content.strip()
        logger.debug(f"  Intermediate answer: {answer[:80]}...")
        return answer
