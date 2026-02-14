"""
DAG-Guided Matrix Propagation — CORE ALGORITHM of HyP-DLM.

Implements:
  - Damped PPR propagation (primary — guaranteed convergence)
  - MAX update propagation (ablation baseline)
  - Full DAG propagation loop
"""

import numpy as np
from scipy import sparse
from typing import Optional

from src.utils.logger import get_logger
from src.utils.sparse_ops import (
    build_propagation_matrix,
    compute_modulation_matrix,
    cosine_sim_vec,
    sparse_diag,
    top_k_indices,
)
from src.utils.embedding import EmbeddingModel
from src.indexing.masking_strategy import MaskingStrategy
from src.indexing.ner_extractor import NERExtractor
from src.retrieval.query_decomposer import QueryDAG, QueryDecomposer

logger = get_logger(__name__)


# ══════════════════════════════════════════════════════════
# Propagation Strategies
# ══════════════════════════════════════════════════════════


def propagate_damped_ppr(
    s_0: np.ndarray,
    A_i: sparse.csr_matrix,
    alpha: float = 0.85,
    max_hops: int = 6,
    pruning_threshold: float = 0.05,
    convergence_eps: float = 0.01,
) -> tuple[np.ndarray, int, list[dict]]:
    """
    Damped Personalized PageRank on hypergraph.

    s_{t+1} = (1 - alpha) * s_0 + alpha * normalize(A_i @ s_t)

    Guaranteed convergence at rate O(alpha^t).

    Returns:
        s_final: converged state vector
        hops_used: actual number of steps
        history: list of per-step metrics for diagnostics
    """
    s_t = s_0.copy()
    history = []

    logger.debug(
        f"  PPR propagation: alpha={alpha}, max_hops={max_hops}, "
        f"prune_thresh={pruning_threshold}, eps={convergence_eps}"
    )
    logger.debug(
        f"  Initial state: {np.sum(s_t > 0)} active entities, "
        f"max={s_t.max():.4f}"
    )

    for t in range(max_hops):
        # Propagate through hypergraph
        propagated = A_i @ s_t

        # Normalize to [0, 1]
        max_val = propagated.max() if len(propagated) > 0 else 0
        if max_val > 0:
            propagated = propagated / max_val

        # Damped PPR update
        s_new = (1 - alpha) * s_0 + alpha * propagated

        # Pruning
        s_new[s_new < pruning_threshold] = 0.0

        # Convergence check
        norm_st = np.linalg.norm(s_t)
        delta = np.linalg.norm(s_new - s_t) / (norm_st + 1e-8)
        active = int(np.sum(s_new > 0))
        max_score = float(s_new.max()) if len(s_new) > 0 else 0.0

        step_info = {
            "step": t + 1,
            "delta": float(delta),
            "active_entities": active,
            "max_score": max_score,
        }
        history.append(step_info)

        logger.convergence(t + 1, delta, active, max_score)

        s_t = s_new

        if delta < convergence_eps:
            logger.debug(f"  PPR converged at step {t + 1} (delta={delta:.6f} < {convergence_eps})")
            break

    return s_t, t + 1, history


def propagate_max_update(
    s_0: np.ndarray,
    A_i: sparse.csr_matrix,
    max_hops: int = 6,
    pruning_threshold: float = 0.05,
    convergence_min_new: int = 2,
) -> tuple[np.ndarray, int, list[dict]]:
    """
    MAX update propagation (ablation baseline).

    s_{t+1} = max(normalize(A_i @ s_t), s_t)

    No convergence guarantee but preserves all discovered entities.
    """
    s_t = s_0.copy()
    history = []

    logger.debug(
        f"  MAX propagation: max_hops={max_hops}, "
        f"prune_thresh={pruning_threshold}, min_new={convergence_min_new}"
    )

    for t in range(max_hops):
        propagated = A_i @ s_t

        max_val = propagated.max() if len(propagated) > 0 else 0
        if max_val > 0:
            propagated = propagated / max_val

        # MAX update
        s_new = np.maximum(propagated, s_t)

        # Pruning
        s_new[s_new < pruning_threshold] = 0.0

        # Check new activations
        new_activated = int(np.sum((s_new > 0) & (s_t == 0)))
        active = int(np.sum(s_new > 0))
        max_score = float(s_new.max()) if len(s_new) > 0 else 0.0

        step_info = {
            "step": t + 1,
            "new_activated": new_activated,
            "active_entities": active,
            "max_score": max_score,
        }
        history.append(step_info)

        logger.convergence(t + 1, float(new_activated), active, max_score)

        s_t = s_new

        if new_activated < convergence_min_new:
            logger.debug(
                f"  MAX converged at step {t + 1} "
                f"(new_activated={new_activated} < {convergence_min_new})"
            )
            break

    return s_t, t + 1, history


# ══════════════════════════════════════════════════════════
# Full DAG Propagation
# ══════════════════════════════════════════════════════════


class DAGPropagation:
    """Full DAG-guided propagation loop."""

    def __init__(self, config: dict):
        self.strategy = config.get("strategy", "damped_ppr")
        self.alpha = config.get("alpha", 0.85)
        self.convergence_eps = config.get("convergence_eps", 0.01)
        self.convergence_min_new = config.get("convergence_min_new", 2)
        self.max_hops = config.get("max_hops", 6)
        self.top_p_clusters = config.get("top_p_clusters", 5)
        self.pruning_threshold = config.get("pruning_threshold", 0.05)
        self.entity_top_k = config.get("entity_top_k", 30)
        self.hyperedge_top_k = config.get("hyperedge_top_k", 30)
        self.synonym_weight = config.get("synonym_weight", 0.3)
        self.alpha_parent = config.get("alpha_parent", 0.5)

        logger.step(
            "DAGPropagation",
            f"Initialized with strategy='{self.strategy}'",
            alpha=self.alpha,
            max_hops=self.max_hops,
        )

    def propagate(
        self,
        dag: QueryDAG,
        H: sparse.csr_matrix,
        S: sparse.csr_matrix,
        masking_strategy: MaskingStrategy,
        entity_embeddings: np.ndarray,
        hyperedge_embeddings: np.ndarray,
        embedder: EmbeddingModel,
        ner_extractor: NERExtractor,
        llm_client=None,
        chunks: list = None,
    ) -> dict:
        """
        Run full DAG-guided propagation.

        Returns:
            {
                "hyperedge_scores": dict[int, float],
                "entity_scores": dict[int, float],
                "answers": dict[str, str],
                "convergence_log": list[dict],
                "all_activated_entities": dict[str, np.ndarray],
            }
        """
        logger.start_timer("dag_propagation")
        logger.step(
            "DAGPropagation",
            f"Starting propagation for {len(dag.nodes)} sub-questions",
        )

        N, M = H.shape
        answers: dict[str, str] = {}
        all_activated_entities: dict[str, np.ndarray] = {}
        all_activated_hyperedges: dict[str, list[int]] = {}
        convergence_log = []

        topo_order = dag.topological_order()

        for qi_idx, q_i in enumerate(topo_order):
            logger.debug(f"\n{'='*60}")
            logger.debug(
                f"Sub-question {qi_idx+1}/{len(topo_order)}: "
                f"[{q_i.id}] '{q_i.question}'"
            )

            # 1. Resolve references
            resolved_question = QueryDecomposer.resolve_references(
                q_i.question, answers
            )

            # 2. Compute guidance vector (FIXED for this sub-question)
            g_i = embedder.encode_single(resolved_question)
            logger.debug(f"  Guidance vector: norm={np.linalg.norm(g_i):.4f}")

            # 3. Initialize state vector s_0
            s_0 = self._initialize_state(
                resolved_question, ner_extractor, entity_embeddings, embedder, N
            )

            # 4. Seed from parent sub-questions
            for parent_id in q_i.depends_on:
                if parent_id in all_activated_entities:
                    parent_state = all_activated_entities[parent_id]
                    s_0 = np.maximum(s_0, self.alpha_parent * parent_state)
                    logger.debug(
                        f"  Seeded from parent '{parent_id}': "
                        f"{np.sum(parent_state > 0)} active entities"
                    )

            logger.debug(
                f"  s_0: {int(np.sum(s_0 > 0))} active entities, "
                f"max={s_0.max():.4f}"
            )

            # 5. Compute FIXED propagation components
            # Phase A: Semantic Masking
            mask = masking_strategy.compute_mask(g_i, top_p=self.top_p_clusters)
            logger.debug(
                f"  Mask: {int(np.sum(mask))}/{M} hyperedges "
                f"({np.sum(mask)/M*100:.1f}%)"
            )

            # Phase B: Dynamic Modulation
            D_i = compute_modulation_matrix(g_i, hyperedge_embeddings, mask)

            # Precompute propagation matrix A_i (FIXED per sub-question)
            A_i = build_propagation_matrix(H, D_i, S, self.synonym_weight)

            # 6. Run propagation
            if self.strategy == "damped_ppr":
                s_final, hops_used, history = propagate_damped_ppr(
                    s_0=s_0,
                    A_i=A_i,
                    alpha=self.alpha,
                    max_hops=self.max_hops,
                    pruning_threshold=self.pruning_threshold,
                    convergence_eps=self.convergence_eps,
                )
            elif self.strategy == "max_update":
                s_final, hops_used, history = propagate_max_update(
                    s_0=s_0,
                    A_i=A_i,
                    max_hops=self.max_hops,
                    pruning_threshold=self.pruning_threshold,
                    convergence_min_new=self.convergence_min_new,
                )
            else:
                raise ValueError(f"Unknown propagation strategy: {self.strategy}")

            convergence_log.append({
                "sub_question": q_i.id,
                "hops_used": hops_used,
                "final_active": int(np.sum(s_final > 0)),
                "history": history,
            })

            # 7. Extract top entities and hyperedges
            top_entity_ids = top_k_indices(s_final, k=self.entity_top_k)
            activated_he = H.T @ (s_final > 0).astype(float)
            attn = cosine_sim_vec(g_i, hyperedge_embeddings)
            attn = (attn + 1.0) / 2.0
            he_scores = activated_he * attn
            top_he_ids = top_k_indices(he_scores, k=self.hyperedge_top_k)

            logger.debug(
                f"  Result: {hops_used} hops, "
                f"{int(np.sum(s_final > 0))} active entities, "
                f"{len(top_he_ids)} top hyperedges"
            )

            # 8. Generate intermediate answer (if there are dependent sub-questions)
            has_dependents = any(
                q_i.id in other.depends_on
                for other in dag.nodes
                if other.id != q_i.id
            )
            if has_dependents and llm_client and chunks:
                context = "\n\n".join(
                    chunks[idx].text for idx in top_he_ids if idx < len(chunks)
                )
                answer = llm_client.generate(
                    f"Based on the following context, briefly answer: {resolved_question}\n\nContext:\n{context}",
                    max_tokens=200,
                )
                answers[q_i.id] = answer.strip()
                logger.debug(f"  Intermediate answer [{q_i.id}]: '{answers[q_i.id][:100]}...'")
            else:
                answers[q_i.id] = ""

            # 9. Accumulate
            all_activated_entities[q_i.id] = s_final
            all_activated_hyperedges[q_i.id] = top_he_ids.tolist()

        # 10. Combine all activated hyperedges
        combined_he_scores = self._aggregate_hyperedge_scores(
            all_activated_hyperedges, M, hyperedge_embeddings, embedder, dag
        )

        # Combined entity scores
        combined_entity_scores = {}
        for q_id, s_vec in all_activated_entities.items():
            for idx in np.where(s_vec > 0)[0]:
                old = combined_entity_scores.get(int(idx), 0.0)
                combined_entity_scores[int(idx)] = max(old, float(s_vec[idx]))

        elapsed = logger.stop_timer("dag_propagation")

        logger.step(
            "DAGPropagation",
            f"Propagation complete",
            total_hyperedges=len(combined_he_scores),
            total_entities=len(combined_entity_scores),
            time=elapsed,
        )

        # Log convergence summary
        logger.summary_table(
            "Propagation Convergence Summary",
            [
                {
                    "Sub-Q": entry["sub_question"],
                    "Hops": entry["hops_used"],
                    "Active Entities": entry["final_active"],
                }
                for entry in convergence_log
            ],
        )

        return {
            "hyperedge_scores": combined_he_scores,
            "entity_scores": combined_entity_scores,
            "answers": answers,
            "convergence_log": convergence_log,
            "all_activated_entities": all_activated_entities,
        }

    def _initialize_state(
        self,
        question: str,
        ner_extractor: NERExtractor,
        entity_embeddings: np.ndarray,
        embedder: EmbeddingModel,
        N: int,
        threshold: float = 0.5,
    ) -> np.ndarray:
        """Initialize activation vector from query entities."""
        s = np.zeros(N)

        # Extract entities from the question
        q_entities = ner_extractor.extract(question)
        logger.debug(
            f"  Query entities: {[e.name for e in q_entities]}"
        )

        if not q_entities:
            # Fallback: use query embedding directly against entity embeddings
            q_emb = embedder.encode_single(question)
            sims = cosine_sim_vec(q_emb, entity_embeddings)
            top_idx = np.argmax(sims)
            if sims[top_idx] > threshold:
                s[top_idx] = float(sims[top_idx])
                logger.debug(
                    f"  No NER entities → fallback to embedding match: "
                    f"idx={top_idx}, score={sims[top_idx]:.4f}"
                )
        else:
            for qe in q_entities:
                qe_emb = embedder.encode_single(qe.name)
                sims = cosine_sim_vec(qe_emb, entity_embeddings)
                best_idx = np.argmax(sims)
                if sims[best_idx] > threshold:
                    s[best_idx] = float(sims[best_idx])
                    logger.debug(
                        f"  Entity '{qe.name}' → idx={best_idx}, "
                        f"score={sims[best_idx]:.4f}"
                    )

        return s

    def _aggregate_hyperedge_scores(
        self,
        all_he: dict[str, list[int]],
        M: int,
        hyperedge_embeddings: np.ndarray,
        embedder: EmbeddingModel,
        dag: QueryDAG,
    ) -> dict[int, float]:
        """Aggregate hyperedge scores across all sub-questions."""
        score_map: dict[int, float] = {}

        # Simple max aggregation
        for q_id, he_ids in all_he.items():
            for rank, he_id in enumerate(he_ids):
                # Score = 1 / (rank + 1)  (rank-based fusion)
                score = 1.0 / (rank + 1)
                old = score_map.get(he_id, 0.0)
                score_map[he_id] = max(old, score)

        return score_map
