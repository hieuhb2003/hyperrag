"""
Semantic Affinity Matrix (A_sem) Builder for HyP-DLM.

This module builds the "soft-link" matrix that connects entity aliases
without explicit entity resolution. Uses hybrid Lexical + Embedding similarity.
"""

import logging
from typing import Dict, List, Tuple, Optional
import numpy as np
from scipy import sparse
from sklearn.neighbors import NearestNeighbors
import jellyfish  # For Jaro-Winkler distance

logger = logging.getLogger(__name__)


class SemanticAffinityBuilder:
    """
    Builds the Semantic Affinity Matrix A_sem for entity soft-linking.
    
    A_sem[i,j] > 0 means entity i and entity j are likely aliases of each other.
    This allows signal propagation between "L.Messi" and "Leo Messi" without
    explicit entity resolution.
    """
    
    def __init__(
        self,
        lexical_weight: float = 0.5,
        similarity_threshold: float = 0.7,
        top_k_candidates: int = 10,
        propagation_factor: float = 0.3
    ):
        """
        Initialize the builder.
        
        Args:
            lexical_weight: Weight for lexical similarity (α). Embedding weight = 1 - α.
            similarity_threshold: Minimum similarity to include in A_sem (τ).
            top_k_candidates: Number of nearest neighbors to consider per entity.
            propagation_factor: β factor for signal leakage in retrieval formula.
        """
        self.lexical_weight = lexical_weight
        self.embedding_weight = 1 - lexical_weight
        self.similarity_threshold = similarity_threshold
        self.top_k_candidates = top_k_candidates
        self.propagation_factor = propagation_factor
    
    def build(
        self,
        entity_to_idx: Dict[str, int],
        entity_embeddings: np.ndarray
    ) -> sparse.csr_matrix:
        """
        Build the Semantic Affinity Matrix.
        
        Args:
            entity_to_idx: Mapping from entity name to index
            entity_embeddings: Entity embeddings matrix (num_entities x embed_dim)
            
        Returns:
            Sparse matrix A_sem (num_entities x num_entities)
        """
        num_entities = len(entity_to_idx)
        logger.info(f"Building A_sem for {num_entities} entities...")
        
        if num_entities == 0:
            return sparse.csr_matrix((0, 0), dtype=np.float32)
        
        # Step A: Blocking - Find candidate pairs using embedding similarity
        candidate_pairs = self._find_candidate_pairs(entity_embeddings)
        logger.info(f"Found {len(candidate_pairs)} candidate pairs")
        
        # Step B: Scoring - Compute hybrid similarity for each pair
        idx_to_entity = {idx: entity for entity, idx in entity_to_idx.items()}
        scored_pairs = self._score_pairs(candidate_pairs, idx_to_entity, entity_embeddings)
        
        # Step C: Sparsification - Filter by threshold and build matrix
        a_sem = self._build_sparse_matrix(scored_pairs, num_entities)
        
        # Count non-zero entries
        nnz = a_sem.nnz
        logger.info(f"A_sem built: {num_entities}x{num_entities}, {nnz} non-zero entries")
        
        return a_sem
    
    def _find_candidate_pairs(
        self,
        entity_embeddings: np.ndarray
    ) -> List[Tuple[int, int]]:
        """
        Find candidate entity pairs using approximate nearest neighbors.
        
        This is the Blocking step that reduces O(n²) to O(n×k).
        """
        num_entities = entity_embeddings.shape[0]
        
        if num_entities <= self.top_k_candidates:
            # If few entities, consider all pairs
            pairs = []
            for i in range(num_entities):
                for j in range(i + 1, num_entities):
                    pairs.append((i, j))
            return pairs
        
        # Use scikit-learn's NearestNeighbors for efficiency
        nn = NearestNeighbors(
            n_neighbors=min(self.top_k_candidates, num_entities),
            metric='cosine',
            algorithm='auto'
        )
        nn.fit(entity_embeddings)
        
        # Find k nearest neighbors for each entity
        _, indices = nn.kneighbors(entity_embeddings)
        
        # Collect unique pairs
        pairs = set()
        for i in range(num_entities):
            for j in indices[i]:
                if i != j:
                    # Use ordered pair to avoid duplicates
                    pair = (min(i, j), max(i, j))
                    pairs.add(pair)
        
        return list(pairs)
    
    def _score_pairs(
        self,
        candidate_pairs: List[Tuple[int, int]],
        idx_to_entity: Dict[int, str],
        entity_embeddings: np.ndarray
    ) -> List[Tuple[int, int, float]]:
        """
        Compute hybrid similarity score for each candidate pair.
        
        Score = α × Lexical + (1-α) × Embedding
        """
        scored_pairs = []
        
        for i, j in candidate_pairs:
            entity_i = idx_to_entity[i]
            entity_j = idx_to_entity[j]
            
            # Lexical similarity using Jaro-Winkler
            lexical_sim = jellyfish.jaro_winkler_similarity(entity_i, entity_j)
            
            # Embedding similarity using cosine
            embed_i = entity_embeddings[i]
            embed_j = entity_embeddings[j]
            cos_sim = np.dot(embed_i, embed_j) / (
                np.linalg.norm(embed_i) * np.linalg.norm(embed_j) + 1e-8
            )
            # Normalize to [0, 1]
            embedding_sim = (cos_sim + 1) / 2
            
            # Combined score
            score = (
                self.lexical_weight * lexical_sim +
                self.embedding_weight * embedding_sim
            )
            
            if score >= self.similarity_threshold:
                scored_pairs.append((i, j, score))
        
        logger.info(f"{len(scored_pairs)} pairs passed similarity threshold {self.similarity_threshold}")
        return scored_pairs
    
    def _build_sparse_matrix(
        self,
        scored_pairs: List[Tuple[int, int, float]],
        num_entities: int
    ) -> sparse.csr_matrix:
        """Build symmetric sparse matrix from scored pairs."""
        rows = []
        cols = []
        data = []
        
        for i, j, score in scored_pairs:
            # Add both directions for symmetry
            rows.extend([i, j])
            cols.extend([j, i])
            data.extend([score, score])
        
        a_sem = sparse.csr_matrix(
            (data, (rows, cols)),
            shape=(num_entities, num_entities),
            dtype=np.float32
        )
        
        return a_sem
    
    def apply_affinity(
        self,
        entity_scores: np.ndarray,
        a_sem: sparse.csr_matrix
    ) -> np.ndarray:
        """
        Apply semantic affinity to propagate scores to alias entities.
        
        This implements: x_new = x + β × A_sem × x
        
        Args:
            entity_scores: Current entity score vector
            a_sem: Semantic affinity matrix
            
        Returns:
            Updated entity scores with alias propagation
        """
        # Propagate through aliases
        alias_contribution = a_sem.dot(entity_scores) * self.propagation_factor
        
        # Combine with original scores
        updated_scores = entity_scores + alias_contribution
        
        return updated_scores


def save_affinity_matrix(a_sem: sparse.csr_matrix, path: str) -> None:
    """Save the affinity matrix to disk."""
    sparse.save_npz(path, a_sem)
    logger.info(f"A_sem saved to {path}")


def load_affinity_matrix(path: str) -> sparse.csr_matrix:
    """Load the affinity matrix from disk."""
    return sparse.load_npz(path)
