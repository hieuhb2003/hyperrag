"""
Semantic Masking Module for HyP-DLM.

This module implements cluster-based filtering to reduce the search space
during retrieval by 90%+. Only propositions in semantically relevant clusters
are considered.
"""

import logging
from typing import Dict, List, Tuple, Optional
import numpy as np
from sklearn.cluster import KMeans
from scipy import sparse

logger = logging.getLogger(__name__)


class SemanticMasker:
    """
    Implements Semantic Masking using proposition clustering.
    
    During retrieval, the guidance vector is compared to cluster centroids,
    and only propositions in the top-P most relevant clusters are considered.
    """
    
    def __init__(
        self,
        num_clusters: int = 100,
        top_p_clusters: int = 10,
        random_state: int = 42
    ):
        """
        Initialize the masker.
        
        Args:
            num_clusters: Number of clusters to create (K)
            top_p_clusters: Number of top clusters to select during retrieval (P)
            random_state: Random seed for reproducibility
        """
        self.num_clusters = num_clusters
        self.top_p_clusters = top_p_clusters
        self.random_state = random_state
        
        self.kmeans: Optional[KMeans] = None
        self.centroids: Optional[np.ndarray] = None
        self.prop_to_cluster: Optional[np.ndarray] = None
    
    def fit(self, proposition_embeddings: np.ndarray) -> None:
        """
        Fit K-Means clustering on proposition embeddings.
        
        Args:
            proposition_embeddings: Matrix of shape (num_propositions, embed_dim)
        """
        num_props = proposition_embeddings.shape[0]
        actual_k = min(self.num_clusters, num_props)
        
        logger.info(f"Fitting K-Means with K={actual_k} on {num_props} propositions...")
        
        self.kmeans = KMeans(
            n_clusters=actual_k,
            random_state=self.random_state,
            n_init=10,
            max_iter=300
        )
        
        self.prop_to_cluster = self.kmeans.fit_predict(proposition_embeddings)
        self.centroids = self.kmeans.cluster_centers_
        
        # Log cluster distribution
        unique, counts = np.unique(self.prop_to_cluster, return_counts=True)
        logger.info(f"Cluster sizes: min={counts.min()}, max={counts.max()}, mean={counts.mean():.1f}")
    
    def get_mask(
        self,
        guidance_vector: np.ndarray,
        num_propositions: int
    ) -> np.ndarray:
        """
        Create a binary mask for propositions based on guidance vector.
        
        Args:
            guidance_vector: The current guidance vector for this retrieval step
            num_propositions: Total number of propositions
            
        Returns:
            Binary mask array of shape (num_propositions,) where 1 = keep, 0 = filter
        """
        if self.centroids is None or self.prop_to_cluster is None:
            # No clustering done, keep all
            return np.ones(num_propositions, dtype=np.float32)
        
        # Compute similarity to all centroids
        # Normalize for cosine similarity
        guidance_norm = guidance_vector / (np.linalg.norm(guidance_vector) + 1e-8)
        centroids_norm = self.centroids / (
            np.linalg.norm(self.centroids, axis=1, keepdims=True) + 1e-8
        )
        
        similarities = np.dot(centroids_norm, guidance_norm)
        
        # Get top-P clusters
        actual_p = min(self.top_p_clusters, len(similarities))
        top_cluster_indices = np.argsort(similarities)[-actual_p:]
        
        # Create mask: 1 if proposition is in a top cluster, 0 otherwise
        mask = np.zeros(num_propositions, dtype=np.float32)
        for cluster_idx in top_cluster_indices:
            mask[self.prop_to_cluster == cluster_idx] = 1.0
        
        kept = int(mask.sum())
        reduction = (1 - kept / num_propositions) * 100
        logger.debug(f"Semantic mask: kept {kept}/{num_propositions} ({reduction:.1f}% reduced)")
        
        return mask
    
    def apply_mask_to_matrix(
        self,
        incidence_matrix: sparse.csr_matrix,
        mask: np.ndarray
    ) -> sparse.csr_matrix:
        """
        Apply mask to incidence matrix columns (propositions).
        
        Args:
            incidence_matrix: H matrix of shape (num_entities, num_propositions)
            mask: Binary mask of shape (num_propositions,)
            
        Returns:
            Masked incidence matrix
        """
        # Create diagonal mask matrix
        mask_diag = sparse.diags(mask, format='csr')
        
        # Apply mask to columns: H_masked = H @ diag(mask)
        masked_matrix = incidence_matrix.dot(mask_diag)
        
        return masked_matrix
    
    def save(self, path: str) -> None:
        """Save clustering model to disk."""
        import os
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        np.savez(
            path,
            centroids=self.centroids,
            prop_to_cluster=self.prop_to_cluster,
            num_clusters=self.num_clusters,
            top_p_clusters=self.top_p_clusters
        )
        logger.info(f"Semantic masker saved to {path}")
    
    def load(self, path: str) -> None:
        """Load clustering model from disk."""
        data = np.load(path)
        
        self.centroids = data['centroids']
        self.prop_to_cluster = data['prop_to_cluster']
        self.num_clusters = int(data['num_clusters'])
        self.top_p_clusters = int(data['top_p_clusters'])
        
        logger.info(f"Semantic masker loaded from {path}")


class AdaptiveMasker(SemanticMasker):
    """
    Extended masker that adapts P based on query complexity.
    
    For simple queries, use fewer clusters. For complex multi-hop queries,
    use more clusters to avoid missing relevant information.
    """
    
    def __init__(
        self,
        num_clusters: int = 100,
        min_p: int = 5,
        max_p: int = 20,
        complexity_threshold: float = 0.5,
        random_state: int = 42
    ):
        super().__init__(num_clusters, max_p, random_state)
        self.min_p = min_p
        self.max_p = max_p
        self.complexity_threshold = complexity_threshold
    
    def get_adaptive_mask(
        self,
        guidance_vector: np.ndarray,
        query_complexity: float,
        num_propositions: int
    ) -> np.ndarray:
        """
        Create mask with adaptive P based on query complexity.
        
        Args:
            guidance_vector: Current guidance vector
            query_complexity: Score from 0 (simple) to 1 (complex)
            num_propositions: Total number of propositions
            
        Returns:
            Binary mask array
        """
        # Interpolate P based on complexity
        if query_complexity < self.complexity_threshold:
            adaptive_p = self.min_p
        else:
            # Linear interpolation from min_p to max_p
            t = (query_complexity - self.complexity_threshold) / (1 - self.complexity_threshold)
            adaptive_p = int(self.min_p + t * (self.max_p - self.min_p))
        
        # Temporarily set top_p_clusters
        original_p = self.top_p_clusters
        self.top_p_clusters = adaptive_p
        
        mask = self.get_mask(guidance_vector, num_propositions)
        
        # Restore
        self.top_p_clusters = original_p
        
        return mask
