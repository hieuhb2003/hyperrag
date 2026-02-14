"""
Semantic Masking Strategies — reduce hyperedge search space during retrieval.

Implements 4 strategies via Strategy Pattern:
  1. HDBSCAN (primary — recommended)
  2. K-Means (baseline for ablation)
  3. FAISS Direct ANN (production / streaming)
  4. NoMasking (ablation only)
"""

from abc import ABC, abstractmethod

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from src.utils.logger import get_logger

logger = get_logger(__name__)


# ══════════════════════════════════════════════════════════
# Base Strategy Interface
# ══════════════════════════════════════════════════════════

class MaskingStrategy(ABC):
    """Base class for all semantic masking strategies."""

    @abstractmethod
    def fit(self, hyperedge_embeddings: np.ndarray) -> None:
        pass

    @abstractmethod
    def compute_mask(self, guidance_vec: np.ndarray, top_p: int) -> np.ndarray:
        pass

    @abstractmethod
    def supports_incremental(self) -> bool:
        pass

    @abstractmethod
    def add_hyperedges(self, new_embeddings: np.ndarray) -> None:
        pass


# ══════════════════════════════════════════════════════════
# Strategy A: HDBSCAN (Primary)
# ══════════════════════════════════════════════════════════

class HDBSCANMasking(MaskingStrategy):
    """
    HDBSCAN-based masking.

    Advantages:
    - No K parameter (auto-determines clusters)
    - Noise detection (bridging facts preserved)
    - Hierarchy-aware
    """

    def __init__(
        self,
        min_cluster_size: int = 15,
        min_samples: int = 5,
        cluster_selection_method: str = "eom",
        include_noise_in_mask: bool = True,
    ):
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples
        self.cluster_selection_method = cluster_selection_method
        self.include_noise_in_mask = include_noise_in_mask

        self.clusterer = None
        self.labels = None
        self.centroids: dict[int, np.ndarray] = {}
        self.centroid_matrix: np.ndarray | None = None
        self.centroid_ids: list[int] = []
        self.noise_indices: np.ndarray | None = None
        self.embeddings: np.ndarray | None = None

    def fit(self, hyperedge_embeddings: np.ndarray) -> None:
        import hdbscan

        logger.start_timer("hdbscan_fit")
        self.embeddings = hyperedge_embeddings
        M = len(hyperedge_embeddings)

        self.clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            metric="euclidean",
            cluster_selection_method=self.cluster_selection_method,
        )
        self.labels = self.clusterer.fit_predict(hyperedge_embeddings)

        # Compute centroids
        cluster_ids = set(self.labels) - {-1}
        self.centroids = {}
        for c_id in cluster_ids:
            members = hyperedge_embeddings[self.labels == c_id]
            centroid = members.mean(axis=0)
            norm = np.linalg.norm(centroid)
            self.centroids[c_id] = centroid / (norm + 1e-8)

        self.noise_indices = np.where(self.labels == -1)[0]

        if self.centroids:
            self.centroid_matrix = np.array(list(self.centroids.values()))
            self.centroid_ids = list(self.centroids.keys())
        else:
            self.centroid_matrix = np.zeros((0, hyperedge_embeddings.shape[1]))
            self.centroid_ids = []

        elapsed = logger.stop_timer("hdbscan_fit")

        K_auto = len(cluster_ids)
        noise_rate = len(self.noise_indices) / M * 100 if M > 0 else 0
        logger.step(
            "HDBSCAN Masking",
            f"Found {K_auto} clusters, {len(self.noise_indices)} noise hyperedges "
            f"({noise_rate:.1f}% noise rate)",
            M=M,
            time=elapsed,
        )

        # Cluster size distribution
        if cluster_ids:
            sizes = [int(np.sum(self.labels == c)) for c in sorted(cluster_ids)]
            logger.debug(
                f"  Cluster sizes: min={min(sizes)}, max={max(sizes)}, "
                f"mean={np.mean(sizes):.1f}"
            )

    def compute_mask(self, guidance_vec: np.ndarray, top_p: int) -> np.ndarray:
        M = len(self.labels)

        if len(self.centroid_ids) == 0:
            logger.debug("  HDBSCAN: No clusters found → returning all-ones mask")
            return np.ones(M)

        scores = cosine_similarity(
            guidance_vec.reshape(1, -1), self.centroid_matrix
        )[0]
        top_p_actual = min(top_p, len(self.centroid_ids))
        top_indices = np.argsort(scores)[-top_p_actual:]
        selected_ids = [self.centroid_ids[i] for i in top_indices]

        mask = np.zeros(M)
        for c_id in selected_ids:
            mask[self.labels == c_id] = 1.0

        if self.include_noise_in_mask:
            mask[self.noise_indices] = 1.0

        mask_size = int(np.sum(mask))
        logger.debug(
            f"  HDBSCAN mask: selected {top_p_actual} clusters → "
            f"{mask_size}/{M} hyperedges ({mask_size/M*100:.1f}%)"
        )

        return mask

    def supports_incremental(self) -> bool:
        return False

    def add_hyperedges(self, new_embeddings: np.ndarray) -> None:
        import hdbscan

        new_labels, _ = hdbscan.approximate_predict(
            self.clusterer, new_embeddings
        )
        self.labels = np.concatenate([self.labels, new_labels])
        self.embeddings = np.concatenate([self.embeddings, new_embeddings])
        self.noise_indices = np.where(self.labels == -1)[0]
        logger.debug(
            f"  HDBSCAN: Added {len(new_embeddings)} hyperedges (approximate)"
        )


# ══════════════════════════════════════════════════════════
# Strategy B: K-Means (Baseline)
# ══════════════════════════════════════════════════════════

class KMeansMasking(MaskingStrategy):
    """K-Means clustering baseline for ablation."""

    def __init__(
        self,
        k: int | None = None,
        k_heuristic: str = "sqrt",
        auto_select: bool = False,
        max_k: int = 200,
        n_init: int = 10,
    ):
        self.k = k
        self.k_heuristic = k_heuristic
        self.auto_select = auto_select
        self.max_k = max_k
        self.n_init = n_init

        self.model = None
        self.labels = None
        self.centroids = None

    def fit(self, hyperedge_embeddings: np.ndarray) -> None:
        from sklearn.cluster import KMeans, MiniBatchKMeans
        from sklearn.metrics import silhouette_score

        logger.start_timer("kmeans_fit")
        M = len(hyperedge_embeddings)

        # Determine K
        if self.k is not None:
            K = self.k
        elif self.auto_select:
            K = self._auto_select_k(hyperedge_embeddings)
        else:
            K = max(10, int(np.sqrt(M / 2)))
        K = min(K, self.max_k, M)

        logger.debug(f"  K-Means: K={K} (from {'config' if self.k else 'heuristic'})")

        algo = MiniBatchKMeans if M > 50000 else KMeans
        self.model = algo(n_clusters=K, n_init=self.n_init, random_state=42)
        self.labels = self.model.fit_predict(hyperedge_embeddings)
        self.centroids = self.model.cluster_centers_.copy()

        # Normalize centroids
        norms = np.linalg.norm(self.centroids, axis=1, keepdims=True)
        self.centroids = self.centroids / (norms + 1e-8)

        elapsed = logger.stop_timer("kmeans_fit")
        logger.step(
            "K-Means Masking",
            f"Fitted {K} clusters on {M} hyperedges",
            time=elapsed,
        )

    def _auto_select_k(self, embeddings: np.ndarray) -> int:
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score

        candidates = [10, 20, 30, 50, 75, 100]
        sample_size = min(5000, len(embeddings))
        best_k, best_score = 10, -1.0
        for k in candidates:
            if k >= len(embeddings):
                break
            labels = KMeans(n_clusters=k, n_init=3, random_state=42).fit_predict(
                embeddings
            )
            score = silhouette_score(embeddings, labels, sample_size=sample_size)
            logger.debug(f"  K-Means auto: K={k}, silhouette={score:.4f}")
            if score > best_score:
                best_k, best_score = k, score
        logger.debug(f"  K-Means auto: selected K={best_k} (score={best_score:.4f})")
        return best_k

    def compute_mask(self, guidance_vec: np.ndarray, top_p: int) -> np.ndarray:
        scores = cosine_similarity(
            guidance_vec.reshape(1, -1), self.centroids
        )[0]
        top_p_actual = min(top_p, len(self.centroids))
        top_clusters = np.argsort(scores)[-top_p_actual:]

        mask = np.zeros(len(self.labels))
        for c in top_clusters:
            mask[self.labels == c] = 1.0

        mask_size = int(np.sum(mask))
        logger.debug(
            f"  K-Means mask: {top_p_actual} clusters → "
            f"{mask_size}/{len(self.labels)} hyperedges"
        )
        return mask

    def supports_incremental(self) -> bool:
        return True

    def add_hyperedges(self, new_embeddings: np.ndarray) -> None:
        new_labels = self.model.predict(new_embeddings)
        self.labels = np.concatenate([self.labels, new_labels])


# ══════════════════════════════════════════════════════════
# Strategy C: FAISS Direct ANN (No Clustering)
# ══════════════════════════════════════════════════════════

class FAISSDirectMasking(MaskingStrategy):
    """FAISS-based ANN masking — skip clustering, search directly."""

    def __init__(self, use_gpu: bool = False):
        self.use_gpu = use_gpu
        self.index = None
        self.M = 0

    def fit(self, hyperedge_embeddings: np.ndarray) -> None:
        import faiss

        logger.start_timer("faiss_fit")
        self.M = len(hyperedge_embeddings)
        dim = hyperedge_embeddings.shape[1]
        embeddings = hyperedge_embeddings.astype(np.float32).copy()

        faiss.normalize_L2(embeddings)

        if self.M < 10_000:
            self.index = faiss.IndexFlatIP(dim)
        elif self.M < 100_000:
            nlist = max(16, int(np.sqrt(self.M)))
            quantizer = faiss.IndexFlatIP(dim)
            self.index = faiss.IndexIVFFlat(quantizer, dim, nlist)
            self.index.train(embeddings)
            self.index.nprobe = min(16, nlist)
        else:
            nlist = max(64, int(np.sqrt(self.M)))
            quantizer = faiss.IndexFlatIP(dim)
            m = max(1, dim // 8)
            self.index = faiss.IndexIVFPQ(quantizer, dim, nlist, m, 8)
            self.index.train(embeddings)
            self.index.nprobe = min(32, nlist)

        self.index.add(embeddings)

        if self.use_gpu:
            try:
                if faiss.get_num_gpus() > 0:
                    self.index = faiss.index_cpu_to_all_gpus(self.index)
                    logger.debug("  FAISS: Using GPU")
            except Exception:
                logger.debug("  FAISS: GPU not available, using CPU")

        elapsed = logger.stop_timer("faiss_fit")
        logger.step(
            "FAISS Masking",
            f"Built index for {self.M} hyperedges (dim={dim})",
            time=elapsed,
        )

    def compute_mask(self, guidance_vec: np.ndarray, top_p: int) -> np.ndarray:
        import faiss

        query = guidance_vec.reshape(1, -1).astype(np.float32).copy()
        faiss.normalize_L2(query)

        top_r = top_p * max(1, self.M // 50)
        top_r = min(top_r, self.M)

        scores, indices = self.index.search(query, top_r)

        mask = np.zeros(self.M)
        valid = indices[0] >= 0
        mask[indices[0][valid]] = 1.0

        logger.debug(f"  FAISS mask: top_r={top_r} → {int(np.sum(mask))}/{self.M}")
        return mask

    def supports_incremental(self) -> bool:
        return True

    def add_hyperedges(self, new_embeddings: np.ndarray) -> None:
        import faiss

        embeddings = new_embeddings.astype(np.float32).copy()
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)
        self.M += len(new_embeddings)


# ══════════════════════════════════════════════════════════
# Strategy D: No Masking (Ablation)
# ══════════════════════════════════════════════════════════

class NoMasking(MaskingStrategy):
    """Ablation baseline: no masking, all hyperedges included."""

    def __init__(self):
        self.M = 0

    def fit(self, hyperedge_embeddings: np.ndarray) -> None:
        self.M = len(hyperedge_embeddings)
        logger.step("NoMasking", f"No masking applied (M={self.M})")

    def compute_mask(self, guidance_vec: np.ndarray, top_p: int) -> np.ndarray:
        return np.ones(self.M)

    def supports_incremental(self) -> bool:
        return True

    def add_hyperedges(self, new_embeddings: np.ndarray) -> None:
        self.M += len(new_embeddings)


# ══════════════════════════════════════════════════════════
# Factory
# ══════════════════════════════════════════════════════════

def create_masking_strategy(config: dict) -> MaskingStrategy:
    """Factory: create masking strategy from config."""
    strategy_name = config["masking"]["strategy"]

    if strategy_name == "hdbscan":
        params = config["masking"].get("hdbscan", {})
        return HDBSCANMasking(**params)
    elif strategy_name == "kmeans":
        params = config["masking"].get("kmeans", {})
        return KMeansMasking(**params)
    elif strategy_name == "faiss_direct":
        params = config["masking"].get("faiss_direct", {})
        return FAISSDirectMasking(**params)
    elif strategy_name == "none":
        return NoMasking()
    else:
        raise ValueError(f"Unknown masking strategy: {strategy_name}")
