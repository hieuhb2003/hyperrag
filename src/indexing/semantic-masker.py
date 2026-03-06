"""Semantic masking: HDBSCAN clustering + multi-label assignment.
Coarse filter for propagation retrieval."""

from __future__ import annotations

import importlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.config import Config

_logging = importlib.import_module("src.logging-setup")
logger = _logging.get_logger("masking")


@dataclass
class SemanticMask:
    cluster_centroids: np.ndarray  # [K x 768] L2-normalized
    chunk_cluster_map: dict[str, list[int]]  # chunk_id -> cluster indices
    cluster_chunk_map: dict[int, list[str]]  # cluster_idx -> chunk_ids
    n_clusters: int
    chunk_labels: dict[str, int]  # chunk_id -> primary HDBSCAN label (-1 = noise)


class SemanticMasker:
    """HDBSCAN clustering + multi-label assignment for retrieval masking."""

    def __init__(self, config: Config):
        self.config = config

    def build(self, chunks: list) -> SemanticMask:
        """Cluster chunks and build multi-label assignments."""
        embeddings = np.array([c.embedding for c in chunks])  # [M x 768]

        # Run HDBSCAN
        import hdbscan
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.config.hdbscan_min_cluster_size,
            metric="euclidean",
            cluster_selection_method="eom",
        )
        labels = clusterer.fit_predict(embeddings)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        noise_count = int((labels == -1).sum())
        logger.info(f"HDBSCAN: {n_clusters} clusters, {noise_count} noise points")

        # Handle edge case: no clusters found
        if n_clusters == 0:
            logger.warning("HDBSCAN found 0 clusters. Assigning all to single cluster.")
            centroid = embeddings.mean(axis=0)
            centroid = centroid / (np.linalg.norm(centroid) + 1e-8)
            centroids = np.array([centroid])
            n_clusters = 1
            chunk_cluster_map = {c.id: [0] for c in chunks}
            cluster_chunk_map = {0: [c.id for c in chunks]}
            chunk_labels = {c.id: 0 for c in chunks}
            return SemanticMask(
                cluster_centroids=centroids,
                chunk_cluster_map=chunk_cluster_map,
                cluster_chunk_map=cluster_chunk_map,
                n_clusters=1,
                chunk_labels=chunk_labels,
            )

        # Compute L2-normalized cluster centroids
        centroids = []
        for k in range(n_clusters):
            mask = labels == k
            centroid = embeddings[mask].mean(axis=0)
            norm = np.linalg.norm(centroid)
            centroids.append(centroid / (norm + 1e-8))
        centroids = np.array(centroids)  # [K x 768]

        # Multi-label assignment
        chunk_cluster_map: dict[str, list[int]] = {}
        cluster_chunk_map: dict[int, list[str]] = {k: [] for k in range(n_clusters)}
        chunk_labels_map: dict[str, int] = {}
        tau = self.config.multi_label_threshold

        for i, chunk in enumerate(chunks):
            emb = embeddings[i]
            emb_norm = emb / (np.linalg.norm(emb) + 1e-8)
            sims = centroids @ emb_norm  # [K]

            assigned = [int(k) for k in range(n_clusters) if sims[k] >= tau]
            # Fallback: assign to nearest centroid
            if not assigned:
                assigned = [int(np.argmax(sims))]

            chunk_cluster_map[chunk.id] = assigned
            chunk_labels_map[chunk.id] = int(labels[i])
            for k in assigned:
                cluster_chunk_map[k].append(chunk.id)

        # Stats
        multi_count = sum(1 for v in chunk_cluster_map.values() if len(v) > 1)
        logger.info(f"Multi-label: {multi_count}/{len(chunks)} chunks in >1 cluster")
        if cluster_chunk_map:
            sizes = [len(v) for v in cluster_chunk_map.values()]
            logger.info(f"Cluster sizes: min={min(sizes)}, max={max(sizes)}, "
                       f"avg={np.mean(sizes):.0f}")

        return SemanticMask(
            cluster_centroids=centroids,
            chunk_cluster_map=chunk_cluster_map,
            cluster_chunk_map=cluster_chunk_map,
            n_clusters=n_clusters,
            chunk_labels=chunk_labels_map,
        )

    def save(self, mask: SemanticMask, storage_path: str | Path):
        """Save mask data to disk."""
        path = Path(storage_path)
        path.mkdir(parents=True, exist_ok=True)
        np.save(str(path / "cluster_centroids.npy"), mask.cluster_centroids)
        with open(path / "chunk_cluster_map.json", "w") as f:
            json.dump(mask.chunk_cluster_map, f)
        with open(path / "cluster_chunk_map.json", "w") as f:
            json.dump({str(k): v for k, v in mask.cluster_chunk_map.items()}, f)
        with open(path / "chunk_labels.json", "w") as f:
            json.dump(mask.chunk_labels, f)
        logger.info(f"Saved semantic mask to {path}")

    def load(self, storage_path: str | Path) -> SemanticMask:
        """Load mask data from disk."""
        path = Path(storage_path)
        centroids = np.load(str(path / "cluster_centroids.npy"))
        with open(path / "chunk_cluster_map.json") as f:
            chunk_cluster_map = json.load(f)
        with open(path / "cluster_chunk_map.json") as f:
            raw = json.load(f)
            cluster_chunk_map = {int(k): v for k, v in raw.items()}

        # Load chunk_labels if available
        chunk_labels = {}
        labels_path = path / "chunk_labels.json"
        if labels_path.exists():
            with open(labels_path) as f:
                chunk_labels = {k: int(v) for k, v in json.load(f).items()}

        return SemanticMask(
            cluster_centroids=centroids,
            chunk_cluster_map=chunk_cluster_map,
            cluster_chunk_map=cluster_chunk_map,
            n_clusters=len(centroids),
            chunk_labels=chunk_labels,
        )
