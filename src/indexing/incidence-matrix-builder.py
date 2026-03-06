"""Build and normalize the hypergraph incidence matrix H[N x M].
H_norm = Dv^(-1/2) * H * De^(-1/2) for degree-normalized propagation."""

from __future__ import annotations

import importlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.sparse as sp

from src.config import Config

_logging = importlib.import_module("src.logging-setup")
logger = _logging.get_logger("incidence-matrix")


@dataclass
class HypergraphData:
    H_norm: sp.csr_matrix  # normalized incidence matrix [N x M]
    entity_to_idx: dict[str, int]
    chunk_to_idx: dict[str, int]
    idx_to_entity: dict[int, str]
    idx_to_chunk: dict[int, str]


class IncidenceMatrixBuilder:
    """Build sparse incidence matrix from entity-chunk mappings."""

    def __init__(self, config: Config):
        self.config = config

    def build(self, entities: list, chunks: list) -> HypergraphData:
        """Build and normalize incidence matrix."""
        # Index mappings
        entity_to_idx = {e.id: i for i, e in enumerate(entities)}
        chunk_to_idx = {c.id: i for i, c in enumerate(chunks)}
        N, M = len(entities), len(chunks)

        # Build sparse binary H
        rows, cols = [], []
        for entity in entities:
            eidx = entity_to_idx[entity.id]
            for chunk_id in entity.chunk_ids:
                if chunk_id in chunk_to_idx:
                    rows.append(eidx)
                    cols.append(chunk_to_idx[chunk_id])

        data = np.ones(len(rows), dtype=np.float32)
        H = sp.csr_matrix((data, (rows, cols)), shape=(N, M))

        # Degree matrices
        Dv_diag = np.array(H.sum(axis=1)).flatten()  # entity degrees
        De_diag = np.array(H.sum(axis=0)).flatten()  # chunk degrees

        # Inverse sqrt (handle zeros)
        Dv_inv_sqrt = np.where(Dv_diag > 0, 1.0 / np.sqrt(Dv_diag), 0.0)
        De_inv_sqrt = np.where(De_diag > 0, 1.0 / np.sqrt(De_diag), 0.0)

        # H_norm = Dv^(-1/2) * H * De^(-1/2)
        H_norm = sp.diags(Dv_inv_sqrt) @ H @ sp.diags(De_inv_sqrt)

        # Log stats
        density = H.nnz / (N * M) if N * M > 0 else 0
        logger.info(f"Incidence matrix: {N} entities x {M} chunks, "
                    f"density={density:.4f}, nnz={H.nnz}")
        if Dv_diag.size > 0:
            logger.info(f"Entity degree: avg={Dv_diag.mean():.1f}, "
                       f"max={Dv_diag.max():.0f}, min={Dv_diag.min():.0f}")
        if De_diag.size > 0:
            logger.info(f"Chunk degree: avg={De_diag.mean():.1f}, "
                       f"max={De_diag.max():.0f}, min={De_diag.min():.0f}")

        return HypergraphData(
            H_norm=H_norm,
            entity_to_idx=entity_to_idx,
            chunk_to_idx=chunk_to_idx,
            idx_to_entity={v: k for k, v in entity_to_idx.items()},
            idx_to_chunk={v: k for k, v in chunk_to_idx.items()},
        )

    def save(self, data: HypergraphData, storage_path: str | Path):
        """Save incidence matrix and mappings to disk."""
        path = Path(storage_path)
        path.mkdir(parents=True, exist_ok=True)

        sp.save_npz(str(path / "h_norm.npz"), data.H_norm)
        with open(path / "entity_map.json", "w") as f:
            json.dump(data.entity_to_idx, f)
        with open(path / "chunk_map.json", "w") as f:
            json.dump(data.chunk_to_idx, f)

        logger.info(f"Saved incidence matrix to {path}")

    def load(self, storage_path: str | Path) -> HypergraphData:
        """Load incidence matrix and mappings from disk."""
        path = Path(storage_path)

        H_norm = sp.load_npz(str(path / "h_norm.npz"))
        with open(path / "entity_map.json") as f:
            entity_to_idx = json.load(f)
        with open(path / "chunk_map.json") as f:
            chunk_to_idx = json.load(f)

        return HypergraphData(
            H_norm=H_norm,
            entity_to_idx=entity_to_idx,
            chunk_to_idx=chunk_to_idx,
            idx_to_entity={int(v): k for k, v in entity_to_idx.items()},
            idx_to_chunk={int(v): k for k, v in chunk_to_idx.items()},
        )
