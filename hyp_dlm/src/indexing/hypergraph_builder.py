"""
Hypergraph Builder — constructs incidence matrix H from chunks and entities.

H[i, j] = 1 if entity v_i appears in chunk/hyperedge e_j.
"""

import numpy as np
from scipy import sparse
from typing import Any

from src.utils.logger import get_logger

logger = get_logger(__name__)


class HypergraphBuilder:
    """Build the hypergraph incidence matrix H."""

    def build(
        self,
        chunks: list,
        global_entities: list,
    ) -> tuple[sparse.csr_matrix, dict[str, int], dict[int, int]]:
        """
        Build incidence matrix H.

        Args:
            chunks: list of Chunk objects (each has .entities)
            global_entities: deduplicated global entity list

        Returns:
            H: incidence matrix (N x M) sparse
            entity_index: {entity_name: global_idx}
            hyperedge_index: {chunk_idx: column_in_H}
        """
        logger.start_timer("build_H")

        # Build entity-to-index mapping
        entity_index = {
            ent.name: idx for idx, ent in enumerate(global_entities)
        }
        N = len(global_entities)
        M = len(chunks)

        logger.debug(f"Building H: {N} entities x {M} hyperedges")

        # Build incidence matrix
        rows = []
        cols = []
        for j, chunk in enumerate(chunks):
            for ent in chunk.entities:
                key = ent.normalized_name if hasattr(ent, 'normalized_name') else ent.name
                if key in entity_index:
                    rows.append(entity_index[key])
                    cols.append(j)

        data = np.ones(len(rows), dtype=np.float64)
        H = sparse.csr_matrix((data, (rows, cols)), shape=(N, M))

        logger.debug(f"  Raw H: {H.shape}, nnz={H.nnz}")

        # Validation: find and log statistics
        row_sums = np.array(H.sum(axis=1)).flatten()
        col_sums = np.array(H.sum(axis=0)).flatten()

        isolated_entities = int(np.sum(row_sums == 0))
        empty_hyperedges = int(np.sum(col_sums == 0))

        # Remove isolated entities and empty hyperedges
        active_rows = row_sums > 0
        active_cols = col_sums > 0

        if isolated_entities > 0 or empty_hyperedges > 0:
            logger.debug(
                f"  Removing {isolated_entities} isolated entities, "
                f"{empty_hyperedges} empty hyperedges"
            )
            # We keep the full matrix but log what's isolated
            # (removing rows/cols would break index mappings)

        # Compute statistics
        active_row_sums = row_sums[active_rows]
        active_col_sums = col_sums[active_cols]

        hyperedge_index = {j: j for j in range(M)}

        elapsed = logger.stop_timer("build_H")

        logger.step(
            "HypergraphBuilder",
            f"Incidence matrix H built: {H.shape}",
            nnz=H.nnz,
            isolated_entities=isolated_entities,
            empty_hyperedges=empty_hyperedges,
            time=elapsed,
        )

        # Log detailed statistics
        if len(active_row_sums) > 0:
            logger.debug(
                f"  Entities per hyperedge: "
                f"min={active_col_sums.min():.0f}, "
                f"max={active_col_sums.max():.0f}, "
                f"mean={active_col_sums.mean():.1f}"
            )
            logger.debug(
                f"  Hyperedges per entity: "
                f"min={active_row_sums.min():.0f}, "
                f"max={active_row_sums.max():.0f}, "
                f"mean={active_row_sums.mean():.1f}"
            )

        logger.matrix("Incidence Matrix H", H)

        # Summary table
        logger.summary_table(
            "Hypergraph Statistics",
            [
                {"Metric": "|V| (entities)", "Value": N},
                {"Metric": "|E_H| (hyperedges)", "Value": M},
                {"Metric": "Edges (nnz in H)", "Value": H.nnz},
                {"Metric": "Isolated entities", "Value": isolated_entities},
                {"Metric": "Empty hyperedges", "Value": empty_hyperedges},
                {
                    "Metric": "Avg entities/hyperedge",
                    "Value": f"{active_col_sums.mean():.2f}" if len(active_col_sums) > 0 else "N/A",
                },
            ],
        )

        return H, entity_index, hyperedge_index

    def expand(
        self,
        old_H: sparse.csr_matrix,
        old_entity_index: dict[str, int],
        old_M: int,
        new_chunks: list,
        old_entities: list,
        combined_entities: list,
    ) -> tuple[sparse.csr_matrix, dict[str, int], dict[int, int]]:
        """
        Incrementally expand incidence matrix H with new chunks.

        Args:
            old_H: existing incidence matrix (N_old x M_old)
            old_entity_index: {entity_name: global_idx} for old entities
            old_M: number of old hyperedges (columns in old_H)
            new_chunks: list of new Chunk objects (each has .entities)
            old_entities: list of old Entity objects
            combined_entities: deduplicated list of all entities (old + new)

        Returns:
            H: expanded incidence matrix (N_total x M_total)
            entity_index: updated {entity_name: global_idx}
            hyperedge_index: updated {chunk_idx: column_in_H}
        """
        logger.start_timer("expand_H")

        # Build combined entity index
        entity_index = {ent.name: idx for idx, ent in enumerate(combined_entities)}
        N_total = len(combined_entities)
        N_old = old_H.shape[0]
        M_new = len(new_chunks)
        M_total = old_M + M_new

        logger.debug(
            f"Expanding H: entities {N_old} -> {N_total}, "
            f"hyperedges {old_M} -> {M_total}"
        )

        # Pad old H with zero rows for new entities
        if N_total > N_old:
            padding = sparse.csr_matrix((N_total - N_old, old_M), dtype=np.float64)
            padded_old_H = sparse.vstack([old_H, padding], format="csr")
        else:
            padded_old_H = old_H

        # Build new columns from new chunks' entity memberships
        rows = []
        cols = []
        for j, chunk in enumerate(new_chunks):
            for ent in chunk.entities:
                key = ent.normalized_name if hasattr(ent, 'normalized_name') else ent.name
                if key in entity_index:
                    rows.append(entity_index[key])
                    cols.append(j)

        if rows:
            data = np.ones(len(rows), dtype=np.float64)
            new_cols = sparse.csr_matrix(
                (data, (rows, cols)), shape=(N_total, M_new)
            )
        else:
            new_cols = sparse.csr_matrix((N_total, M_new), dtype=np.float64)

        # Combine old + new
        H = sparse.hstack([padded_old_H, new_cols], format="csr")

        # Build hyperedge index
        hyperedge_index = {j: j for j in range(M_total)}

        elapsed = logger.stop_timer("expand_H")
        logger.step(
            "HypergraphBuilder",
            f"Incidence matrix H expanded: {H.shape}",
            nnz=H.nnz,
            new_hyperedges=M_new,
            time=elapsed,
        )
        logger.matrix("Expanded Incidence Matrix H", H)

        return H, entity_index, hyperedge_index
