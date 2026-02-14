"""
Bipartite Graph Storage — store hypergraph as bipartite for incremental updates.

Following HyperGraphRAG's Proposition 2: bipartite graph losslessly preserves hypergraph.
Uses NetworkX for simplicity (suitable for graphs < 100k nodes).
"""

import os
import pickle
from pathlib import Path
from typing import Any, Optional

import numpy as np
from scipy import sparse

from src.utils.logger import get_logger

logger = get_logger(__name__)


class BipartiteStorage:
    """
    Store hypergraph as a bipartite graph.

    Node types:
      - "entity:<name>" — entity nodes
      - "hyperedge:<id>" — hyperedge nodes
    """

    def __init__(self):
        import networkx as nx

        self.graph = nx.Graph()
        self._entity_count = 0
        self._hyperedge_count = 0

    def build_from_incidence(
        self,
        H: sparse.csr_matrix,
        entities: list,
        chunks: list,
    ) -> None:
        """
        Build bipartite graph from incidence matrix H.

        Args:
            H: Incidence matrix (N x M)
            entities: global entity list
            chunks: list of Chunk objects
        """
        import networkx as nx

        logger.start_timer("bipartite_build")

        self.graph = nx.Graph()

        # Add entity nodes
        for idx, ent in enumerate(entities):
            self.graph.add_node(
                f"entity:{ent.name}",
                node_type="entity",
                entity_type=ent.entity_type,
                global_idx=idx,
            )

        # Add hyperedge nodes
        for idx, chunk in enumerate(chunks):
            self.graph.add_node(
                f"hyperedge:{idx}",
                node_type="hyperedge",
                text=chunk.text[:200],  # Store truncated text
                doc_id=chunk.doc_id,
                chunk_id=chunk.chunk_id,
            )

        # Add edges from incidence matrix
        H_coo = H.tocoo()
        for row, col in zip(H_coo.row, H_coo.col):
            ent_name = entities[row].name
            self.graph.add_edge(f"entity:{ent_name}", f"hyperedge:{col}")

        self._entity_count = len(entities)
        self._hyperedge_count = len(chunks)

        elapsed = logger.stop_timer("bipartite_build")

        logger.step(
            "BipartiteStorage",
            f"Built bipartite graph: {self.graph.number_of_nodes()} nodes, "
            f"{self.graph.number_of_edges()} edges",
            entities=self._entity_count,
            hyperedges=self._hyperedge_count,
            time=elapsed,
        )

    def get_entity_neighbors(self, entity_name: str) -> list[int]:
        """Get hyperedge IDs connected to an entity."""
        node = f"entity:{entity_name}"
        if node not in self.graph:
            return []
        neighbors = []
        for n in self.graph.neighbors(node):
            if n.startswith("hyperedge:"):
                neighbors.append(int(n.split(":")[1]))
        return neighbors

    def get_hyperedge_entities(self, hyperedge_id: int) -> list[str]:
        """Get entity names connected to a hyperedge."""
        node = f"hyperedge:{hyperedge_id}"
        if node not in self.graph:
            return []
        entities = []
        for n in self.graph.neighbors(node):
            if n.startswith("entity:"):
                entities.append(n.split(":", 1)[1])
        return entities

    def add_document(
        self,
        new_chunks: list,
        new_entities: list,
        H_new_cols: sparse.csr_matrix,
        existing_entities: list,
    ) -> None:
        """Incremental: add new document's chunks and entities."""
        logger.debug(
            f"Incremental add: {len(new_chunks)} chunks, {len(new_entities)} new entities"
        )

        # Add new entity nodes
        for ent in new_entities:
            node_id = f"entity:{ent.name}"
            if node_id not in self.graph:
                self.graph.add_node(
                    node_id,
                    node_type="entity",
                    entity_type=ent.entity_type,
                    global_idx=self._entity_count,
                )
                self._entity_count += 1

        # Add new hyperedge nodes and edges
        base_he_id = self._hyperedge_count
        H_coo = H_new_cols.tocoo()
        for idx, chunk in enumerate(new_chunks):
            he_id = base_he_id + idx
            self.graph.add_node(
                f"hyperedge:{he_id}",
                node_type="hyperedge",
                text=chunk.text[:200],
                doc_id=chunk.doc_id,
                chunk_id=chunk.chunk_id,
            )

        for row, col in zip(H_coo.row, H_coo.col):
            ent_name = existing_entities[row].name
            he_id = base_he_id + col
            self.graph.add_edge(f"entity:{ent_name}", f"hyperedge:{he_id}")

        self._hyperedge_count += len(new_chunks)

        logger.step(
            "BipartiteStorage",
            f"Incremental update: now {self.graph.number_of_nodes()} nodes, "
            f"{self.graph.number_of_edges()} edges",
        )

    def save(self, output_dir: str) -> None:
        """Persist bipartite graph to disk."""
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        with open(path / "bipartite_graph.pkl", "wb") as f:
            pickle.dump(self.graph, f)
        logger.debug(f"Bipartite graph saved to {output_dir}")

    def load(self, input_dir: str) -> None:
        """Load bipartite graph from disk."""
        path = Path(input_dir) / "bipartite_graph.pkl"
        with open(path, "rb") as f:
            self.graph = pickle.load(f)
        # Reconstruct counts
        self._entity_count = sum(
            1 for n in self.graph.nodes if n.startswith("entity:")
        )
        self._hyperedge_count = sum(
            1 for n in self.graph.nodes if n.startswith("hyperedge:")
        )
        logger.debug(
            f"Loaded bipartite graph: {self.graph.number_of_nodes()} nodes, "
            f"{self.graph.number_of_edges()} edges"
        )
