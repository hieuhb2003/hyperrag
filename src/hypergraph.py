"""
Hypergraph Matrix Builder for HyP-DLM.

This module constructs the Incidence Matrix H for the bipartite hypergraph
where entities are nodes and propositions are hyperedges.
"""

import logging
from typing import Dict, List, Tuple, Set, Optional
from dataclasses import dataclass
import numpy as np
from scipy import sparse
from collections import defaultdict

from src.propositions import Proposition
from src.ner import SpacyNER

logger = logging.getLogger(__name__)


@dataclass
class HypergraphData:
    """Container for hypergraph data structures."""
    
    # Incidence matrix H: (num_entities x num_propositions)
    incidence_matrix: sparse.csr_matrix
    
    # Entity mappings
    entity_to_idx: Dict[str, int]
    idx_to_entity: Dict[int, str]
    
    # Proposition mappings
    prop_to_idx: Dict[str, int]
    idx_to_prop: Dict[int, str]
    
    # Entity embeddings (computed from entity names)
    entity_embeddings: Optional[np.ndarray] = None
    
    # Proposition embeddings
    proposition_embeddings: Optional[np.ndarray] = None
    
    @property
    def num_entities(self) -> int:
        return self.incidence_matrix.shape[0]
    
    @property
    def num_propositions(self) -> int:
        return self.incidence_matrix.shape[1]
    
    def get_entities_in_proposition(self, prop_id: str) -> List[str]:
        """Get all entities that appear in a proposition."""
        if prop_id not in self.prop_to_idx:
            return []
        
        prop_idx = self.prop_to_idx[prop_id]
        entity_indices = self.incidence_matrix[:, prop_idx].nonzero()[0]
        return [self.idx_to_entity[idx] for idx in entity_indices]
    
    def get_propositions_with_entity(self, entity: str) -> List[str]:
        """Get all propositions that contain an entity."""
        entity_lower = entity.lower()
        if entity_lower not in self.entity_to_idx:
            return []
        
        entity_idx = self.entity_to_idx[entity_lower]
        prop_indices = self.incidence_matrix[entity_idx, :].nonzero()[1]
        return [self.idx_to_prop[idx] for idx in prop_indices]


class HypergraphBuilder:
    """Builds the hypergraph incidence matrix from propositions and entities."""
    
    def __init__(
        self,
        ner_model: SpacyNER,
        embedding_model,
        batch_size: int = 128
    ):
        self.ner_model = ner_model
        self.embedding_model = embedding_model
        self.batch_size = batch_size
    
    def build(
        self,
        propositions: Dict[str, Proposition]
    ) -> HypergraphData:
        """
        Build hypergraph from propositions.
        
        Args:
            propositions: Dictionary of proposition_id -> Proposition
            
        Returns:
            HypergraphData containing incidence matrix and mappings
        """
        logger.info(f"Building hypergraph from {len(propositions)} propositions...")
        
        # Step 1: Extract entities from all propositions
        prop_to_entities = self._extract_entities(propositions)
        
        # Step 2: Build entity and proposition indices
        entity_to_idx, idx_to_entity = self._build_entity_index(prop_to_entities)
        prop_to_idx = {prop_id: idx for idx, prop_id in enumerate(propositions.keys())}
        idx_to_prop = {idx: prop_id for prop_id, idx in prop_to_idx.items()}
        
        logger.info(f"Found {len(entity_to_idx)} unique entities")
        
        # Step 3: Build incidence matrix
        incidence_matrix = self._build_incidence_matrix(
            prop_to_entities,
            entity_to_idx,
            prop_to_idx
        )
        
        # Step 4: Collect proposition embeddings
        proposition_embeddings = self._collect_proposition_embeddings(propositions, prop_to_idx)
        
        # Step 5: Compute entity embeddings
        entity_embeddings = self._compute_entity_embeddings(idx_to_entity)
        
        # Update propositions with their entities
        for prop_id, entities in prop_to_entities.items():
            if prop_id in propositions:
                propositions[prop_id].entities = list(entities)
        
        hypergraph = HypergraphData(
            incidence_matrix=incidence_matrix,
            entity_to_idx=entity_to_idx,
            idx_to_entity=idx_to_entity,
            prop_to_idx=prop_to_idx,
            idx_to_prop=idx_to_prop,
            entity_embeddings=entity_embeddings,
            proposition_embeddings=proposition_embeddings
        )
        
        logger.info(
            f"Hypergraph built: {hypergraph.num_entities} entities, "
            f"{hypergraph.num_propositions} propositions"
        )
        
        return hypergraph
    
    def _extract_entities(
        self,
        propositions: Dict[str, Proposition]
    ) -> Dict[str, Set[str]]:
        """Extract entities from each proposition using NER."""
        prop_to_entities: Dict[str, Set[str]] = {}
        
        # Prepare for batch NER
        prop_id_to_text = {
            prop_id: prop.text for prop_id, prop in propositions.items()
        }
        
        # Run batch NER
        _, sentence_to_entities = self.ner_model.batch_ner(
            prop_id_to_text, 
            max_workers=1  # Use single worker for simplicity
        )
        
        # Map back to propositions
        for prop_id, prop in propositions.items():
            entities = set()
            # Check if proposition text matches any sentence
            if prop.text in sentence_to_entities:
                entities = set(e.lower() for e in sentence_to_entities[prop.text])
            prop_to_entities[prop_id] = entities
        
        return prop_to_entities
    
    def _build_entity_index(
        self,
        prop_to_entities: Dict[str, Set[str]]
    ) -> Tuple[Dict[str, int], Dict[int, str]]:
        """Build entity to index mapping."""
        all_entities = set()
        for entities in prop_to_entities.values():
            all_entities.update(entities)
        
        # Sort for deterministic ordering
        sorted_entities = sorted(all_entities)
        entity_to_idx = {entity: idx for idx, entity in enumerate(sorted_entities)}
        idx_to_entity = {idx: entity for entity, idx in entity_to_idx.items()}
        
        return entity_to_idx, idx_to_entity
    
    def _build_incidence_matrix(
        self,
        prop_to_entities: Dict[str, Set[str]],
        entity_to_idx: Dict[str, int],
        prop_to_idx: Dict[str, int]
    ) -> sparse.csr_matrix:
        """Build sparse incidence matrix H where H[i,j] = 1 if entity i is in proposition j."""
        num_entities = len(entity_to_idx)
        num_props = len(prop_to_idx)
        
        rows = []
        cols = []
        data = []
        
        for prop_id, entities in prop_to_entities.items():
            if prop_id not in prop_to_idx:
                continue
            
            prop_idx = prop_to_idx[prop_id]
            for entity in entities:
                if entity in entity_to_idx:
                    entity_idx = entity_to_idx[entity]
                    rows.append(entity_idx)
                    cols.append(prop_idx)
                    data.append(1.0)
        
        incidence_matrix = sparse.csr_matrix(
            (data, (rows, cols)),
            shape=(num_entities, num_props),
            dtype=np.float32
        )
        
        return incidence_matrix
    
    def _collect_proposition_embeddings(
        self,
        propositions: Dict[str, Proposition],
        prop_to_idx: Dict[str, int]
    ) -> np.ndarray:
        """Collect proposition embeddings into a matrix."""
        num_props = len(prop_to_idx)
        
        # Get embedding dimension from first proposition
        sample_prop = next(iter(propositions.values()))
        if sample_prop.embedding is None:
            logger.warning("Propositions have no embeddings, skipping")
            return None
        
        embed_dim = sample_prop.embedding.shape[0]
        embeddings = np.zeros((num_props, embed_dim), dtype=np.float32)
        
        for prop_id, prop in propositions.items():
            if prop_id in prop_to_idx and prop.embedding is not None:
                embeddings[prop_to_idx[prop_id]] = prop.embedding
        
        return embeddings
    
    def _compute_entity_embeddings(
        self,
        idx_to_entity: Dict[int, str]
    ) -> np.ndarray:
        """Compute embeddings for entity names."""
        logger.info(f"Computing embeddings for {len(idx_to_entity)} entities...")
        
        # Sort by index to maintain order
        entities = [idx_to_entity[i] for i in range(len(idx_to_entity))]
        
        embeddings = self.embedding_model.encode(
            entities,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        return embeddings


def save_hypergraph(hypergraph: HypergraphData, path: str) -> None:
    """Save hypergraph data to disk."""
    import json
    import os
    
    os.makedirs(path, exist_ok=True)
    
    # Save sparse matrix
    sparse.save_npz(os.path.join(path, "incidence_matrix.npz"), hypergraph.incidence_matrix)
    
    # Save embeddings
    if hypergraph.entity_embeddings is not None:
        np.save(os.path.join(path, "entity_embeddings.npy"), hypergraph.entity_embeddings)
    
    if hypergraph.proposition_embeddings is not None:
        np.save(os.path.join(path, "proposition_embeddings.npy"), hypergraph.proposition_embeddings)
    
    # Save mappings
    mappings = {
        "entity_to_idx": hypergraph.entity_to_idx,
        "idx_to_entity": {str(k): v for k, v in hypergraph.idx_to_entity.items()},
        "prop_to_idx": hypergraph.prop_to_idx,
        "idx_to_prop": {str(k): v for k, v in hypergraph.idx_to_prop.items()}
    }
    
    with open(os.path.join(path, "mappings.json"), "w", encoding="utf-8") as f:
        json.dump(mappings, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Hypergraph saved to {path}")


def load_hypergraph(path: str) -> HypergraphData:
    """Load hypergraph data from disk."""
    import json
    import os
    
    # Load sparse matrix
    incidence_matrix = sparse.load_npz(os.path.join(path, "incidence_matrix.npz"))
    
    # Load embeddings
    entity_embeddings = None
    prop_embeddings = None
    
    if os.path.exists(os.path.join(path, "entity_embeddings.npy")):
        entity_embeddings = np.load(os.path.join(path, "entity_embeddings.npy"))
    
    if os.path.exists(os.path.join(path, "proposition_embeddings.npy")):
        prop_embeddings = np.load(os.path.join(path, "proposition_embeddings.npy"))
    
    # Load mappings
    with open(os.path.join(path, "mappings.json"), "r", encoding="utf-8") as f:
        mappings = json.load(f)
    
    return HypergraphData(
        incidence_matrix=incidence_matrix,
        entity_to_idx=mappings["entity_to_idx"],
        idx_to_entity={int(k): v for k, v in mappings["idx_to_entity"].items()},
        prop_to_idx=mappings["prop_to_idx"],
        idx_to_prop={int(k): v for k, v in mappings["idx_to_prop"].items()},
        entity_embeddings=entity_embeddings,
        proposition_embeddings=prop_embeddings
    )
