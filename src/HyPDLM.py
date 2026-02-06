"""
HyP-DLM: Hypergraph Propagation with Dynamic Logic Modulation.

Main class that orchestrates all components for indexing and retrieval.
"""

import logging
import os
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import numpy as np

from src.propositions import PropositionExtractor, Proposition
from src.hypergraph import HypergraphBuilder, HypergraphData, save_hypergraph, load_hypergraph
from src.semantic_affinity import SemanticAffinityBuilder, save_affinity_matrix, load_affinity_matrix
from src.semantic_masking import SemanticMasker
from src.dynamic_modulation import DynamicModulator, LogicDecomposer, PropagationResult
from src.ner import SpacyNER
from src.utils import LLM_Model, compute_mdhash_id

logger = logging.getLogger(__name__)


@dataclass
class HyPDLMConfig:
    """Configuration for HyP-DLM."""
    
    # Model configs
    embedding_model: Any
    llm_model: LLM_Model
    spacy_model: str = "en_core_web_trf"
    
    # Working directory
    working_dir: str = "./hypdlm_index"
    
    # Clustering configs
    num_clusters: int = 100
    top_p_clusters: int = 10
    
    # Affinity configs
    lexical_weight: float = 0.5
    affinity_threshold: float = 0.7
    propagation_factor: float = 0.3
    
    # Retrieval configs
    max_hops: int = 5
    top_k_entities: int = 20
    top_k_propositions: int = 10
    
    # Processing configs
    max_workers: int = 16
    batch_size: int = 32


@dataclass
class RetrievalResult:
    """Result from HyP-DLM retrieval."""
    
    query: str
    answer: Optional[str]
    top_propositions: List[str]
    top_entities: List[str]
    source_chunks: List[str]
    num_hops: int
    citations: List[Dict[str, str]]


class HyPDLM:
    """
    Main HyP-DLM class for indexing and retrieval.
    
    Implements the full pipeline:
    1. Offline Indexing: Compression → Proposition Extraction → Hypergraph Building
    2. Online Retrieval: Router → Decomposition → Dynamic Propagation → Generation
    """
    
    def __init__(self, config: HyPDLMConfig):
        self.config = config
        
        # Initialize components
        self.ner = SpacyNER(config.spacy_model)
        self.extractor = PropositionExtractor(
            llm_model=config.llm_model,
            embedding_model=config.embedding_model,
            max_workers=config.max_workers,
            batch_size=config.batch_size
        )
        self.hypergraph_builder = HypergraphBuilder(
            ner_model=self.ner,
            embedding_model=config.embedding_model,
            batch_size=config.batch_size
        )
        self.affinity_builder = SemanticAffinityBuilder(
            lexical_weight=config.lexical_weight,
            similarity_threshold=config.affinity_threshold,
            propagation_factor=config.propagation_factor
        )
        self.masker = SemanticMasker(
            num_clusters=config.num_clusters,
            top_p_clusters=config.top_p_clusters
        )
        self.modulator = DynamicModulator(
            max_hops=config.max_hops,
            propagation_factor=config.propagation_factor,
            top_k_entities=config.top_k_entities,
            top_k_propositions=config.top_k_propositions
        )
        self.decomposer = LogicDecomposer(
            llm_model=config.llm_model,
            embedding_model=config.embedding_model
        )
        
        # Data structures (populated after indexing)
        self.hypergraph: Optional[HypergraphData] = None
        self.a_sem: Optional[Any] = None
        self.propositions: Dict[str, Proposition] = {}
        self.chunks: Dict[str, str] = {}  # chunk_id -> text
        self.prop_to_chunk: Dict[str, str] = {}  # prop_id -> chunk_id
        
        # Ensure working directory exists
        os.makedirs(config.working_dir, exist_ok=True)
    
    def index(self, passages: List[str]) -> None:
        """
        Index a list of passages (chunks).
        
        Args:
            passages: List of text passages in format "idx:content"
        """
        logger.info(f"Starting HyP-DLM indexing for {len(passages)} passages...")
        
        # Parse passages
        self.chunks = {}
        for passage in passages:
            if ':' in passage:
                idx, content = passage.split(':', 1)
                chunk_id = compute_mdhash_id(content, prefix="chunk_")
                self.chunks[chunk_id] = content
            else:
                chunk_id = compute_mdhash_id(passage, prefix="chunk_")
                self.chunks[chunk_id] = passage
        
        # Step 1: Proposition Extraction
        logger.info("Step 1: Proposition Extraction...")
        self.propositions = self.extractor.extract_propositions(self.chunks)
        
        # Track proposition to chunk mapping
        for prop_id, prop in self.propositions.items():
            self.prop_to_chunk[prop_id] = prop.source_chunk_id
        
        # Step 2: Hypergraph Building
        logger.info("Step 2: Building Hypergraph...")
        self.hypergraph = self.hypergraph_builder.build(self.propositions)
        
        # Step 3: Semantic Affinity Matrix
        logger.info("Step 3: Building A_sem Matrix...")
        if self.hypergraph.entity_embeddings is not None:
            self.a_sem = self.affinity_builder.build(
                self.hypergraph.entity_to_idx,
                self.hypergraph.entity_embeddings
            )
        
        # Step 4: Semantic Masking Clustering
        logger.info("Step 4: Fitting Semantic Masker...")
        if self.hypergraph.proposition_embeddings is not None:
            self.masker.fit(self.hypergraph.proposition_embeddings)
        
        # Save index
        self._save_index()
        
        logger.info(f"Indexing complete! {self.hypergraph.num_entities} entities, "
                    f"{self.hypergraph.num_propositions} propositions")
    
    def retrieve(self, query: str) -> PropagationResult:
        """
        Retrieve relevant entities and propositions for a query.
        
        Args:
            query: The user query
            
        Returns:
            PropagationResult with scores and top items
        """
        if self.hypergraph is None:
            raise ValueError("Index not built. Call index() first.")
        
        # Step 1: Find seed entities from query
        seed_entities = self.ner.question_ner(query)
        seed_entity_indices = []
        seed_scores = []
        
        for entity in seed_entities:
            entity_lower = entity.lower()
            if entity_lower in self.hypergraph.entity_to_idx:
                seed_entity_indices.append(self.hypergraph.entity_to_idx[entity_lower])
                seed_scores.append(1.0)
        
        # If no entities found, use embedding search
        if not seed_entity_indices:
            logger.warning("No seed entities found, using embedding fallback")
            query_embedding = self.config.embedding_model.encode([query])[0]
            if self.hypergraph.entity_embeddings is not None:
                similarities = np.dot(self.hypergraph.entity_embeddings, query_embedding)
                top_indices = np.argsort(similarities)[-5:]
                seed_entity_indices = top_indices.tolist()
                seed_scores = similarities[top_indices].tolist()
        
        # Step 2: Decompose query into guidance vectors
        guidance_vectors = self.decomposer.decompose(query)
        
        # Step 3: Dynamic propagation
        result = self.modulator.propagate(
            hypergraph=self.hypergraph,
            seed_entity_indices=seed_entity_indices,
            seed_scores=seed_scores,
            guidance_vectors=guidance_vectors,
            a_sem=self.a_sem,
            masker=self.masker
        )
        
        return result
    
    def qa(self, questions: List[Dict]) -> List[Dict]:
        """
        Answer a list of questions.
        
        Args:
            questions: List of question dicts with 'question' key
            
        Returns:
            List of question dicts with added 'answer' and 'evidence' keys
        """
        results = []
        
        for q_item in questions:
            question = q_item.get('question', '')
            
            # Retrieve
            retrieval_result = self.retrieve(question)
            
            # Get top propositions text
            top_props = []
            for prop_idx in retrieval_result.top_proposition_indices:
                prop_id = self.hypergraph.idx_to_prop.get(prop_idx)
                if prop_id and prop_id in self.propositions:
                    top_props.append(self.propositions[prop_id].text)
            
            # Get source chunks
            source_chunks = []
            for prop_idx in retrieval_result.top_proposition_indices:
                prop_id = self.hypergraph.idx_to_prop.get(prop_idx)
                if prop_id and prop_id in self.prop_to_chunk:
                    chunk_id = self.prop_to_chunk[prop_id]
                    if chunk_id in self.chunks:
                        source_chunks.append(self.chunks[chunk_id])
            
            # Generate answer using LLM
            context = "\n".join(f"- {prop}" for prop in top_props[:5])
            answer = self._generate_answer(question, context)
            
            # Build result
            result_item = q_item.copy()
            result_item['answer'] = answer
            result_item['evidence'] = top_props[:5]
            result_item['source_chunks'] = source_chunks[:3]
            result_item['num_hops'] = retrieval_result.num_hops
            
            results.append(result_item)
        
        return results
    
    def _generate_answer(self, question: str, context: str) -> str:
        """Generate answer using LLM."""
        prompt = f"""Based on the following evidence, answer the question concisely.

Evidence:
{context}

Question: {question}

Answer:"""
        
        messages = [{"role": "user", "content": prompt}]
        
        try:
            return self.config.llm_model.infer(messages)
        except Exception as e:
            logger.error(f"Answer generation failed: {e}")
            return "Unable to generate answer"
    
    def _save_index(self) -> None:
        """Save all index data to disk."""
        import json
        
        # Save hypergraph
        save_hypergraph(self.hypergraph, os.path.join(self.config.working_dir, "hypergraph"))
        
        # Save A_sem
        if self.a_sem is not None:
            save_affinity_matrix(
                self.a_sem,
                os.path.join(self.config.working_dir, "a_sem.npz")
            )
        
        # Save masker
        self.masker.save(os.path.join(self.config.working_dir, "masker.npz"))
        
        # Save propositions
        props_data = {
            prop_id: {
                "text": prop.text,
                "source_chunk_id": prop.source_chunk_id,
                "entities": prop.entities
            }
            for prop_id, prop in self.propositions.items()
        }
        with open(os.path.join(self.config.working_dir, "propositions.json"), "w") as f:
            json.dump(props_data, f, ensure_ascii=False, indent=2)
        
        # Save chunks
        with open(os.path.join(self.config.working_dir, "chunks.json"), "w") as f:
            json.dump(self.chunks, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Index saved to {self.config.working_dir}")
    
    def load_index(self) -> None:
        """Load index from disk."""
        import json
        
        # Load hypergraph
        self.hypergraph = load_hypergraph(os.path.join(self.config.working_dir, "hypergraph"))
        
        # Load A_sem
        a_sem_path = os.path.join(self.config.working_dir, "a_sem.npz")
        if os.path.exists(a_sem_path):
            self.a_sem = load_affinity_matrix(a_sem_path)
        
        # Load masker
        masker_path = os.path.join(self.config.working_dir, "masker.npz")
        if os.path.exists(masker_path):
            self.masker.load(masker_path)
        
        # Load propositions
        with open(os.path.join(self.config.working_dir, "propositions.json"), "r") as f:
            props_data = json.load(f)
        
        self.propositions = {}
        for prop_id, data in props_data.items():
            self.propositions[prop_id] = Proposition(
                id=prop_id,
                text=data["text"],
                source_chunk_id=data["source_chunk_id"],
                entities=data.get("entities", [])
            )
            self.prop_to_chunk[prop_id] = data["source_chunk_id"]
        
        # Load chunks
        with open(os.path.join(self.config.working_dir, "chunks.json"), "r") as f:
            self.chunks = json.load(f)
        
        logger.info(f"Index loaded from {self.config.working_dir}")
