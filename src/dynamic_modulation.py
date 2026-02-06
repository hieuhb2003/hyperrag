"""
Dynamic Logic Modulation Module for HyP-DLM.

This module implements the core retrieval algorithm with attention-weighted
propagation through the hypergraph. Unlike static PageRank, the weights
change dynamically based on the current guidance vector.
"""

import logging
from typing import List, Tuple, Optional, Dict
import numpy as np
from scipy import sparse
from dataclasses import dataclass

from src.hypergraph import HypergraphData
from src.semantic_affinity import SemanticAffinityBuilder
from src.semantic_masking import SemanticMasker

logger = logging.getLogger(__name__)


@dataclass
class PropagationResult:
    """Result from dynamic propagation."""
    entity_scores: np.ndarray
    proposition_scores: np.ndarray
    top_entity_indices: List[int]
    top_proposition_indices: List[int]
    num_hops: int


class DynamicModulator:
    """
    Implements Dynamic Logic Modulation for hypergraph traversal.
    
    The key formula:
    x_{t+1} = H^T × D_t × H × x_t + β × A_sem × x_t
    
    Where:
    - H: Incidence matrix (entities × propositions)
    - D_t: Diagonal attention matrix based on guidance vector g_t
    - A_sem: Semantic affinity matrix for entity aliases
    - β: Propagation factor for alias leakage
    """
    
    def __init__(
        self,
        max_hops: int = 5,
        convergence_threshold: float = 1e-4,
        propagation_factor: float = 0.3,
        top_k_entities: int = 20,
        top_k_propositions: int = 10
    ):
        """
        Initialize the modulator.
        
        Args:
            max_hops: Maximum number of propagation steps (T)
            convergence_threshold: Stop if score change is below this
            propagation_factor: β for A_sem alias propagation
            top_k_entities: Number of top entities to return
            top_k_propositions: Number of top propositions to return
        """
        self.max_hops = max_hops
        self.convergence_threshold = convergence_threshold
        self.propagation_factor = propagation_factor
        self.top_k_entities = top_k_entities
        self.top_k_propositions = top_k_propositions
    
    def propagate(
        self,
        hypergraph: HypergraphData,
        seed_entity_indices: List[int],
        seed_scores: List[float],
        guidance_vectors: List[np.ndarray],
        a_sem: Optional[sparse.csr_matrix] = None,
        masker: Optional[SemanticMasker] = None
    ) -> PropagationResult:
        """
        Run dynamic propagation through the hypergraph.
        
        Args:
            hypergraph: The hypergraph data structure
            seed_entity_indices: Indices of seed entities (from query)
            seed_scores: Initial scores for seed entities
            guidance_vectors: List of guidance vectors [g_1, g_2, ..., g_T]
            a_sem: Optional semantic affinity matrix for alias propagation
            masker: Optional semantic masker for cluster filtering
            
        Returns:
            PropagationResult with final scores and top items
        """
        num_entities = hypergraph.num_entities
        num_props = hypergraph.num_propositions
        H = hypergraph.incidence_matrix
        prop_embeddings = hypergraph.proposition_embeddings
        
        # Initialize entity score vector
        x = np.zeros(num_entities, dtype=np.float32)
        for idx, score in zip(seed_entity_indices, seed_scores):
            if 0 <= idx < num_entities:
                x[idx] = score
        
        logger.debug(f"Starting propagation with {len(seed_entity_indices)} seed entities")
        
        # Track for convergence
        prev_x = x.copy()
        actual_hops = 0
        
        # Propagation loop
        for t, g_t in enumerate(guidance_vectors[:self.max_hops]):
            actual_hops = t + 1
            
            # Step 1: Semantic Masking (if available)
            if masker is not None and prop_embeddings is not None:
                mask = masker.get_mask(g_t, num_props)
                H_masked = masker.apply_mask_to_matrix(H, mask)
            else:
                H_masked = H
            
            # Step 2: Compute attention weights D_t
            D_t = self._compute_attention_weights(g_t, prop_embeddings, num_props)
            
            # Step 3: Main propagation path
            # x_{t+1} = H^T × D_t × H × x_t
            # Break down: (1) y = H × x  (entities to props)
            #             (2) z = D_t × y (weight by attention)
            #             (3) x_new = H^T × z (props to entities)
            
            y = H_masked.T.dot(x)  # Shape: (num_props,)
            z = D_t * y            # Element-wise multiplication
            x_main = H_masked.dot(z)  # Shape: (num_entities,)
            
            # Step 4: Alias propagation path (if A_sem available)
            if a_sem is not None:
                x_alias = a_sem.dot(x) * self.propagation_factor
                x = x_main + x_alias
            else:
                x = x_main
            
            # Normalize to prevent explosion
            x_max = np.max(x)
            if x_max > 0:
                x = x / x_max
            
            # Check convergence
            diff = np.linalg.norm(x - prev_x)
            logger.debug(f"Hop {t+1}: score change = {diff:.6f}")
            
            if diff < self.convergence_threshold:
                logger.info(f"Converged after {actual_hops} hops")
                break
            
            prev_x = x.copy()
        
        # Compute proposition scores from final entity scores
        prop_scores = H.T.dot(x)
        
        # Get top-k results
        top_entity_indices = np.argsort(x)[-self.top_k_entities:][::-1].tolist()
        top_prop_indices = np.argsort(prop_scores)[-self.top_k_propositions:][::-1].tolist()
        
        return PropagationResult(
            entity_scores=x,
            proposition_scores=prop_scores,
            top_entity_indices=top_entity_indices,
            top_proposition_indices=top_prop_indices,
            num_hops=actual_hops
        )
    
    def _compute_attention_weights(
        self,
        guidance_vector: np.ndarray,
        proposition_embeddings: Optional[np.ndarray],
        num_propositions: int
    ) -> np.ndarray:
        """
        Compute attention weights D_t for propositions based on guidance vector.
        
        This is the "Dynamic" part - weights change based on what we're looking for.
        """
        if proposition_embeddings is None:
            # No embeddings, use uniform weights
            return np.ones(num_propositions, dtype=np.float32)
        
        # Normalize guidance vector
        g_norm = guidance_vector / (np.linalg.norm(guidance_vector) + 1e-8)
        
        # Normalize proposition embeddings
        prop_norms = np.linalg.norm(proposition_embeddings, axis=1, keepdims=True) + 1e-8
        prop_norm = proposition_embeddings / prop_norms
        
        # Compute cosine similarity
        similarities = np.dot(prop_norm, g_norm)
        
        # Apply softmax-like transformation to get attention weights
        # Use temperature scaling for sharper attention
        temperature = 0.1
        exp_sim = np.exp(similarities / temperature)
        attention = exp_sim / (np.sum(exp_sim) + 1e-8)
        
        # Scale back up (softmax normalizes too much)
        attention = attention * num_propositions
        
        return attention


class LogicDecomposer:
    """
    Decomposes complex queries into guidance vectors.
    
    Example:
    Query: "Wife of Microsoft founder's birth year"
    → g_1: "Founder of Microsoft"
    → g_2: "Wife"
    → g_3: "Birth year"
    """
    
    def __init__(self, llm_model, embedding_model):
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        
        self.decomposition_prompt = """Decompose the following question into a sequence of sub-questions.
Each sub-question should represent one logical step needed to answer the main question.
Return only the sub-questions, one per line, numbered.

Question: {question}

Sub-questions:"""
    
    def decompose(self, question: str) -> List[np.ndarray]:
        """
        Decompose question into guidance vectors.
        
        Args:
            question: The complex question to decompose
            
        Returns:
            List of guidance vectors (embeddings of sub-questions)
        """
        # Use LLM to decompose
        prompt = self.decomposition_prompt.format(question=question)
        messages = [{"role": "user", "content": prompt}]
        
        try:
            response = self.llm_model.infer(messages)
            sub_questions = self._parse_sub_questions(response)
        except Exception as e:
            logger.warning(f"Decomposition failed: {e}, using original question")
            sub_questions = [question]
        
        # Embed sub-questions to get guidance vectors
        guidance_vectors = self.embedding_model.encode(
            sub_questions,
            convert_to_numpy=True
        )
        
        logger.info(f"Decomposed into {len(guidance_vectors)} guidance vectors")
        return list(guidance_vectors)
    
    def _parse_sub_questions(self, response: str) -> List[str]:
        """Parse LLM response into list of sub-questions."""
        sub_questions = []
        
        for line in response.strip().split('\n'):
            line = line.strip()
            if not line:
                continue
            
            # Remove numbering
            for prefix in ['1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.', '10.']:
                if line.startswith(prefix):
                    line = line[len(prefix):].strip()
                    break
            
            if len(line) > 5:
                sub_questions.append(line)
        
        return sub_questions if sub_questions else [""]
