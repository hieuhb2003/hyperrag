"""
Atomic Proposition Extraction Module for HyP-DLM.

This module extracts atomic propositions from text chunks using LLM.
Atomic propositions are simple, self-contained statements that can be 
verified independently.
"""

import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import numpy as np

from src.utils import LLM_Model, compute_mdhash_id

logger = logging.getLogger(__name__)


@dataclass
class Proposition:
    """Represents an atomic proposition extracted from text."""
    id: str
    text: str
    source_chunk_id: str
    embedding: Optional[np.ndarray] = None
    entities: List[str] = field(default_factory=list)
    
    def __hash__(self):
        return hash(self.id)
    
    def __eq__(self, other):
        return self.id == other.id


PROPOSITION_EXTRACTION_PROMPT = """You are an expert at extracting atomic propositions from text.

An atomic proposition is a simple, self-contained statement that:
1. Contains exactly ONE fact or relationship
2. Replaces all pronouns with their actual referents (de-contextualization)
3. Can be verified as true or false independently
4. Is grammatically complete

Examples:
- Input: "He was born in 1990 in Hanoi and worked at Google."
- Output:
  1. Nguyen Van A was born in 1990.
  2. Nguyen Van A was born in Hanoi.
  3. Nguyen Van A worked at Google.

Extract atomic propositions from the following text. Return ONLY the propositions, one per line, numbered.

Text:
{text}

Atomic Propositions:"""


class PropositionExtractor:
    """Extracts atomic propositions from text chunks using LLM."""
    
    def __init__(
        self,
        llm_model: LLM_Model,
        embedding_model: Any,
        max_workers: int = 16,
        batch_size: int = 32
    ):
        self.llm_model = llm_model
        self.embedding_model = embedding_model
        self.max_workers = max_workers
        self.batch_size = batch_size
    
    def extract_propositions(
        self, 
        chunks: Dict[str, str]
    ) -> Dict[str, Proposition]:
        """
        Extract atomic propositions from a dictionary of chunks.
        
        Args:
            chunks: Dictionary mapping chunk_id -> chunk_text
            
        Returns:
            Dictionary mapping proposition_id -> Proposition object
        """
        logger.info(f"Extracting propositions from {len(chunks)} chunks...")
        
        all_propositions: Dict[str, Proposition] = {}
        
        # Process chunks in parallel
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(self._extract_from_chunk, chunk_id, chunk_text): chunk_id
                for chunk_id, chunk_text in chunks.items()
            }
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Extracting propositions"):
                chunk_id = futures[future]
                try:
                    props = future.result()
                    for prop in props:
                        all_propositions[prop.id] = prop
                except Exception as e:
                    logger.error(f"Error extracting from chunk {chunk_id}: {e}")
        
        logger.info(f"Extracted {len(all_propositions)} propositions from {len(chunks)} chunks")
        
        # Compute embeddings in batches
        self._compute_embeddings(all_propositions)
        
        return all_propositions
    
    def _extract_from_chunk(self, chunk_id: str, chunk_text: str) -> List[Proposition]:
        """Extract propositions from a single chunk using LLM."""
        
        prompt = PROPOSITION_EXTRACTION_PROMPT.format(text=chunk_text)
        messages = [{"role": "user", "content": prompt}]
        
        try:
            response = self.llm_model.infer(messages)
            propositions = self._parse_propositions(response, chunk_id)
            return propositions
        except Exception as e:
            logger.warning(f"LLM extraction failed for chunk {chunk_id}: {e}")
            # Fallback: treat entire chunk as one proposition
            prop_id = compute_mdhash_id(chunk_text, prefix="prop_")
            return [Proposition(id=prop_id, text=chunk_text, source_chunk_id=chunk_id)]
    
    def _parse_propositions(self, response: str, chunk_id: str) -> List[Proposition]:
        """Parse LLM response into Proposition objects."""
        propositions = []
        
        lines = response.strip().split('\n')
        for line in lines:
            # Remove numbering (e.g., "1.", "1)", "1:")
            line = line.strip()
            if not line:
                continue
            
            # Remove common prefixes
            for prefix in ['1.', '2.', '3.', '4.', '5.', '6.', '7.', '8.', '9.', '10.']:
                if line.startswith(prefix):
                    line = line[len(prefix):].strip()
                    break
            
            # Skip empty or too short
            if len(line) < 10:
                continue
            
            prop_id = compute_mdhash_id(line, prefix="prop_")
            propositions.append(Proposition(
                id=prop_id,
                text=line,
                source_chunk_id=chunk_id
            ))
        
        return propositions
    
    def _compute_embeddings(self, propositions: Dict[str, Proposition]) -> None:
        """Compute embeddings for all propositions in batches."""
        logger.info(f"Computing embeddings for {len(propositions)} propositions...")
        
        prop_list = list(propositions.values())
        texts = [p.text for p in prop_list]
        
        # Batch encode
        embeddings = self.embedding_model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True
        )
        
        # Assign embeddings back
        for i, prop in enumerate(prop_list):
            prop.embedding = embeddings[i]
        
        logger.info("Embeddings computed successfully")
