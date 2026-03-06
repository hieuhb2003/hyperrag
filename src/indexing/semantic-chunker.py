"""Semantic chunking: split documents into semantic chunks (hyperedges).
Uses sentence embeddings + cosine similarity breakpoints."""

from __future__ import annotations

import importlib
import re
from dataclasses import dataclass, field

import numpy as np

from src.config import Config

_logging = importlib.import_module("src.logging-setup")
get_logger = _logging.get_logger

logger = get_logger("chunker")


@dataclass
class Chunk:
    id: str  # "{doc_id}_chunk_{idx}"
    doc_id: str
    text: str
    embedding: np.ndarray  # 768-dim
    token_count: int
    sentence_indices: list[int] = field(default_factory=list)


class SemanticChunker:
    """Split documents into semantic chunks using embedding-based breakpoints."""

    def __init__(self, config: Config):
        self.config = config
        self.embed_model = None

    def load_model(self):
        """Load sentence-transformers model."""
        from sentence_transformers import SentenceTransformer
        self.embed_model = SentenceTransformer(
            self.config.embedding_model,
            device=self.config.device,
        )
        logger.info(f"Loaded embedding model: {self.config.embedding_model}")

    def get_embed_model(self):
        """Get the embedding model (load if needed). For reuse by other modules."""
        if self.embed_model is None:
            self.load_model()
        return self.embed_model

    def chunk_documents(self, documents: list) -> list[Chunk]:
        """Chunk all documents into semantic chunks."""
        from tqdm import tqdm

        if self.embed_model is None:
            self.load_model()

        all_chunks = []
        for doc in tqdm(documents, desc="Chunking"):
            chunks = self._chunk_single(doc)
            all_chunks.extend(chunks)
            logger.debug(f"Doc {doc.id}: {len(chunks)} chunks")

        if all_chunks:
            sizes = [c.token_count for c in all_chunks]
            logger.info(f"Total chunks: {len(all_chunks)}, "
                       f"avg={np.mean(sizes):.0f}, min={np.min(sizes)}, max={np.max(sizes)} tokens")
        return all_chunks

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        """Encode texts using the embedding model. Public for reuse."""
        if self.embed_model is None:
            self.load_model()
        return self.embed_model.encode(texts, show_progress_bar=False, convert_to_numpy=True)

    def _chunk_single(self, doc) -> list[Chunk]:
        """Chunk a single document."""
        sentences = self._split_sentences(doc.text)
        if not sentences:
            return []

        # Single sentence → single chunk
        if len(sentences) == 1:
            emb = self.embed_model.encode(sentences, show_progress_bar=False, convert_to_numpy=True)
            return [self._make_chunk(doc.id, 0, sentences, emb, [0])]

        # Compute sentence embeddings
        embeddings = self.embed_model.encode(sentences, show_progress_bar=False, convert_to_numpy=True)

        # Cosine similarities between consecutive sentences
        similarities = []
        for i in range(len(embeddings) - 1):
            sim = self._cosine_sim(embeddings[i], embeddings[i + 1])
            similarities.append(sim)

        # Breakpoints: similarity below percentile threshold
        if similarities:
            threshold = np.percentile(similarities, 100 - self.config.chunk_breakpoint_percentile)
            breakpoints = [i + 1 for i, sim in enumerate(similarities) if sim < threshold]
        else:
            breakpoints = []

        # Group sentences
        groups = self._group_sentences(sentences, breakpoints, embeddings)

        # Enforce min/max token constraints
        chunks = self._enforce_constraints(doc.id, groups)
        return chunks

    def _split_sentences(self, text: str) -> list[str]:
        """Split text into sentences using regex (avoids spacy dependency for chunking)."""
        # Split on sentence-ending punctuation followed by space or end
        parts = re.split(r'(?<=[.!?])\s+', text.strip())
        return [s.strip() for s in parts if s.strip()]

    def _cosine_sim(self, a: np.ndarray, b: np.ndarray) -> float:
        """Cosine similarity between two vectors."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))

    def _group_sentences(self, sentences, breakpoints, embeddings):
        """Group sentences by breakpoints into chunk groups."""
        groups = []
        prev = 0
        for bp in breakpoints:
            groups.append({
                "sentences": sentences[prev:bp],
                "indices": list(range(prev, bp)),
                "embeddings": embeddings[prev:bp],
            })
            prev = bp
        groups.append({
            "sentences": sentences[prev:],
            "indices": list(range(prev, len(sentences))),
            "embeddings": embeddings[prev:],
        })
        return groups

    def _enforce_constraints(self, doc_id: str, groups: list[dict]) -> list[Chunk]:
        """Merge small groups, split large groups to meet token limits."""
        min_tok = self.config.chunk_min_tokens
        max_tok = self.config.chunk_max_tokens

        # First pass: merge tiny groups with neighbors
        merged = []
        buffer = {"sentences": [], "indices": [], "embeddings": []}

        for g in groups:
            token_count = sum(len(s.split()) for s in g["sentences"])
            buf_tokens = sum(len(s.split()) for s in buffer["sentences"])

            if buf_tokens + token_count <= max_tok:
                buffer["sentences"].extend(g["sentences"])
                buffer["indices"].extend(g["indices"])
                buffer["embeddings"].extend(
                    g["embeddings"] if isinstance(g["embeddings"], list) else list(g["embeddings"])
                )
            else:
                if buffer["sentences"]:
                    merged.append(buffer)
                buffer = {
                    "sentences": list(g["sentences"]),
                    "indices": list(g["indices"]),
                    "embeddings": list(g["embeddings"]) if isinstance(g["embeddings"], list) else list(g["embeddings"]),
                }
        if buffer["sentences"]:
            merged.append(buffer)

        # Second pass: if any group is still < min_tokens, merge with previous
        final_groups = []
        for g in merged:
            token_count = sum(len(s.split()) for s in g["sentences"])
            if final_groups and token_count < min_tok:
                prev = final_groups[-1]
                prev_tokens = sum(len(s.split()) for s in prev["sentences"])
                if prev_tokens + token_count <= max_tok:
                    prev["sentences"].extend(g["sentences"])
                    prev["indices"].extend(g["indices"])
                    prev["embeddings"].extend(g["embeddings"])
                    continue
            final_groups.append(g)

        # Create Chunk objects
        chunks = []
        for idx, g in enumerate(final_groups):
            embs = np.array(g["embeddings"]) if g["embeddings"] else np.zeros((1, 768))
            chunks.append(self._make_chunk(doc_id, idx, g["sentences"], embs, g["indices"]))
        return chunks

    def _make_chunk(self, doc_id, idx, sentences, embeddings, indices) -> Chunk:
        """Create a Chunk object with mean embedding."""
        text = " ".join(sentences)
        if isinstance(embeddings, np.ndarray) and embeddings.ndim == 2:
            embedding = np.mean(embeddings, axis=0)
        elif isinstance(embeddings, np.ndarray) and embeddings.ndim == 1:
            embedding = embeddings
        else:
            embedding = np.mean(np.array(embeddings), axis=0)

        return Chunk(
            id=f"{doc_id}_chunk_{idx}",
            doc_id=doc_id,
            text=text,
            embedding=embedding,
            token_count=len(text.split()),
            sentence_indices=list(indices),
        )
