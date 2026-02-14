"""
Semantic Chunker — splits documents into semantically coherent chunks.

Each chunk becomes a hyperedge in the hypergraph.

Two strategies:
  1. SemanticChunker — cosine similarity breakpoint detection (embedding-based)
  2. AnchorChunker  — LLM-based knowledge boundary detection (near-zero output tokens)

Factory: create_chunker(config) returns the right class based on config["method"].
"""

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from src.utils.logger import get_logger, get_progress
from src.utils.embedding import EmbeddingModel

logger = get_logger(__name__)


@dataclass
class Chunk:
    """A semantic chunk = one hyperedge in the hypergraph."""
    text: str
    doc_id: str
    chunk_id: int
    start_char: int
    end_char: int
    entities: list = field(default_factory=list)
    metadata: Optional[dict] = field(default=None)

    @property
    def token_count(self) -> int:
        return len(self.text.split())


# ══════════════════════════════════════════════════════════
# Shared post-processing
# ══════════════════════════════════════════════════════════

def _post_process_groups(
    groups: list[list[str]],
    full_text: str,
    doc_id: str,
    min_chunk_tokens: int,
    max_chunk_tokens: int,
    overlap_sentences: int,
) -> list[Chunk]:
    """Merge small groups, split large groups, build Chunk objects.

    Shared between SemanticChunker and AnchorChunker.
    """
    # Merge small groups
    merged = []
    buffer: list[str] = []
    for group in groups:
        buffer.extend(group)
        token_count = sum(len(s.split()) for s in buffer)
        if token_count >= min_chunk_tokens:
            merged.append(buffer)
            if overlap_sentences > 0:
                buffer = buffer[-overlap_sentences:]
            else:
                buffer = []
    if buffer:
        if merged:
            merged[-1].extend(buffer)
        else:
            merged.append(buffer)

    # Split large groups
    final_groups = []
    for group in merged:
        text_joined = " ".join(group)
        tokens = text_joined.split()
        if len(tokens) > max_chunk_tokens:
            for i in range(0, len(tokens), max_chunk_tokens):
                sub = " ".join(tokens[i : i + max_chunk_tokens])
                final_groups.append(sub)
        else:
            final_groups.append(text_joined)

    # Build Chunk objects with character offsets
    chunks = []
    search_start = 0
    for idx, chunk_text in enumerate(final_groups):
        start = full_text.find(chunk_text[:50], search_start)
        if start == -1:
            start = search_start
        end = start + len(chunk_text)
        search_start = max(search_start, start + 1)

        chunks.append(
            Chunk(
                text=chunk_text,
                doc_id=doc_id,
                chunk_id=idx,
                start_char=start,
                end_char=end,
            )
        )

    return chunks


# ══════════════════════════════════════════════════════════
# Strategy 1: Semantic (embedding-based) Chunker
# ══════════════════════════════════════════════════════════

class SemanticChunker:
    """Split documents into semantically coherent knowledge fragments."""

    def __init__(self, config: dict):
        self.similarity_threshold = config.get("similarity_threshold", 0.5)
        self.min_chunk_tokens = config.get("min_chunk_tokens", 50)
        self.max_chunk_tokens = config.get("max_chunk_tokens", 300)
        self.overlap_sentences = config.get("overlap_sentences", 1)
        self.embedding_model_name = config.get(
            "embedding_model", "sentence-transformers/all-MiniLM-L6-v2"
        )
        self._embedder: Optional[EmbeddingModel] = None

        logger.step(
            "SemanticChunker",
            "Initialized",
            threshold=self.similarity_threshold,
            min_tokens=self.min_chunk_tokens,
            max_tokens=self.max_chunk_tokens,
        )

    @property
    def embedder(self) -> EmbeddingModel:
        if self._embedder is None:
            self._embedder = EmbeddingModel(model_name=self.embedding_model_name)
        return self._embedder

    def set_embedder(self, embedder: EmbeddingModel) -> None:
        """Allow sharing an embedder instance across components."""
        self._embedder = embedder

    @staticmethod
    def sentence_tokenize(text: str) -> list[str]:
        """Split text into sentences using regex-based rules."""
        pattern = r'(?<=[.!?])\s+(?=[A-Z])|(?<=[.!?])\s*$'
        raw_splits = re.split(pattern, text.strip())
        sentences = [s.strip() for s in raw_splits if s and s.strip()]
        return sentences

    def chunk_document(self, text: str, doc_id: str = "doc_0") -> list[Chunk]:
        """
        Chunk a single document into semantic fragments.

        Algorithm:
        1. Split into sentences
        2. Compute sentence embeddings
        3. Find breakpoints via cosine similarity drops
        4. Group sentences between breakpoints
        5. Post-process: merge small, split large
        """
        logger.start_timer(f"chunk_{doc_id}")
        logger.debug(f"Chunking document '{doc_id}': {len(text)} chars")

        # Step 1: Sentence tokenization
        sentences = self.sentence_tokenize(text)
        if not sentences:
            logger.warning(f"No sentences found in document '{doc_id}'")
            return []
        logger.debug(f"  Sentence count: {len(sentences)}")

        # Step 2: Compute sentence embeddings
        embeddings = self.embedder.encode(sentences)
        logger.debug(f"  Sentence embeddings: {embeddings.shape}")

        # Step 3: Compute consecutive cosine similarities & find breakpoints
        if len(sentences) < 2:
            groups = [sentences]
        else:
            sims = []
            for i in range(len(embeddings) - 1):
                sim = float(np.dot(embeddings[i], embeddings[i + 1]))
                sims.append(sim)

            logger.debug(
                f"  Cosine similarities: min={min(sims):.3f}, "
                f"max={max(sims):.3f}, mean={np.mean(sims):.3f}"
            )

            breakpoints = [
                i + 1 for i, s in enumerate(sims)
                if s < self.similarity_threshold
            ]
            logger.debug(f"  Breakpoints at positions: {breakpoints}")

            groups = []
            prev = 0
            for bp in breakpoints:
                groups.append(sentences[prev:bp])
                prev = bp
            groups.append(sentences[prev:])

        # Step 5: Post-process
        chunks = _post_process_groups(
            groups, text, doc_id,
            self.min_chunk_tokens, self.max_chunk_tokens, self.overlap_sentences,
        )

        elapsed = logger.stop_timer(f"chunk_{doc_id}")
        logger.step(
            "SemanticChunker",
            f"Document '{doc_id}' -> {len(chunks)} chunks",
            sentences=len(sentences),
            time=elapsed,
        )

        token_counts = [c.token_count for c in chunks]
        if token_counts:
            logger.debug(
                f"  Chunk tokens: min={min(token_counts)}, max={max(token_counts)}, "
                f"mean={np.mean(token_counts):.1f}"
            )

        return chunks

    def _post_process(
        self, groups: list[list[str]], full_text: str, doc_id: str
    ) -> list[Chunk]:
        """Backward-compat wrapper around module-level _post_process_groups."""
        return _post_process_groups(
            groups, full_text, doc_id,
            self.min_chunk_tokens, self.max_chunk_tokens, self.overlap_sentences,
        )

    def chunk_documents(
        self, documents: list[dict[str, str]]
    ) -> list[Chunk]:
        """
        Chunk multiple documents.

        Args:
            documents: list of {"id": str, "text": str}

        Returns:
            All chunks across all documents.
        """
        all_chunks = []
        with get_progress() as progress:
            task = progress.add_task(
                "Semantic chunking...", total=len(documents)
            )
            for doc in documents:
                chunks = self.chunk_document(
                    text=doc["text"], doc_id=doc["id"]
                )
                all_chunks.extend(chunks)
                progress.advance(task)

        logger.step(
            "SemanticChunker",
            f"Total: {len(all_chunks)} chunks from {len(documents)} documents",
        )
        return all_chunks

    def chunk_documents_parallel(
        self, documents: list[dict[str, str]], max_workers: int = 4
    ) -> list[Chunk]:
        """Compatibility stub — delegates to sequential (embedding model is not thread-safe)."""
        logger.debug("SemanticChunker: parallel not supported, falling back to sequential")
        return self.chunk_documents(documents)


# ══════════════════════════════════════════════════════════
# Strategy 2: LLM Anchor-Based Chunker
# ══════════════════════════════════════════════════════════

_ANCHOR_PROMPT = """You are a text segmentation expert. Below is a text where sentence boundaries \
have been marked with numbers in brackets like [1], [2], [3], etc.

Identify which markers start a NEW knowledge segment (a coherent block of \
related information about the same topic/event/fact).

Return ONLY the marker numbers that begin a new segment, separated by commas. \
The first marker [1] always starts a segment.

Text:
{marked_text}

Answer (comma-separated numbers only):"""


class AnchorChunker:
    """Split documents using LLM-identified knowledge boundaries.

    The LLM sees sentence boundaries marked with [1], [2], ... and returns
    which markers begin a new knowledge segment. Near-zero output tokens.
    """

    def __init__(self, config: dict):
        self.llm_model = config.get("llm_model", "gpt-4o-mini")
        self.llm_temperature = config.get("llm_temperature", 0.0)
        self.llm_max_tokens = config.get("llm_max_tokens", 256)
        self.min_chunk_tokens = config.get("min_chunk_tokens", 50)
        self.max_chunk_tokens = config.get("max_chunk_tokens", 300)
        self.overlap_sentences = config.get("overlap_sentences", 1)
        self._llm = None

        logger.step(
            "AnchorChunker",
            "Initialized",
            llm_model=self.llm_model,
            min_tokens=self.min_chunk_tokens,
            max_tokens=self.max_chunk_tokens,
        )

    def set_llm(self, llm_client) -> None:
        """Inject a shared LLMClient instance."""
        self._llm = llm_client

    @property
    def llm(self):
        if self._llm is None:
            from src.utils.llm_client import LLMClient
            self._llm = LLMClient(
                model=self.llm_model,
                temperature=self.llm_temperature,
                max_tokens=self.llm_max_tokens,
            )
        return self._llm

    # ── Sentence splitting ──

    @staticmethod
    def robust_sentence_split(text: str) -> list[str]:
        """Split text into sentences, handling abbreviations, decimals, URLs, and ellipsis.

        Uses a protect-replace-restore pattern: known non-boundary periods are
        temporarily replaced with a placeholder before splitting.
        """
        if not text or not text.strip():
            return []

        working = text.strip()

        # Protect patterns that contain periods but are NOT sentence boundaries
        protections: list[tuple[str, str]] = []
        counter = [0]

        def _protect(match: re.Match) -> str:
            placeholder = f"\x00PROTECT{counter[0]}\x00"
            protections.append((placeholder, match.group(0)))
            counter[0] += 1
            return placeholder

        # URLs (http://... or www....)
        working = re.sub(r'https?://\S+|www\.\S+', _protect, working)
        # Common abbreviations: Dr. Mr. Mrs. Ms. Prof. St. Jr. Sr. Inc. Ltd. Corp. Co. vs. etc. e.g. i.e.
        working = re.sub(
            r'\b(Dr|Mr|Mrs|Ms|Prof|St|Jr|Sr|Inc|Ltd|Corp|Co|vs|etc|e\.g|i\.e|U\.S|U\.K|U\.N)\.',
            _protect, working
        )
        # Decimals: 3.14, 0.5, etc.
        working = re.sub(r'\b\d+\.\d+', _protect, working)
        # Ellipsis
        working = re.sub(r'\.{2,}', _protect, working)

        # Split on sentence-ending punctuation followed by whitespace
        pattern = r'(?<=[.!?])\s+(?=[A-Z"\'])|(?<=[.!?])\s*$'
        raw_splits = re.split(pattern, working)
        sentences = [s.strip() for s in raw_splits if s and s.strip()]

        # Restore protected tokens
        restored = []
        for sent in sentences:
            for placeholder, original in protections:
                sent = sent.replace(placeholder, original)
            restored.append(sent)

        return restored

    # ── Marker insertion ──

    @staticmethod
    def _insert_markers(sentences: list[str]) -> tuple[str, dict[int, int]]:
        """Insert [1], [2], ... at each sentence boundary.

        Returns:
            marked_text: text with markers inserted
            marker_map: {marker_number: sentence_index}
        """
        parts = []
        marker_map = {}
        for i, sent in enumerate(sentences):
            marker_num = i + 1  # 1-indexed
            marker_map[marker_num] = i
            parts.append(f"[{marker_num}] {sent}")
        marked_text = " ".join(parts)
        return marked_text, marker_map

    # ── LLM response parsing ──

    @staticmethod
    def _parse_llm_response(response: str, max_marker: int) -> list[int]:
        """Extract boundary marker numbers from LLM response.

        Handles: bare numbers, bracket-wrapped [3], ranges, out-of-range values.
        Always ensures marker 1 is present.
        """
        # Extract all integers from the response
        numbers = [int(x) for x in re.findall(r'\d+', response)]

        # Filter to valid range
        valid = sorted(set(n for n in numbers if 1 <= n <= max_marker))

        # Ensure [1] is always present
        if not valid or valid[0] != 1:
            valid = [1] + [n for n in valid if n != 1]
            valid.sort()

        return valid

    # ── Core chunking ──

    def chunk_document(self, text: str, doc_id: str = "doc_0") -> list[Chunk]:
        """Chunk a single document using LLM anchor-based segmentation."""
        logger.start_timer(f"anchor_chunk_{doc_id}")
        logger.debug(f"Anchor chunking document '{doc_id}': {len(text)} chars")

        # Step 1: Robust sentence splitting
        sentences = self.robust_sentence_split(text)
        if not sentences:
            logger.warning(f"No sentences found in document '{doc_id}'")
            return []
        logger.debug(f"  Sentence count: {len(sentences)}")

        # Short-circuit: if only 1 sentence, no need for LLM
        if len(sentences) == 1:
            groups = [sentences]
        else:
            # Step 2: Insert markers
            marked_text, marker_map = self._insert_markers(sentences)

            # Step 3: LLM call
            prompt = _ANCHOR_PROMPT.format(marked_text=marked_text)
            response = self.llm.generate(
                prompt,
                temperature=self.llm_temperature,
                max_tokens=self.llm_max_tokens,
            )
            logger.debug(f"  LLM anchor response: {response.strip()!r}")

            # Step 4: Parse response into boundary indices
            boundaries = self._parse_llm_response(response, max_marker=len(sentences))
            logger.debug(f"  Segment boundaries at markers: {boundaries}")

            # Step 5: Group sentences between boundaries
            boundary_indices = [marker_map[m] for m in boundaries]
            groups = []
            for k in range(len(boundary_indices)):
                start_idx = boundary_indices[k]
                end_idx = boundary_indices[k + 1] if k + 1 < len(boundary_indices) else len(sentences)
                groups.append(sentences[start_idx:end_idx])

        # Step 6: Post-process (merge small, split large)
        chunks = _post_process_groups(
            groups, text, doc_id,
            self.min_chunk_tokens, self.max_chunk_tokens, self.overlap_sentences,
        )

        elapsed = logger.stop_timer(f"anchor_chunk_{doc_id}")
        logger.step(
            "AnchorChunker",
            f"Document '{doc_id}' -> {len(chunks)} chunks",
            sentences=len(sentences),
            time=elapsed,
        )

        token_counts = [c.token_count for c in chunks]
        if token_counts:
            logger.debug(
                f"  Chunk tokens: min={min(token_counts)}, max={max(token_counts)}, "
                f"mean={np.mean(token_counts):.1f}"
            )

        return chunks

    def chunk_documents(
        self, documents: list[dict[str, str]]
    ) -> list[Chunk]:
        """Chunk multiple documents sequentially."""
        all_chunks = []
        with get_progress() as progress:
            task = progress.add_task(
                "Anchor chunking...", total=len(documents)
            )
            for doc in documents:
                chunks = self.chunk_document(
                    text=doc["text"], doc_id=doc["id"]
                )
                all_chunks.extend(chunks)
                progress.advance(task)

        logger.step(
            "AnchorChunker",
            f"Total: {len(all_chunks)} chunks from {len(documents)} documents",
        )
        return all_chunks

    def chunk_documents_parallel(
        self, documents: list[dict[str, str]], max_workers: int = 4
    ) -> list[Chunk]:
        """Chunk multiple documents in parallel using ThreadPoolExecutor.

        LLM calls are I/O-bound and openai.OpenAI is thread-safe, so
        parallel dispatch provides near-linear speedup.
        """
        if len(documents) <= 1:
            return self.chunk_documents(documents)

        logger.step(
            "AnchorChunker",
            f"Parallel chunking: {len(documents)} docs with {max_workers} workers",
        )

        # Submit all documents for parallel processing
        results: dict[int, list[Chunk]] = {}
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_idx = {
                executor.submit(self.chunk_document, doc["text"], doc["id"]): i
                for i, doc in enumerate(documents)
            }
            with get_progress() as progress:
                task = progress.add_task(
                    "Parallel anchor chunking...", total=len(documents)
                )
                for future in as_completed(future_to_idx):
                    idx = future_to_idx[future]
                    results[idx] = future.result()
                    progress.advance(task)

        # Reassemble in document order
        all_chunks = []
        for i in range(len(documents)):
            all_chunks.extend(results[i])

        logger.step(
            "AnchorChunker",
            f"Total: {len(all_chunks)} chunks from {len(documents)} documents (parallel)",
        )
        return all_chunks


# ══════════════════════════════════════════════════════════
# Factory
# ══════════════════════════════════════════════════════════

def create_chunker(config: dict):
    """Create a chunker based on config["method"].

    Returns:
        SemanticChunker or AnchorChunker instance.
    """
    method = config.get("method", "similarity")
    if method == "similarity":
        return SemanticChunker(config)
    elif method == "llm_anchor":
        return AnchorChunker(config)
    else:
        raise ValueError(f"Unknown chunking method: {method!r}. Use 'similarity' or 'llm_anchor'.")
