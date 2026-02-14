"""End-to-end integration tests for HyP-DLM."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
from scipy import sparse


def test_hypergraph_builder():
    """Test building incidence matrix from chunks and entities."""
    from src.indexing.semantic_chunker import Chunk
    from src.indexing.ner_extractor import Entity
    from src.indexing.hypergraph_builder import HypergraphBuilder

    entities = [
        Entity(name="Albert Einstein", entity_type="PERSON"),
        Entity(name="Germany", entity_type="LOC"),
        Entity(name="Theory Of Relativity", entity_type="MISC"),
        Entity(name="Paris", entity_type="LOC"),
    ]

    chunk1 = Chunk(text="chunk1", doc_id="d1", chunk_id=0, start_char=0, end_char=10)
    chunk1.entities = [
        Entity(name="Albert Einstein", entity_type="PERSON"),
        Entity(name="Germany", entity_type="LOC"),
    ]
    chunk2 = Chunk(text="chunk2", doc_id="d1", chunk_id=1, start_char=10, end_char=20)
    chunk2.entities = [
        Entity(name="Albert Einstein", entity_type="PERSON"),
        Entity(name="Theory Of Relativity", entity_type="MISC"),
    ]
    chunk3 = Chunk(text="chunk3", doc_id="d1", chunk_id=2, start_char=20, end_char=30)
    chunk3.entities = [
        Entity(name="Paris", entity_type="LOC"),
    ]

    builder = HypergraphBuilder()
    H, entity_index, hyperedge_index = builder.build([chunk1, chunk2, chunk3], entities)

    assert H.shape == (4, 3)  # 4 entities, 3 chunks
    assert H.nnz == 5  # 2 + 2 + 1 connections

    # Einstein should be in chunks 0 and 1
    einstein_idx = entity_index["Albert Einstein"]
    assert H[einstein_idx, 0] == 1
    assert H[einstein_idx, 1] == 1


def test_masking_no_masking():
    """Test NoMasking strategy."""
    from src.indexing.masking_strategy import NoMasking

    strategy = NoMasking()
    embeddings = np.random.rand(100, 384).astype(np.float32)
    strategy.fit(embeddings)

    mask = strategy.compute_mask(np.random.rand(384), top_p=5)
    assert mask.shape == (100,)
    assert np.all(mask == 1.0)


def test_masking_kmeans():
    """Test KMeans masking strategy."""
    from src.indexing.masking_strategy import KMeansMasking

    np.random.seed(42)
    M = 200
    dim = 32
    embeddings = np.random.rand(M, dim).astype(np.float32)

    strategy = KMeansMasking(k=10)
    strategy.fit(embeddings)

    mask = strategy.compute_mask(np.random.rand(dim).astype(np.float32), top_p=3)
    assert mask.shape == (M,)
    # Should mask out some hyperedges
    assert np.sum(mask) < M
    assert np.sum(mask) > 0


def test_router():
    """Test familiarity router."""
    from src.retrieval.router import FamiliarityRouter

    config = {
        "probe_k": 5,
        "temperature": 1.0,
        "threshold_high": 0.75,
        "threshold_low": 0.45,
        "entropy_low": 1.5,
        "entropy_high": 2.5,
    }

    router = FamiliarityRouter(config)

    np.random.seed(42)
    query_emb = np.random.rand(384).astype(np.float32)
    he_embs = np.random.rand(100, 384).astype(np.float32)

    result = router.route(query_emb, he_embs)
    assert result["route"] in ("direct", "graph", "hybrid")
    assert "mean_score" in result
    assert "entropy" in result


def test_chunk_retriever():
    """Test dense chunk retrieval."""
    from src.retrieval.chunk_retriever import ChunkRetriever

    np.random.seed(42)
    retriever = ChunkRetriever({"top_k": 3})

    query_emb = np.random.rand(384).astype(np.float32)
    he_embs = np.random.rand(50, 384).astype(np.float32)

    results = retriever.retrieve(query_emb, he_embs)
    assert len(results) == 3
    assert results[0]["score"] >= results[1]["score"]  # Sorted by score


def test_passage_ranker():
    """Test PPR passage ranking."""
    from src.retrieval.passage_ranker import PassageRanker

    config = {
        "damping_factor": 0.85,
        "iterations": 10,
        "top_k_passages": 3,
        "query_weight": 0.3,
    }

    ranker = PassageRanker(config)

    N, M = 10, 20
    np.random.seed(42)

    entity_scores = np.zeros(N)
    entity_scores[0] = 1.0
    entity_scores[3] = 0.5

    H = sparse.random(N, M, density=0.3, format="csr")
    H.data[:] = 1.0

    query_emb = np.random.rand(384).astype(np.float32)
    he_embs = np.random.rand(M, 384).astype(np.float32)

    results = ranker.rank(entity_scores, H, query_emb, he_embs)
    assert len(results) == 3
    assert all("hyperedge_id" in r and "score" in r for r in results)
