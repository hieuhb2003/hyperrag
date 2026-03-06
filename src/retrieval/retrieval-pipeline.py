"""Retrieval pipeline: orchestrates query decomposition, propagation, and scoring."""

from __future__ import annotations

import importlib

import numpy as np
from tqdm import tqdm

from src.config import Config

_logging = importlib.import_module("src.logging-setup")
_tt = importlib.import_module("src.token-tracker")
_timer = importlib.import_module("src.time-tracker")
_qd = importlib.import_module("src.retrieval.query-decomposer")
_si = importlib.import_module("src.retrieval.state-initializer")
_dp = importlib.import_module("src.retrieval.dag-propagator")
_ms = importlib.import_module("src.retrieval.multi-granular-scorer")

logger = _logging.get_logger("retrieval")


def run_retrieval(config: Config, queries_path: str, storage_path=None):
    """Entry point for retrieval command. Loads index, runs retrieval for all queries."""
    _dl = importlib.import_module("src.data-loader")

    queries = _dl.load_queries(queries_path)
    index_data = _load_index(config, storage_path)

    token_tracker = _tt.TokenTracker(config.metrics_dir)
    time_tracker = _timer.TimeTracker(config.metrics_dir)

    pipeline = RetrievalPipeline(config, token_tracker, time_tracker)
    pipeline.setup(index_data)

    results = pipeline.run_all(queries, index_data)

    # Save retrieval results to JSON for recall evaluation
    _os = importlib.import_module("src.evaluation.output-saver")
    saver = _os.OutputSaver(config)
    saver.save_retrieval_results(results, config.dataset_name)

    # Save metrics
    token_tracker.save(config.dataset_name)
    time_tracker.save(config.dataset_name)

    logger.info(f"Retrieval complete: {len(results)} queries processed")
    return results


class RetrievalPipeline:
    """Orchestrate retrieval for multiple queries."""

    def __init__(self, config: Config, token_tracker: _tt.TokenTracker,
                 time_tracker: _timer.TimeTracker):
        self.config = config
        self.token_tracker = token_tracker
        self.time_tracker = time_tracker
        self.decomposer = _qd.QueryDecomposer(config, token_tracker)
        self.initializer = _si.StateInitializer(config)
        self.propagator = _dp.DAGPropagator(config, token_tracker)
        self.scorer = _ms.MultiGranularScorer(config)

    def setup(self, index_data: dict):
        """Set shared models from index data."""
        embed_model = index_data.get("embed_model")
        if embed_model:
            self.initializer.set_embed_model(embed_model)
            self.propagator.set_embed_model(embed_model)

        ner_extractor = index_data.get("ner_extractor")
        if ner_extractor:
            self.initializer.set_ner_extractor(ner_extractor)

    def run_all(self, queries: list, index_data: dict) -> list[dict]:
        """Run retrieval for all queries."""
        results = []
        for query in tqdm(queries, desc="Retrieval"):
            result = self.run_single(query, index_data)
            results.append(result)
        return results

    def run_single(self, query, index_data: dict) -> dict:
        """Run full retrieval pipeline for a single query."""
        with self.time_tracker.track("retrieval", query.id):
            # 1. Decompose query into DAG
            dag = self.decomposer.decompose(query.query, query.id)

            # 2. DAG-guided propagation
            entity_states, intermediate = self.propagator.propagate(
                dag=dag,
                H_norm=index_data["hypergraph"].H_norm,
                S_conf=index_data["S_conf"],
                semantic_mask=index_data["semantic_mask"],
                chunks=index_data["chunks"],
                entities=index_data["entities"],
                entity_embeddings=index_data["entity_embeddings"],
                state_initializer=self.initializer,
                query_decomposer=self.decomposer,
                query_id=query.id,
            )

            # 3. Multi-granular scoring
            # Compute query embedding for hyperedge scoring
            query_emb = self.propagator.embed_model.encode(
                [query.query], show_progress_bar=False, convert_to_numpy=True
            )[0]
            chunk_embeddings = np.array([c.embedding for c in index_data["chunks"]])

            top_results = self.scorer.score(
                entity_states=entity_states,
                H_norm=index_data["hypergraph"].H_norm,
                chunks=index_data["chunks"],
                entities=index_data["entities"],
                chunk_embeddings=chunk_embeddings,
                query_embedding=query_emb,
            )

        return {
            "query_id": query.id,
            "query": query.query,
            "dag": dag,
            "top_results": top_results,
            "intermediate_answers": intermediate,
            "entity_states": entity_states,
        }


def _load_index(config: Config, storage_path=None) -> dict:
    """Load index artifacts from disk."""
    import json
    from pathlib import Path

    path = Path(storage_path) if storage_path else config.storage_path()

    _imb = importlib.import_module("src.indexing.incidence-matrix-builder")
    _smk = importlib.import_module("src.indexing.semantic-masker")
    _eb = importlib.import_module("src.indexing.entity-extractor-base")
    _sc = importlib.import_module("src.indexing.semantic-chunker")

    import scipy.sparse as sp

    # Load hypergraph
    matrix_builder = _imb.IncidenceMatrixBuilder(config)
    hypergraph = matrix_builder.load(str(path))

    # Load S_conf
    S_conf = sp.load_npz(str(path / "synonym_conf.npz"))

    # Load chunks
    with open(path / "chunks.json") as f:
        chunk_data = json.load(f)
    chunk_embeddings = np.load(str(path / "chunk_embeddings.npy"))
    chunks = []
    for i, cd in enumerate(chunk_data):
        chunks.append(_sc.Chunk(
            id=cd["id"], doc_id=cd["doc_id"], text=cd["text"],
            embedding=chunk_embeddings[i], token_count=cd["token_count"],
        ))

    # Load entities
    with open(path / "entities.json") as f:
        entity_data = json.load(f)
    entities = [_eb.Entity(id=ed["id"], text=ed["text"], type=ed["type"],
                           chunk_ids=ed["chunk_ids"]) for ed in entity_data]

    # Load semantic mask
    masker = _smk.SemanticMasker(config)
    semantic_mask = masker.load(str(path))

    # Load embedding model
    chunker = _sc.SemanticChunker(config)
    chunker.load_model()

    # Compute entity embeddings
    entity_embeddings = chunker.encode_texts([e.text for e in entities])

    # Load NER extractor
    ner_extractor = _eb.create_entity_extractor(config)
    ner_extractor.load_model()

    logger.info(f"Loaded index from {path}: {len(chunks)} chunks, {len(entities)} entities")

    return {
        "hypergraph": hypergraph,
        "S_conf": S_conf,
        "chunks": chunks,
        "entities": entities,
        "entity_embeddings": entity_embeddings,
        "semantic_mask": semantic_mask,
        "embed_model": chunker.embed_model,
        "ner_extractor": ner_extractor,
    }
