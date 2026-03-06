"""Multithreaded indexing pipeline: orchestrates all indexing phases.
Coref → Chunking → NER → Incidence Matrix → Synonym → Masking."""

from __future__ import annotations

import concurrent.futures
import importlib
import json
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sp
from tqdm import tqdm

from src.config import Config

_logging = importlib.import_module("src.logging-setup")
_tt = importlib.import_module("src.token-tracker")
_timer = importlib.import_module("src.time-tracker")

logger = _logging.get_logger("indexing-pipeline")


def run_indexing(config: Config, data_path: str):
    """Entry point for indexing command."""
    _dl = importlib.import_module("src.data-loader")

    documents = _dl.load_documents(data_path)
    token_tracker = _tt.TokenTracker(config.metrics_dir)
    time_tracker = _timer.TimeTracker(config.metrics_dir)

    pipeline = IndexingPipeline(config, token_tracker, time_tracker)
    result = pipeline.run(documents)

    # Save metrics
    token_tracker.save(config.dataset_name)
    time_tracker.save(config.dataset_name)

    logger.info("Indexing pipeline complete")
    return result


class IndexingPipeline:
    """Orchestrate all indexing phases with multithreading."""

    def __init__(self, config: Config, token_tracker: _tt.TokenTracker,
                 time_tracker: _timer.TimeTracker):
        self.config = config
        self.token_tracker = token_tracker
        self.time_tracker = time_tracker
        self.storage_path = config.storage_path()

    def run(self, documents: list) -> dict:
        """Run full indexing pipeline. Returns index data dict."""
        self.storage_path.mkdir(parents=True, exist_ok=True)
        start = time.perf_counter()

        # --- Phase 1: Coreference Resolution ---
        logger.info("=== Phase 1: Coreference Resolution ===")
        _coref = importlib.import_module("src.indexing.coreference-resolver")
        resolver = _coref.CoreferenceResolver(self.config)

        with self.time_tracker.track("indexing.coreference", "all"):
            resolved_docs = self._run_coreference(resolver, documents)

        # --- Phase 2: Semantic Chunking ---
        logger.info("=== Phase 2: Semantic Chunking ===")
        _chunker = importlib.import_module("src.indexing.semantic-chunker")
        chunker = _chunker.SemanticChunker(self.config)
        chunker.load_model()

        with self.time_tracker.track("indexing.chunking", "all"):
            chunks = chunker.chunk_documents(resolved_docs)

        # --- Phase 3: Entity Extraction (NER) ---
        logger.info("=== Phase 3: Entity Extraction (NER) ===")
        _ner = importlib.import_module("src.indexing.entity-extractor-base")
        extractor = _ner.create_entity_extractor(self.config)
        extractor.load_model()

        with self.time_tracker.track("indexing.ner", "all"):
            entities = extractor.extract_all(chunks)

        # --- Phase 4: Incidence Matrix ---
        logger.info("=== Phase 4: Incidence Matrix ===")
        _imb = importlib.import_module("src.indexing.incidence-matrix-builder")
        matrix_builder = _imb.IncidenceMatrixBuilder(self.config)

        with self.time_tracker.track("indexing.incidence_matrix", "all"):
            hypergraph = matrix_builder.build(entities, chunks)

        # --- Phase 5: Synonym Matrix ---
        logger.info("=== Phase 5: Synonym Matrix ===")
        _smb = importlib.import_module("src.indexing.synonym-matrix-builder")
        synonym_builder = _smb.SynonymMatrixBuilder(self.config)
        synonym_builder.set_embed_model(chunker.get_embed_model())

        with self.time_tracker.track("indexing.synonym_matrix", "all"):
            S_conf = synonym_builder.build(entities, chunks, hypergraph.entity_to_idx)

        # --- Phase 6: Semantic Masking ---
        logger.info("=== Phase 6: Semantic Masking ===")
        _mask = importlib.import_module("src.indexing.semantic-masker")
        masker = _mask.SemanticMasker(self.config)

        with self.time_tracker.track("indexing.semantic_masking", "all"):
            semantic_mask = masker.build(chunks)

        # --- Save all artifacts ---
        logger.info("=== Saving Artifacts ===")
        self._save_artifacts(chunks, entities, hypergraph, S_conf, semantic_mask, chunker)

        # Compute entity embeddings (reuse chunker's model)
        entity_embeddings = chunker.encode_texts([e.text for e in entities])

        elapsed = time.perf_counter() - start
        throughput = len(documents) / elapsed if elapsed > 0 else 0
        logger.info(f"Indexing complete: {len(documents)} docs, {len(chunks)} chunks, "
                    f"{len(entities)} entities in {elapsed:.1f}s ({throughput:.1f} docs/sec)")

        # Save config for reproducibility
        with open(self.storage_path / "config.json", "w") as f:
            json.dump(self.config.to_dict(), f, indent=2)

        return {
            "chunks": chunks,
            "entities": entities,
            "hypergraph": hypergraph,
            "S_conf": S_conf,
            "semantic_mask": semantic_mask,
            "entity_embeddings": entity_embeddings,
            "embed_model": chunker.get_embed_model(),
            "ner_extractor": extractor,
        }

    def _run_coreference(self, resolver, documents: list) -> list:
        """Run coreference with multithreading per document."""
        if self.config.max_workers <= 1:
            resolver.load_model()
            return resolver.resolve(documents)

        # Batch mode for efficiency (model not thread-safe for GPU)
        resolver.load_model()
        return resolver.resolve_batch(documents)

    def _save_artifacts(self, chunks, entities, hypergraph, S_conf, semantic_mask, chunker):
        """Save all index artifacts to storage path."""
        path = self.storage_path
        path.mkdir(parents=True, exist_ok=True)

        # Chunks: metadata + embeddings
        chunk_data = [
            {"id": c.id, "doc_id": c.doc_id, "text": c.text, "token_count": c.token_count}
            for c in chunks
        ]
        with open(path / "chunks.json", "w") as f:
            json.dump(chunk_data, f, indent=2)
        np.save(str(path / "chunk_embeddings.npy"), np.array([c.embedding for c in chunks]))

        # Entities
        entity_data = [
            {"id": e.id, "text": e.text, "type": e.type, "chunk_ids": e.chunk_ids}
            for e in entities
        ]
        with open(path / "entities.json", "w") as f:
            json.dump(entity_data, f, indent=2)

        # Matrices
        sp.save_npz(str(path / "h_norm.npz"), hypergraph.H_norm)
        sp.save_npz(str(path / "synonym_conf.npz"), S_conf)

        # Index mappings
        with open(path / "entity_map.json", "w") as f:
            json.dump(hypergraph.entity_to_idx, f)
        with open(path / "chunk_map.json", "w") as f:
            json.dump(hypergraph.chunk_to_idx, f)

        # Semantic mask
        np.save(str(path / "cluster_centroids.npy"), semantic_mask.cluster_centroids)
        with open(path / "chunk_cluster_map.json", "w") as f:
            json.dump(semantic_mask.chunk_cluster_map, f)
        with open(path / "cluster_chunk_map.json", "w") as f:
            json.dump({str(k): v for k, v in semantic_mask.cluster_chunk_map.items()}, f)
        with open(path / "chunk_labels.json", "w") as f:
            json.dump(semantic_mask.chunk_labels, f)

        logger.info(f"All artifacts saved to {path}")
