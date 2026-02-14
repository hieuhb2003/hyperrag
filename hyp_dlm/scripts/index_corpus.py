#!/usr/bin/env python3
"""
CLI: Run full indexing pipeline.

Supports:
  - Factory-based chunker selection (similarity / llm_anchor)
  - Parallel chunking and NER extraction
  - Incremental updates with SHA-256 content hashing

Usage:
    python scripts/index_corpus.py --input_dir data/raw --output_dir data/indexed --config config/default.yaml
    python scripts/index_corpus.py --input_dir data/raw --output_dir data/indexed --full-rebuild
"""

import argparse
import hashlib
import json
import os
import pickle
import sys
from collections import Counter
from pathlib import Path

import yaml
import numpy as np
from scipy import sparse

# Add parent to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.logger import get_logger
from src.utils.embedding import EmbeddingModel
from src.indexing.semantic_chunker import SemanticChunker, AnchorChunker, create_chunker
from src.indexing.ner_extractor import NERExtractor, deduplicate_entities
from src.indexing.entity_linker import EntityLinker
from src.indexing.hypergraph_builder import HypergraphBuilder
from src.indexing.masking_strategy import create_masking_strategy
from src.indexing.embedding_store import EmbeddingStore
from src.indexing.bipartite_storage import BipartiteStorage

logger = get_logger(__name__)


# ══════════════════════════════════════════════════════════
# Document loading
# ══════════════════════════════════════════════════════════

def load_documents(input_dir: str) -> list[dict[str, str]]:
    """Load documents from a directory. Supports .txt, .md files."""
    documents = []
    input_path = Path(input_dir)

    if not input_path.exists():
        logger.warning(f"Input directory does not exist: {input_dir}")
        return documents

    for ext in ["*.txt", "*.md"]:
        for fpath in sorted(input_path.glob(ext)):
            text = fpath.read_text(encoding="utf-8", errors="ignore")
            if text.strip():
                documents.append({
                    "id": fpath.stem,
                    "text": text,
                    "path": str(fpath),
                })

    logger.step("LoadDocuments", f"Loaded {len(documents)} documents from {input_dir}")
    for doc in documents:
        logger.debug(f"  {doc['id']}: {len(doc['text'])} chars")

    return documents


# ══════════════════════════════════════════════════════════
# JSON / JSONL document loading
# ══════════════════════════════════════════════════════════

def _validate_document_entry(entry: dict, index: int) -> dict:
    """Validate a single document entry from JSON input.

    Returns normalized document dict with keys: id, text, path, metadata.
    Raises ValueError if required fields missing or wrong type.
    """
    if not isinstance(entry, dict):
        raise ValueError(f"Document at index {index}: expected dict, got {type(entry).__name__}")
    if "id" not in entry:
        raise ValueError(f"Document at index {index}: missing required field 'id'")
    if "text" not in entry:
        raise ValueError(f"Document at index {index}: missing required field 'text'")
    if not isinstance(entry["id"], str):
        raise ValueError(f"Document at index {index}: 'id' must be string")
    if not isinstance(entry["text"], str):
        raise ValueError(f"Document at index {index}: 'text' must be string")
    if not entry["text"].strip():
        raise ValueError(f"Document at index {index} (id='{entry['id']}'): 'text' is empty")

    return {
        "id": entry["id"],
        "text": entry["text"],
        "path": entry.get("path", ""),
        "metadata": entry.get("metadata"),
    }


def _load_json(path: Path) -> list[dict]:
    """Load from a .json file containing an array of document objects."""
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, list):
        raise ValueError(f"JSON file must contain a top-level array, got {type(raw).__name__}")

    documents = []
    for i, entry in enumerate(raw):
        doc = _validate_document_entry(entry, i)
        documents.append(doc)

    logger.step("LoadDocuments", f"Loaded {len(documents)} documents from {path}")
    return documents


def _load_jsonl(path: Path) -> list[dict]:
    """Load from a .jsonl file with one document object per line."""
    documents = []
    with open(path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"JSONL parse error at line {line_num}: {e}") from e
            doc = _validate_document_entry(entry, line_num - 1)
            documents.append(doc)

    logger.step("LoadDocuments", f"Loaded {len(documents)} documents from {path} (JSONL)")
    return documents


def load_documents_json(json_path: str) -> list[dict]:
    """Load documents from a JSON or JSONL file.

    Auto-detects format by file extension (.json vs .jsonl).
    """
    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(f"JSON input file not found: {json_path}")

    if path.suffix == ".jsonl":
        return _load_jsonl(path)
    return _load_json(path)


# ══════════════════════════════════════════════════════════
# Manifest tracking (SHA-256 content hashing)
# ══════════════════════════════════════════════════════════

def compute_doc_hash(text: str) -> str:
    """Compute SHA-256 hash of document content."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_manifest(output_dir: str) -> dict:
    """Load manifest.json from index directory. Returns empty dict if not found."""
    manifest_path = Path(output_dir) / "manifest.json"
    if manifest_path.exists():
        with open(manifest_path) as f:
            return json.load(f)
    return {}


def save_manifest(output_dir: str, manifest: dict) -> None:
    """Save manifest.json to index directory."""
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    with open(path / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    logger.debug(f"Manifest saved: {len(manifest.get('documents', {}))} documents tracked")


def build_manifest(documents: list[dict[str, str]]) -> dict:
    """Build a manifest dict from a list of documents."""
    doc_hashes = {}
    for doc in documents:
        doc_hashes[doc["id"]] = {
            "hash": compute_doc_hash(doc["text"]),
            "path": doc.get("path", ""),
            "chars": len(doc["text"]),
        }
    return {"documents": doc_hashes}


# ══════════════════════════════════════════════════════════
# Index save / load
# ══════════════════════════════════════════════════════════

def save_index(output_dir: str, **artifacts) -> None:
    """Save all indexing artifacts to disk."""
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)

    # Save sparse matrices
    if "H" in artifacts:
        sparse.save_npz(path / "incidence_matrix_H.npz", artifacts["H"])
    if "S" in artifacts:
        sparse.save_npz(path / "synonym_matrix_S.npz", artifacts["S"])

    # Save embeddings
    if "entity_embs" in artifacts:
        np.save(path / "entity_embeddings.npy", artifacts["entity_embs"])
    if "hyperedge_embs" in artifacts:
        np.save(path / "hyperedge_embeddings.npy", artifacts["hyperedge_embs"])

    # Save metadata (pickle)
    metadata = {}
    for key in ["entity_index", "hyperedge_index", "chunks", "global_entities"]:
        if key in artifacts:
            metadata[key] = artifacts[key]
    with open(path / "metadata.pkl", "wb") as f:
        pickle.dump(metadata, f)

    # Save masking strategy
    if "masking_strategy" in artifacts:
        with open(path / "masking_strategy.pkl", "wb") as f:
            pickle.dump(artifacts["masking_strategy"], f)

    # Save bipartite graph
    if "bipartite" in artifacts:
        artifacts["bipartite"].save(output_dir)

    logger.step("SaveIndex", f"All artifacts saved to {output_dir}")
    logger.debug(f"  Files: {[f.name for f in path.iterdir()]}")


def load_existing_index(output_dir: str) -> dict:
    """Load all existing index artifacts from disk.

    Returns dict with keys: H, S, entity_embs, hyperedge_embs, metadata,
    masking_strategy, bipartite. Missing artifacts are None.
    """
    path = Path(output_dir)
    result = {}

    # Sparse matrices
    h_path = path / "incidence_matrix_H.npz"
    result["H"] = sparse.load_npz(h_path) if h_path.exists() else None

    s_path = path / "synonym_matrix_S.npz"
    result["S"] = sparse.load_npz(s_path) if s_path.exists() else None

    # Embeddings
    ee_path = path / "entity_embeddings.npy"
    result["entity_embs"] = np.load(ee_path) if ee_path.exists() else None

    he_path = path / "hyperedge_embeddings.npy"
    result["hyperedge_embs"] = np.load(he_path) if he_path.exists() else None

    # Metadata
    meta_path = path / "metadata.pkl"
    if meta_path.exists():
        with open(meta_path, "rb") as f:
            result["metadata"] = pickle.load(f)
    else:
        result["metadata"] = None

    # Masking strategy
    ms_path = path / "masking_strategy.pkl"
    if ms_path.exists():
        with open(ms_path, "rb") as f:
            result["masking_strategy"] = pickle.load(f)
    else:
        result["masking_strategy"] = None

    # Bipartite
    bp_path = path / "bipartite_graph.pkl"
    if bp_path.exists():
        bipartite = BipartiteStorage()
        bipartite.load(output_dir)
        result["bipartite"] = bipartite
    else:
        result["bipartite"] = None

    logger.debug(f"Loaded existing index from {output_dir}: {[k for k, v in result.items() if v is not None]}")
    return result


# ══════════════════════════════════════════════════════════
# Full indexing pipeline
# ══════════════════════════════════════════════════════════

def index_corpus(input_dir: str, output_dir: str, config: dict,
                 documents: list[dict] | None = None) -> None:
    """Full offline indexing pipeline.

    Args:
        input_dir: Directory with raw documents (ignored if documents provided)
        output_dir: Directory for indexed artifacts
        config: Full configuration dict
        documents: Pre-loaded documents (from JSON input). If None, loads from input_dir.
    """
    logger.step("IndexCorpus", "=" * 50)
    logger.step("IndexCorpus", "Starting full indexing pipeline")
    logger.step("IndexCorpus", "=" * 50)

    indexing_config = config.get("indexing", {})

    # Shared embedding model
    embedder = EmbeddingModel(
        model_name=config["embedding"]["model"],
        batch_size=config["embedding"]["batch_size"],
        normalize=config["embedding"]["normalize"],
    )

    # Step 1: Load documents
    if documents is None:
        documents = load_documents(input_dir)
    if not documents:
        logger.warning("No documents found. Exiting.")
        return

    # Debug: Step 1
    if logger.is_debug:
        logger.start_timer("debug_step01")
        logger.debug_sample("Loaded Documents", [
            f"id={d['id']}, chars={len(d['text'])}" for d in documents
        ])
        char_counts = [len(d["text"]) for d in documents]
        logger.debug_distribution("Document Sizes (chars)", {
            "< 1K": sum(1 for c in char_counts if c < 1000),
            "1K-5K": sum(1 for c in char_counts if 1000 <= c < 5000),
            "5K-10K": sum(1 for c in char_counts if 5000 <= c < 10000),
            "10K+": sum(1 for c in char_counts if c >= 10000),
        })
        elapsed = logger.stop_timer("debug_step01")
        logger.debug_checkpoint("step01_documents", {
            "doc_count": len(documents),
            "doc_ids": [d["id"] for d in documents],
        }, step_time=elapsed)

    # Step 2: Chunking (factory-based)
    chunker = create_chunker(config["chunking"])

    # Set embedder if chunker uses it (SemanticChunker)
    if hasattr(chunker, "set_embedder"):
        chunker.set_embedder(embedder)

    # Set LLM if chunker uses it (AnchorChunker)
    if hasattr(chunker, "set_llm"):
        from src.utils.llm_client import LLMClient
        llm_client = LLMClient(
            model=config["chunking"].get("llm_model", "gpt-4o-mini"),
            temperature=config["chunking"].get("llm_temperature", 0.0),
            max_tokens=config["chunking"].get("llm_max_tokens", 256),
        )
        chunker.set_llm(llm_client)

    # Dispatch: parallel or sequential
    if indexing_config.get("parallel_chunking", False):
        max_workers = indexing_config.get("max_workers", 4)
        chunks = chunker.chunk_documents_parallel(documents, max_workers=max_workers)
    else:
        chunks = chunker.chunk_documents(documents)
    # Propagate document metadata to chunks (from JSON input)
    doc_metadata = {d["id"]: d.get("metadata") for d in documents if d.get("metadata")}
    if doc_metadata:
        for chunk in chunks:
            chunk.metadata = doc_metadata.get(chunk.doc_id)

    logger.step("IndexCorpus", f"Step 2 complete: {len(chunks)} chunks (= hyperedges)")

    # Debug: Step 2
    if logger.is_debug:
        logger.start_timer("debug_step02")
        logger.debug_sample("Chunks", [
            f"doc={c.doc_id}, chunk={c.chunk_id}, tokens={c.token_count}, text='{c.text[:80]}...'"
            for c in chunks
        ])
        token_counts = [c.token_count for c in chunks]
        chunks_per_doc = Counter(c.doc_id for c in chunks)
        logger.debug_distribution("Chunks per Document", dict(chunks_per_doc))
        elapsed = logger.stop_timer("debug_step02")
        logger.debug_checkpoint("step02_chunking", {
            "chunk_count": len(chunks),
            "token_stats": {"min": min(token_counts), "max": max(token_counts),
                            "mean": round(sum(token_counts) / len(token_counts), 1)},
            "chunks_per_doc": dict(chunks_per_doc),
        }, step_time=elapsed)

    # Step 3: NER extraction
    ner = NERExtractor(config["ner"])
    if indexing_config.get("parallel_ner", False):
        batch_size = indexing_config.get("ner_batch_size", 50)
        ner.extract_from_chunks_batch(chunks, batch_size=batch_size)
    else:
        ner.extract_from_chunks(chunks)
    logger.step("IndexCorpus", "Step 3 complete: NER extraction done")

    # Debug: Step 3
    if logger.is_debug:
        logger.start_timer("debug_step03")
        for chunk in chunks[:logger._debug_max_samples]:
            entity_strs = [f"{e.name} ({e.entity_type})" for e in chunk.entities]
            logger.debug_sample(f"Entities in '{chunk.doc_id}:{chunk.chunk_id}'", entity_strs)
        type_counts = Counter()
        total_mentions = 0
        for chunk in chunks:
            for e in chunk.entities:
                type_counts[e.entity_type] += 1
                total_mentions += 1
        logger.debug_distribution("Entity Type Distribution", dict(type_counts))
        elapsed = logger.stop_timer("debug_step03")
        logger.debug_checkpoint("step03_ner", {
            "total_entity_mentions": total_mentions,
            "type_distribution": dict(type_counts),
            "avg_entities_per_chunk": round(total_mentions / len(chunks), 1) if chunks else 0,
        }, step_time=elapsed)

    # Step 4: Global entity deduplication
    global_entities = deduplicate_entities(chunks)
    logger.step("IndexCorpus", f"Step 4 complete: {len(global_entities)} global entities")

    # Debug: Step 4
    if logger.is_debug:
        total_mentions = sum(len(c.entities) for c in chunks)
        dedup_ratio = 1 - len(global_entities) / total_mentions if total_mentions > 0 else 0
        logger.debug_sample("Global Entities", [e.name for e in global_entities])
        logger.debug_checkpoint("step04_dedup", {
            "total_mentions": total_mentions,
            "unique_entities": len(global_entities),
            "dedup_ratio": round(dedup_ratio, 4),
        })

    # Step 5: Compute embeddings (needed for entity linking + masking)
    entity_names = [e.name for e in global_entities]
    hyperedge_texts = [c.text for c in chunks]

    entity_embs = embedder.encode(entity_names, show_progress=True)
    hyperedge_embs = embedder.encode(hyperedge_texts, show_progress=True)
    logger.step(
        "IndexCorpus",
        f"Step 5 complete: embeddings computed",
        entity_shape=entity_embs.shape,
        hyperedge_shape=hyperedge_embs.shape,
    )

    # Debug: Step 5
    if logger.is_debug:
        logger.debug_matrix_detail("Entity Embeddings", entity_embs)
        logger.debug_matrix_detail("Hyperedge Embeddings", hyperedge_embs)
        logger.debug_checkpoint("step05_embeddings", {
            "entity_emb_shape": list(entity_embs.shape),
            "hyperedge_emb_shape": list(hyperedge_embs.shape),
        })

    # Step 6: Entity linking -> synonym matrix S
    linker = EntityLinker(config["entity_linking"], embedder=embedder)
    S = linker.build_synonym_matrix(global_entities, entity_embeddings=entity_embs)
    logger.step("IndexCorpus", f"Step 6 complete: synonym matrix S ({S.shape}, nnz={S.nnz})")

    # Debug: Step 6
    if logger.is_debug:
        logger.debug_matrix_detail("Synonym Matrix S", S)
        # Sample synonym pairs
        S_coo = S.tocoo()
        synonym_pairs = []
        for i, j, v in zip(S_coo.row, S_coo.col, S_coo.data):
            if i < j:
                synonym_pairs.append((global_entities[i].name, global_entities[j].name, float(v)))
        synonym_pairs.sort(key=lambda x: -x[2])
        logger.debug_sample("Synonym Pairs (by score)", [
            f"{a} <-> {b} (score={s:.3f})" for a, b, s in synonym_pairs
        ])
        logger.debug_checkpoint("step06_linking", {
            "S_shape": list(S.shape), "S_nnz": int(S.nnz),
            "synonym_pair_count": len(synonym_pairs),
            "top_pairs": [{"a": a, "b": b, "score": round(s, 4)} for a, b, s in synonym_pairs[:10]],
        })

    # Step 7: Build incidence matrix H
    builder = HypergraphBuilder()
    H, entity_index, hyperedge_index = builder.build(chunks, global_entities)
    logger.step("IndexCorpus", f"Step 7 complete: incidence matrix H {H.shape}")

    # Debug: Step 7
    if logger.is_debug:
        logger.debug_matrix_detail("Incidence Matrix H", H)
        row_nnz = np.diff(H.tocsr().indptr)
        col_nnz = np.diff(H.tocsc().indptr)
        logger.debug_checkpoint("step07_hypergraph", {
            "H_shape": list(H.shape), "H_nnz": int(H.nnz),
            "isolated_entities": int(np.sum(row_nnz == 0)),
            "empty_hyperedges": int(np.sum(col_nnz == 0)),
        })

    # Step 8: Fit semantic masking strategy
    masking = create_masking_strategy(config)
    masking.fit(hyperedge_embs)
    logger.step("IndexCorpus", f"Step 8 complete: masking strategy '{config['masking']['strategy']}' fitted")

    # Debug: Step 8
    if logger.is_debug:
        logger.debug_checkpoint("step08_masking", {
            "strategy": config["masking"]["strategy"],
        })

    # Step 9: Build bipartite graph
    bipartite = BipartiteStorage()
    bipartite.build_from_incidence(H, global_entities, chunks)
    logger.step("IndexCorpus", "Step 9 complete: bipartite graph built")

    # Debug: Step 9
    if logger.is_debug:
        logger.debug_checkpoint("step09_bipartite", {
            "entity_count": len(global_entities),
            "hyperedge_count": len(chunks),
        })

    # Step 10: Save everything
    save_index(
        output_dir,
        H=H,
        S=S,
        entity_embs=entity_embs,
        hyperedge_embs=hyperedge_embs,
        masking_strategy=masking,
        entity_index=entity_index,
        hyperedge_index=hyperedge_index,
        chunks=chunks,
        global_entities=global_entities,
        bipartite=bipartite,
    )

    # Save manifest
    manifest = build_manifest(documents)
    save_manifest(output_dir, manifest)

    # Debug: Step 10
    if logger.is_debug:
        out_path = Path(output_dir)
        files_info = {f.name: f.stat().st_size for f in out_path.iterdir() if f.is_file()}
        logger.debug_checkpoint("step10_save", {
            "files": files_info,
            "total_bytes": sum(files_info.values()),
        })

    logger.step("IndexCorpus", "=" * 50)
    logger.step("IndexCorpus", "INDEXING PIPELINE COMPLETE")
    logger.step("IndexCorpus", "=" * 50)

    # Debug report
    if logger.is_debug:
        logger.debug_report(extra={
            "pipeline": "indexing",
            "documents": len(documents),
            "chunks": len(chunks),
            "entities": len(global_entities),
            "H_shape": list(H.shape),
            "S_nnz": int(S.nnz),
        })

    # Final summary
    logger.summary_table(
        "Indexing Summary",
        [
            {"Metric": "Documents", "Value": len(documents)},
            {"Metric": "Chunks (Hyperedges)", "Value": len(chunks)},
            {"Metric": "Global Entities", "Value": len(global_entities)},
            {"Metric": "H shape", "Value": str(H.shape)},
            {"Metric": "H nnz", "Value": H.nnz},
            {"Metric": "S nnz (synonym links)", "Value": S.nnz},
            {"Metric": "Masking strategy", "Value": config["masking"]["strategy"]},
            {"Metric": "Output dir", "Value": output_dir},
        ],
    )


# ══════════════════════════════════════════════════════════
# Incremental indexing pipeline
# ══════════════════════════════════════════════════════════

def index_corpus_incremental(input_dir: str, output_dir: str, config: dict,
                             documents: list[dict] | None = None) -> None:
    """Incremental indexing pipeline.

    Compares document hashes with manifest to identify new/changed/deleted docs.
    Only processes new documents incrementally. Changed/deleted docs trigger
    a warning and recommend full rebuild.
    """
    logger.step("IndexCorpus", "=" * 50)
    logger.step("IndexCorpus", "Starting INCREMENTAL indexing pipeline")
    logger.step("IndexCorpus", "=" * 50)

    # Load manifest
    manifest = load_manifest(output_dir)
    if not manifest or "documents" not in manifest:
        logger.step("IndexCorpus", "No existing manifest found, falling back to full rebuild")
        index_corpus(input_dir, output_dir, config, documents=documents)
        return

    # Load existing index
    existing = load_existing_index(output_dir)
    if existing["H"] is None or existing["metadata"] is None:
        logger.step("IndexCorpus", "Incomplete existing index, falling back to full rebuild")
        index_corpus(input_dir, output_dir, config, documents=documents)
        return

    # Load current documents
    if documents is None:
        documents = load_documents(input_dir)
    if not documents:
        logger.warning("No documents found. Exiting.")
        return

    # Classify documents
    old_doc_hashes = manifest["documents"]
    new_docs = []
    changed_docs = []
    unchanged_docs = []

    current_ids = set()
    for doc in documents:
        current_ids.add(doc["id"])
        doc_hash = compute_doc_hash(doc["text"])
        if doc["id"] not in old_doc_hashes:
            new_docs.append(doc)
        elif old_doc_hashes[doc["id"]]["hash"] != doc_hash:
            changed_docs.append(doc)
        else:
            unchanged_docs.append(doc)

    deleted_ids = set(old_doc_hashes.keys()) - current_ids

    logger.step(
        "IndexCorpus",
        f"Document status: {len(new_docs)} new, {len(changed_docs)} changed, "
        f"{len(deleted_ids)} deleted, {len(unchanged_docs)} unchanged",
    )

    # Handle changed/deleted documents
    if changed_docs or deleted_ids:
        if deleted_ids:
            logger.warning(
                f"Deleted documents detected: {deleted_ids}. "
                "Recommend full rebuild with --full-rebuild."
            )
        if changed_docs:
            logger.warning(
                f"Changed documents detected: {[d['id'] for d in changed_docs]}. "
                "Recommend full rebuild with --full-rebuild."
            )
        logger.step("IndexCorpus", "Falling back to full rebuild due to changed/deleted documents")
        index_corpus(input_dir, output_dir, config, documents=documents)
        return

    # No new documents
    if not new_docs:
        logger.step("IndexCorpus", "No new or changed documents. Index is up to date.")
        return

    # ── Incremental path: only new documents ──

    indexing_config = config.get("indexing", {})

    # Shared embedding model
    embedder = EmbeddingModel(
        model_name=config["embedding"]["model"],
        batch_size=config["embedding"]["batch_size"],
        normalize=config["embedding"]["normalize"],
    )

    # Extract existing data
    old_metadata = existing["metadata"]
    old_chunks = old_metadata["chunks"]
    old_global_entities = old_metadata["global_entities"]
    old_entity_index = old_metadata["entity_index"]
    old_H = existing["H"]
    old_S = existing["S"]
    old_entity_embs = existing["entity_embs"]
    old_hyperedge_embs = existing["hyperedge_embs"]
    old_masking = existing["masking_strategy"]
    old_bipartite = existing["bipartite"]
    old_M = old_H.shape[1]

    logger.step(
        "IndexCorpus",
        f"Existing index: {len(old_chunks)} chunks, {len(old_global_entities)} entities",
    )

    # Step 1: Chunk new documents
    chunker = create_chunker(config["chunking"])
    if hasattr(chunker, "set_embedder"):
        chunker.set_embedder(embedder)
    if hasattr(chunker, "set_llm"):
        from src.utils.llm_client import LLMClient
        llm_client = LLMClient(
            model=config["chunking"].get("llm_model", "gpt-4o-mini"),
            temperature=config["chunking"].get("llm_temperature", 0.0),
            max_tokens=config["chunking"].get("llm_max_tokens", 256),
        )
        chunker.set_llm(llm_client)

    if indexing_config.get("parallel_chunking", False):
        max_workers = indexing_config.get("max_workers", 4)
        new_chunks = chunker.chunk_documents_parallel(new_docs, max_workers=max_workers)
    else:
        new_chunks = chunker.chunk_documents(new_docs)
    logger.step("IndexCorpus", f"Incremental: {len(new_chunks)} new chunks from {len(new_docs)} docs")

    # Step 2: NER on new chunks
    ner = NERExtractor(config["ner"])
    if indexing_config.get("parallel_ner", False):
        batch_size = indexing_config.get("ner_batch_size", 50)
        ner.extract_from_chunks_batch(new_chunks, batch_size=batch_size)
    else:
        ner.extract_from_chunks(new_chunks)

    # Step 3: Deduplicate entities, find truly new ones
    new_chunk_entities = deduplicate_entities(new_chunks)
    old_entity_names = {e.name for e in old_global_entities}
    truly_new_entities = [e for e in new_chunk_entities if e.name not in old_entity_names]
    combined_entities = list(old_global_entities) + truly_new_entities

    logger.step(
        "IndexCorpus",
        f"Incremental: {len(truly_new_entities)} truly new entities, "
        f"{len(combined_entities)} total",
    )

    # Step 4: Compute embeddings for new entities and chunks
    if truly_new_entities:
        new_entity_names = [e.name for e in truly_new_entities]
        new_entity_embs = embedder.encode(new_entity_names, show_progress=True)
        entity_embs = np.vstack([old_entity_embs, new_entity_embs])
    else:
        new_entity_embs = np.empty((0, old_entity_embs.shape[1]))
        entity_embs = old_entity_embs

    new_hyperedge_texts = [c.text for c in new_chunks]
    new_hyperedge_embs = embedder.encode(new_hyperedge_texts, show_progress=True)
    hyperedge_embs = np.vstack([old_hyperedge_embs, new_hyperedge_embs])

    # Step 5: Expand synonym matrix S
    linker = EntityLinker(config["entity_linking"], embedder=embedder)
    if truly_new_entities:
        S = linker.expand_synonym_matrix(
            old_S, old_global_entities, truly_new_entities,
            old_entity_embs, new_entity_embs,
        )
    else:
        S = old_S

    # Step 6: Expand incidence matrix H
    builder = HypergraphBuilder()
    H, entity_index, hyperedge_index = builder.expand(
        old_H, old_entity_index, old_M,
        new_chunks, old_global_entities, combined_entities,
    )

    # Step 7: Update masking strategy
    old_masking.add_hyperedges(new_hyperedge_embs)
    masking = old_masking

    # Step 8: Update bipartite graph
    old_bipartite.add_document(
        new_chunks, truly_new_entities,
        H[:, old_M:],  # new columns only
        combined_entities,
    )
    bipartite = old_bipartite

    # Combine chunks
    all_chunks = list(old_chunks) + list(new_chunks)

    # Step 9: Save everything
    save_index(
        output_dir,
        H=H,
        S=S,
        entity_embs=entity_embs,
        hyperedge_embs=hyperedge_embs,
        masking_strategy=masking,
        entity_index=entity_index,
        hyperedge_index=hyperedge_index,
        chunks=all_chunks,
        global_entities=combined_entities,
        bipartite=bipartite,
    )

    # Update manifest with all documents
    manifest = build_manifest(documents)
    save_manifest(output_dir, manifest)

    logger.step("IndexCorpus", "=" * 50)
    logger.step("IndexCorpus", "INCREMENTAL INDEXING COMPLETE")
    logger.step("IndexCorpus", "=" * 50)

    logger.summary_table(
        "Incremental Indexing Summary",
        [
            {"Metric": "New documents", "Value": len(new_docs)},
            {"Metric": "New chunks", "Value": len(new_chunks)},
            {"Metric": "New entities", "Value": len(truly_new_entities)},
            {"Metric": "Total chunks", "Value": len(all_chunks)},
            {"Metric": "Total entities", "Value": len(combined_entities)},
            {"Metric": "H shape", "Value": str(H.shape)},
            {"Metric": "H nnz", "Value": H.nnz},
            {"Metric": "S nnz", "Value": S.nnz},
            {"Metric": "Output dir", "Value": output_dir},
        ],
    )


# ══════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="HyP-DLM: Index a corpus")

    # Input source: mutually exclusive
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input_dir", help="Directory with raw documents (.txt, .md)")
    input_group.add_argument("--input_json", help="JSON or JSONL file with document objects")

    parser.add_argument("--output_dir", required=True, help="Directory for indexed artifacts")
    parser.add_argument("--config", default="config/default.yaml", help="Config YAML file")
    parser.add_argument(
        "--full-rebuild", action="store_true",
        help="Force full rebuild even if incremental is enabled",
    )
    parser.add_argument("--debug", action="store_true", help="Enable debug mode with detailed output")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # Set log level from config
    log_level = config.get("logging", {}).get("level", "DEBUG")
    logger.setLevel(log_level)

    # Debug mode
    debug_config = config.get("debug", {})
    if args.debug or debug_config.get("enabled", False):
        debug_output = debug_config.get("output_dir", "data/debug")
        max_samples = debug_config.get("max_sample_items", 3)
        logger.enable_debug(debug_output, max_samples=max_samples)

    # Determine input source
    documents = None
    if args.input_json:
        documents = load_documents_json(args.input_json)

    incremental_enabled = config.get("indexing", {}).get("incremental", False)

    if incremental_enabled and not args.full_rebuild:
        index_corpus_incremental(
            args.input_dir or "", args.output_dir, config, documents=documents,
        )
    else:
        index_corpus(
            args.input_dir or "", args.output_dir, config, documents=documents,
        )


if __name__ == "__main__":
    main()
