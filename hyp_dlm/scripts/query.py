#!/usr/bin/env python3
"""
CLI: Run a single query through the HyP-DLM pipeline.

Usage:
    python scripts/query.py --index_dir data/indexed --query "Who founded Microsoft?" --config config/default.yaml
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

import yaml
import numpy as np
from rich.panel import Panel
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.logger import get_logger, console
from src.utils.embedding import EmbeddingModel
from src.utils.llm_client import LLMClient
from src.indexing.ner_extractor import NERExtractor
from src.retrieval.router import FamiliarityRouter
from src.retrieval.query_decomposer import QueryDecomposer
from src.retrieval.propagation import DAGPropagation
from src.retrieval.passage_ranker import PassageRanker
from src.retrieval.chunk_retriever import ChunkRetriever
from src.generation.generator import HybridRAGGenerator

logger = get_logger(__name__)


def load_index(index_dir: str) -> dict:
    """Load all indexed artifacts."""
    path = Path(index_dir)
    logger.start_timer("load_index")

    artifacts = {}

    # Sparse matrices
    artifacts["H"] = sparse.load_npz(path / "incidence_matrix_H.npz")
    artifacts["S"] = sparse.load_npz(path / "synonym_matrix_S.npz")

    # Embeddings
    artifacts["entity_embs"] = np.load(path / "entity_embeddings.npy")
    artifacts["hyperedge_embs"] = np.load(path / "hyperedge_embeddings.npy")

    # Metadata
    with open(path / "metadata.pkl", "rb") as f:
        metadata = pickle.load(f)
    artifacts.update(metadata)

    # Masking strategy
    with open(path / "masking_strategy.pkl", "rb") as f:
        artifacts["masking_strategy"] = pickle.load(f)

    elapsed = logger.stop_timer("load_index")
    logger.step(
        "LoadIndex",
        f"Loaded index from {index_dir}",
        H=artifacts["H"].shape,
        entities=len(artifacts.get("global_entities", [])),
        time=elapsed,
    )

    return artifacts


def run_query(query: str, index_dir: str, config: dict) -> dict:
    """Run a single query through the full pipeline."""

    logger.step("Query", "=" * 60)
    logger.step("Query", f"Query: '{query}'")
    logger.step("Query", "=" * 60)

    # Load index
    artifacts = load_index(index_dir)
    H = artifacts["H"]
    S = artifacts["S"]
    entity_embs = artifacts["entity_embs"]
    hyperedge_embs = artifacts["hyperedge_embs"]
    chunks = artifacts.get("chunks", [])
    global_entities = artifacts.get("global_entities", [])
    masking_strategy = artifacts["masking_strategy"]

    # Initialize components
    embedder = EmbeddingModel(
        model_name=config["embedding"]["model"],
        batch_size=config["embedding"]["batch_size"],
        normalize=config["embedding"]["normalize"],
    )
    llm = LLMClient(
        model=config.get("generation", {}).get("llm_model", "gpt-4o-mini"),
        temperature=config.get("generation", {}).get("temperature", 0.0),
    )

    # Step 1: Encode query
    query_emb = embedder.encode_single(query)
    logger.step("Query", "Step 1: Query encoded")

    # Debug: Step 1
    if logger.is_debug:
        emb_norm = float(np.linalg.norm(query_emb))
        console.print(Panel(
            f"Query: '{query}'\nEmbedding shape: {query_emb.shape}\nNorm (L2): {emb_norm:.4f}",
            title="[DEBUG] Step 1: Query Encoding", border_style="yellow",
        ))
        logger.debug_checkpoint("query_step01_encode", {
            "query": query, "emb_shape": list(query_emb.shape), "emb_norm": round(emb_norm, 4),
        })

    # Step 2: Route
    router = FamiliarityRouter(config["router"])
    route_result = router.route(query_emb, hyperedge_embs)
    route = route_result["route"]
    logger.step("Query", f"Step 2: Routed to '{route}'")

    # Debug: Step 2
    if logger.is_debug:
        lines = [f"Route: {route}"]
        for k, v in route_result.items():
            if k not in ("route", "scores"):
                lines.append(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
        if "scores" in route_result and hasattr(route_result["scores"], '__len__'):
            top_scores = sorted(route_result["scores"], reverse=True)[:5]
            lines.append(f"  Top-5 scores: {[round(s, 4) for s in top_scores]}")
        console.print(Panel("\n".join(lines), title="[DEBUG] Step 2: Routing", border_style="yellow"))
        logger.debug_checkpoint("query_step02_route", {
            "route": route,
            "details": {k: round(v, 4) if isinstance(v, float) else v
                        for k, v in route_result.items() if k != "scores"},
        })

    # Step 3: Dense retrieval (always computed)
    chunk_retriever = ChunkRetriever(config["chunk_retrieval"])
    chunk_passages = chunk_retriever.retrieve(query_emb, hyperedge_embs)

    # Debug: Step 3
    if logger.is_debug:
        logger.debug_sample("Dense Retrieved Chunks", [
            f"idx={p.get('hyperedge_id', '?')}, score={p.get('score', 0):.4f}"
            for p in chunk_passages
        ])
        logger.debug_checkpoint("query_step03_dense", {
            "chunk_count": len(chunk_passages),
            "top_scores": [round(p.get("score", 0), 4) for p in chunk_passages[:5]],
        })

    graph_passages = []

    if route in ("graph", "hybrid"):
        # Step 4: Query decomposition
        decomposer = QueryDecomposer(config["decomposition"], llm_client=llm)
        dag = decomposer.decompose(query)
        logger.step("Query", f"Step 4: Decomposed into {len(dag.nodes)} sub-questions")

        # Debug: Step 4
        if logger.is_debug:
            dag_lines = []
            for node in dag.nodes:
                deps = ", ".join(node.depends_on) if node.depends_on else "none"
                dag_lines.append(f"  [{node.id}] {node.question}  (depends: {deps})")
            topo_order = [n.id for n in dag.topological_order()]
            dag_lines.append(f"\n  Topological order: {' -> '.join(topo_order)}")
            console.print(Panel("\n".join(dag_lines),
                title=f"[DEBUG] Step 4: Query DAG ({len(dag.nodes)} sub-questions)", border_style="yellow"))
            logger.debug_checkpoint("query_step04_decompose", {
                "node_count": len(dag.nodes),
                "dag_structure": [{"id": n.id, "question": n.question, "depends_on": n.depends_on} for n in dag.nodes],
                "topological_order": topo_order,
            })

        # Step 5: DAG-guided propagation
        ner = NERExtractor(config["ner"])
        propagator = DAGPropagation(config["propagation"])
        prop_result = propagator.propagate(
            dag=dag,
            H=H,
            S=S,
            masking_strategy=masking_strategy,
            entity_embeddings=entity_embs,
            hyperedge_embeddings=hyperedge_embs,
            embedder=embedder,
            ner_extractor=ner,
            llm_client=llm,
            chunks=chunks,
        )
        logger.step("Query", "Step 5: Propagation complete")

        # Debug: Step 5
        if logger.is_debug:
            entity_scores_dict = prop_result.get("entity_scores", {})
            top_entities = sorted(entity_scores_dict.items(), key=lambda x: -x[1])[:10]
            entity_lines = []
            for idx, score in top_entities:
                name = global_entities[idx].name if idx < len(global_entities) else f"entity_{idx}"
                entity_lines.append(f"  [{idx}] {name}: {score:.4f}")
            console.print(Panel(
                f"Activated entities: {len(entity_scores_dict)}\nTop-10:\n" + "\n".join(entity_lines),
                title="[DEBUG] Step 5: Propagation", border_style="magenta"))
            logger.debug_checkpoint("query_step05_propagation", {
                "activated_entities": len(entity_scores_dict),
                "top_entities": [{"idx": int(idx), "score": round(score, 4)} for idx, score in top_entities],
            })

        # Step 6: PPR passage ranking
        # Build entity score vector from propagation
        entity_scores = np.zeros(H.shape[0])
        for idx, score in prop_result["entity_scores"].items():
            entity_scores[idx] = score

        ranker = PassageRanker(config["ppr"])
        graph_passages = ranker.rank(
            entity_scores=entity_scores,
            H=H,
            query_embedding=query_emb,
            hyperedge_embeddings=hyperedge_embs,
        )
        logger.step("Query", f"Step 6: PPR ranked {len(graph_passages)} passages")

        # Debug: Step 6
        if logger.is_debug:
            logger.debug_sample("PPR Ranked Passages", [
                f"hyperedge={p.get('hyperedge_id', '?')}, score={p.get('score', 0):.4f}"
                for p in graph_passages
            ])
            logger.debug_checkpoint("query_step06_ranking", {
                "passage_count": len(graph_passages),
                "top_passages": [{"hyperedge_id": p.get("hyperedge_id"), "score": round(p.get("score", 0), 4)}
                                 for p in graph_passages[:10]],
            })
    else:
        logger.step("Query", "Steps 4-6: Skipped (direct route)")

    # Step 7: Generate answer
    generator = HybridRAGGenerator(config["generation"], llm_client=llm)
    result = generator.generate(
        query=query,
        graph_passages=graph_passages,
        chunk_passages=chunk_passages,
        chunks=chunks,
    )
    logger.step("Query", "Step 7: Answer generated")

    # Debug: Step 7
    if logger.is_debug:
        retrieved = result.get("retrieved_passages", [])
        context_tokens = sum(len(p.get("text", "").split()) for p in retrieved) if retrieved else 0
        console.print(Panel(
            f"Merged passages: {len(retrieved)}\nContext tokens: ~{context_tokens}\n"
            f"Answer: {result.get('answer', 'N/A')}",
            title="[DEBUG] Step 7: Generation", border_style="green"))
        logger.debug_checkpoint("query_step07_generation", {
            "merged_passage_count": len(retrieved),
            "context_tokens_estimate": context_tokens,
            "answer": result.get("answer", ""),
            "route": route,
        })

    # Debug report
    if logger.is_debug:
        logger.debug_report(extra={
            "pipeline": "retrieval", "query": query, "route": route,
            "answer": result.get("answer", ""),
            "graph_passages": len(graph_passages),
            "chunk_passages": len(chunk_passages),
        })

    # Final output
    logger.step("Query", "=" * 60)
    logger.step("Query", f"ANSWER: {result['answer']}")
    logger.step("Query", "=" * 60)

    result["route"] = route
    result["route_details"] = route_result
    return result


def main():
    parser = argparse.ArgumentParser(description="HyP-DLM: Query")
    parser.add_argument("--index_dir", required=True, help="Directory with indexed artifacts")
    parser.add_argument("--query", required=True, help="The question to answer")
    parser.add_argument("--config", default="config/default.yaml", help="Config YAML file")
    parser.add_argument("--output", default=None, help="Save result JSON to this path")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode with detailed output")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    log_level = config.get("logging", {}).get("level", "DEBUG")
    logger.setLevel(log_level)

    # Debug mode
    debug_config = config.get("debug", {})
    if args.debug or debug_config.get("enabled", False):
        debug_output = debug_config.get("output_dir", "data/debug")
        max_samples = debug_config.get("max_sample_items", 3)
        logger.enable_debug(debug_output, max_samples=max_samples)

    result = run_query(args.query, args.index_dir, config)

    if args.output:
        # Save result (strip non-serializable data)
        output = {
            "query": args.query,
            "answer": result["answer"],
            "route": result["route"],
            "route_details": result["route_details"],
            "passages_used": len(result.get("retrieved_passages", [])),
        }
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        logger.debug(f"Result saved to {args.output}")

    print(f"\nAnswer: {result['answer']}")


if __name__ == "__main__":
    main()
