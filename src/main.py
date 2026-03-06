"""CLI entry point for HyP-DLM pipeline."""

from __future__ import annotations

import argparse
import sys


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hypdlm",
        description="HyP-DLM: Hypergraph Propagation with Dynamic Logic Modulation",
    )
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML")
    parser.add_argument("--device", help="Override device (cuda/cpu/mps)")
    parser.add_argument("--embedding-model", help="Override embedding model name")
    parser.add_argument("--ner-backend", choices=["gliner", "spacy"], help="NER backend")
    parser.add_argument("--max-workers", type=int, help="Override max workers")
    parser.add_argument("--dataset-name", help="Dataset name for metrics grouping")
    parser.add_argument("--log-level", help="Override log level")

    sub = parser.add_subparsers(dest="command", required=True)

    # Index command
    idx = sub.add_parser("index", help="Build hypergraph index from documents")
    idx.add_argument("--data", required=True, help="Path to documents JSON")

    # Retrieve command
    ret = sub.add_parser("retrieve", help="Retrieve passages for queries")
    ret.add_argument("--queries", required=True, help="Path to queries JSON")
    ret.add_argument("--storage", help="Override storage path")

    # Generate command
    gen = sub.add_parser("generate", help="Generate answers for queries")
    gen.add_argument("--queries", required=True, help="Path to queries JSON")
    gen.add_argument("--storage", help="Override storage path")

    # Evaluate command
    eva = sub.add_parser("evaluate", help="Evaluate outputs against gold answers")
    eva.add_argument("--output", help="Path to output directory")

    # Run-all command (index + retrieve + generate)
    run = sub.add_parser("run", help="Run full pipeline: index → retrieve → generate")
    run.add_argument("--data", required=True, help="Path to documents JSON")
    run.add_argument("--queries", required=True, help="Path to queries JSON")

    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    # Import here to avoid circular + keep CLI fast
    from src.config import Config
    import importlib
    _logging_setup = importlib.import_module("src.logging-setup")

    # Build config from YAML + CLI overrides
    overrides = {}
    for key in ("device", "embedding_model", "ner_backend", "max_workers", "dataset_name", "log_level"):
        cli_key = key.replace("_", "-") if key != "log_level" else key
        val = getattr(args, key.replace("-", "_"), None) if hasattr(args, key) else None
        if val is None:
            val = getattr(args, key, None)
        if val is not None:
            overrides[key] = val

    config = Config.from_args(args.config, **overrides)
    _logging_setup.setup_logging(config)

    logger = _logging_setup.get_logger("main")
    logger.info(f"Command: {args.command}")
    logger.info(f"Config: device={config.device}, embedding={config.embedding_model}, "
                f"ner={config.ner_backend}, workers={config.max_workers}")

    if args.command == "index":
        _run_index(config, args.data)
    elif args.command == "retrieve":
        _run_retrieve(config, args.queries, getattr(args, "storage", None))
    elif args.command == "generate":
        _run_generate(config, args.queries, getattr(args, "storage", None))
    elif args.command == "evaluate":
        _run_evaluate(config, getattr(args, "output", None))
    elif args.command == "run":
        _run_index(config, args.data)
        _run_retrieve(config, args.queries, None)
        _run_generate(config, args.queries, None)


def _run_index(config, data_path: str):
    """Run indexing pipeline."""
    import importlib
    pipeline = importlib.import_module("src.indexing.indexing-pipeline")
    pipeline.run_indexing(config, data_path)


def _run_retrieve(config, queries_path: str, storage_path=None):
    """Run retrieval pipeline."""
    import importlib
    pipeline = importlib.import_module("src.retrieval.retrieval-pipeline")
    pipeline.run_retrieval(config, queries_path, storage_path)


def _run_generate(config, queries_path: str, storage_path=None):
    """Run generation pipeline."""
    import importlib
    pipeline = importlib.import_module("src.generation.generation-pipeline")
    pipeline.run_generation(config, queries_path, storage_path)


def _run_evaluate(config, output_path=None):
    """Run evaluation."""
    import importlib
    evaluator = importlib.import_module("src.evaluation.metrics-aggregator")
    evaluator.run_evaluation(config, output_path)


if __name__ == "__main__":
    main()
