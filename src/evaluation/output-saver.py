"""Save retrieval and generation results to JSON for evaluation."""

from __future__ import annotations

import importlib
import json
import os
from pathlib import Path

from src.config import Config

_logging = importlib.import_module("src.logging-setup")
logger = _logging.get_logger("output-saver")


class OutputSaver:
    """Save pipeline outputs to JSON files."""

    def __init__(self, config: Config):
        self.config = config

    def save_retrieval_results(self, results: list[dict], dataset_name: str) -> str:
        """Save top-10 retrieval results per query to JSON for recall measurement."""
        out_dir = Path(self.config.output_dir) / dataset_name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "retrieval-results.json"

        output = []
        for r in results:
            output.append({
                "query_id": r["query_id"],
                "query": r.get("query", ""),
                "top_results": [
                    {"chunk_id": cid, "score": float(score)}
                    for cid, score in r["top_results"][:10]
                ],
            })

        with open(out_path, "w") as f:
            json.dump(output, f, indent=2)

        logger.info(f"Saved retrieval results ({len(output)} queries) to {out_path}")
        return str(out_path)

    def save_generation_results(self, results: list[dict], dataset_name: str) -> str:
        """Save generated answers to JSON for evaluation."""
        out_dir = Path(self.config.output_dir) / dataset_name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "generation-results.json"

        output = []
        for r in results:
            output.append({
                "query_id": r["query_id"],
                "query": r["query"],
                "generated_answer": r["generated_answer"],
                "ground_truth": r["ground_truth"],
                "evidence_doc_ids": r["evidence_doc_ids"],
                "prompt_tokens": r.get("prompt_tokens", 0),
                "completion_tokens": r.get("completion_tokens", 0),
            })

        with open(out_path, "w") as f:
            json.dump(output, f, indent=2)

        logger.info(f"Saved generation results ({len(output)} queries) to {out_path}")
        return str(out_path)
