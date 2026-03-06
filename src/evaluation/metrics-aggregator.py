"""Evaluation metrics: Recall@K, EM, F1, token/timing aggregation."""

from __future__ import annotations

import importlib
import json
import os
import re
from pathlib import Path

import numpy as np

from src.config import Config

_logging = importlib.import_module("src.logging-setup")
logger = _logging.get_logger("evaluator")


def run_evaluation(config: Config, output_path=None):
    """Entry point for evaluation command."""
    aggregator = MetricsAggregator(config)
    report = aggregator.aggregate_metrics(config.dataset_name)

    # Print summary
    print("\n=== Evaluation Report ===")
    if "retrieval" in report:
        r = report["retrieval"]
        print(f"Retrieval Recall@{r.get('k', 10)}: {r.get('recall_at_k', 0):.4f} "
              f"({r.get('n_queries', 0)} queries)")
    if "generation" in report:
        g = report["generation"]
        print(f"Generation EM: {g.get('exact_match', 0):.4f}, "
              f"F1: {g.get('f1', 0):.4f} ({g.get('n_queries', 0)} queries)")
    print("========================\n")

    return report


class MetricsAggregator:
    """Aggregate and compute evaluation metrics."""

    def __init__(self, config: Config):
        self.config = config

    def evaluate_retrieval(self, dataset_name: str, queries_path: str = None) -> dict:
        """Compute retrieval recall@K from saved results."""
        results_path = Path(self.config.output_dir) / dataset_name / "retrieval-results.json"
        if not results_path.exists():
            logger.warning(f"No retrieval results at {results_path}")
            return {"recall_at_k": 0, "k": self.config.score_top_n_final, "n_queries": 0}

        with open(results_path) as f:
            results = json.load(f)

        # Load queries for ground truth
        _dl = importlib.import_module("src.data-loader")
        if queries_path:
            queries = _dl.load_queries(queries_path)
        else:
            queries = []

        gt_map = {q.id: q.evidence for q in queries}

        recalls = []
        for r in results:
            query_id = r["query_id"]
            gt_evidence = gt_map.get(query_id, [])
            if not gt_evidence:
                continue

            # Extract doc IDs from chunk IDs (format: "{doc_id}_chunk_{idx}")
            retrieved_doc_ids = set()
            for item in r["top_results"]:
                cid = item["chunk_id"]
                parts = cid.rsplit("_chunk_", 1)
                if parts:
                    retrieved_doc_ids.add(parts[0])

            recall = len(retrieved_doc_ids & set(gt_evidence)) / len(gt_evidence)
            recalls.append(recall)

        avg_recall = float(np.mean(recalls)) if recalls else 0
        logger.info(f"Retrieval Recall@{self.config.score_top_n_final}: {avg_recall:.4f} "
                    f"({len(recalls)} queries)")
        return {
            "recall_at_k": avg_recall,
            "k": self.config.score_top_n_final,
            "n_queries": len(recalls),
        }

    def evaluate_generation(self, dataset_name: str) -> dict:
        """Compute EM and token F1 from saved generation results."""
        results_path = Path(self.config.output_dir) / dataset_name / "generation-results.json"
        if not results_path.exists():
            logger.warning(f"No generation results at {results_path}")
            return {"exact_match": 0, "f1": 0, "n_queries": 0}

        with open(results_path) as f:
            results = json.load(f)

        em_scores, f1_scores = [], []
        for r in results:
            pred = r.get("generated_answer", "")
            gold = r.get("ground_truth", "")
            if not gold:
                continue
            em_scores.append(self._exact_match(pred, gold))
            f1_scores.append(self._token_f1(pred, gold))

        avg_em = float(np.mean(em_scores)) if em_scores else 0
        avg_f1 = float(np.mean(f1_scores)) if f1_scores else 0
        logger.info(f"Generation EM: {avg_em:.4f}, F1: {avg_f1:.4f} ({len(em_scores)} queries)")

        return {"exact_match": avg_em, "f1": avg_f1, "n_queries": len(em_scores)}

    def aggregate_metrics(self, dataset_name: str) -> dict:
        """Aggregate all metrics into single report."""
        metrics = {}

        # Retrieval + Generation scores
        metrics["retrieval"] = self.evaluate_retrieval(dataset_name)
        metrics["generation"] = self.evaluate_generation(dataset_name)

        # Token usage
        token_path = Path(self.config.metrics_dir) / dataset_name / "token-usage.json"
        if token_path.exists():
            with open(token_path) as f:
                token_records = json.load(f)
            metrics["token_usage"] = self._summarize_tokens(token_records)

        # Timing
        timing_path = Path(self.config.metrics_dir) / dataset_name / "timing.json"
        if timing_path.exists():
            with open(timing_path) as f:
                timing_records = json.load(f)
            metrics["timing"] = self._summarize_timing(timing_records)

        # Save report
        report_dir = Path(self.config.output_dir) / dataset_name
        report_dir.mkdir(parents=True, exist_ok=True)
        report_path = report_dir / "evaluation-report.json"
        with open(report_path, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Evaluation report saved to {report_path}")

        return metrics

    def _summarize_tokens(self, records: list[dict]) -> dict:
        """Summarize token usage by phase."""
        summary = {}
        for phase in ("indexing", "retrieval", "generation"):
            phase_recs = [r for r in records if r.get("phase") == phase]
            if not phase_recs:
                summary[phase] = {"avg_input": 0, "avg_output": 0, "total": 0, "count": 0}
                continue
            n = len(phase_recs)
            summary[phase] = {
                "avg_input": sum(r["input_tokens"] for r in phase_recs) / n,
                "avg_output": sum(r["output_tokens"] for r in phase_recs) / n,
                "total_input": sum(r["input_tokens"] for r in phase_recs),
                "total_output": sum(r["output_tokens"] for r in phase_recs),
                "total": sum(r["total_tokens"] for r in phase_recs),
                "count": n,
            }
        return summary

    def _summarize_timing(self, records: list[dict]) -> dict:
        """Summarize timing by phase."""
        phases: dict[str, list[float]] = {}
        for r in records:
            phases.setdefault(r["phase"], []).append(r["duration_s"])
        return {
            phase: {
                "avg_s": round(sum(d) / len(d), 4),
                "total_s": round(sum(d), 4),
                "count": len(d),
            }
            for phase, d in phases.items()
        }

    def _exact_match(self, pred: str, gold: str) -> float:
        return float(self._normalize(pred) == self._normalize(gold))

    def _token_f1(self, pred: str, gold: str) -> float:
        """SQuAD-style token F1 using Counter for duplicate handling."""
        from collections import Counter
        pred_tokens = Counter(self._normalize(pred).split())
        gold_tokens = Counter(self._normalize(gold).split())
        if not gold_tokens:
            return float(not pred_tokens)
        if not pred_tokens:
            return 0.0
        # Intersection counts (min of each token count)
        common = sum((pred_tokens & gold_tokens).values())
        if common == 0:
            return 0.0
        precision = common / sum(pred_tokens.values())
        recall = common / sum(gold_tokens.values())
        return 2 * precision * recall / (precision + recall)

    def _normalize(self, text: str) -> str:
        """SQuAD-style normalization."""
        text = text.lower().strip()
        text = re.sub(r'\b(a|an|the)\b', ' ', text)
        text = re.sub(r'[^\w\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text
