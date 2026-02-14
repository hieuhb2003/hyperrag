#!/usr/bin/env python3
"""
CLI: Run evaluation on benchmarks.

Usage:
    python scripts/evaluate.py \
        --index_dir data/indexed/hotpotqa \
        --benchmark data/benchmarks/hotpotqa_dev.json \
        --config config/default.yaml \
        --output results/hotpotqa_results.json
"""

import argparse
import json
import re
import string
import sys
from collections import Counter
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.utils.logger import get_logger, get_progress

logger = get_logger(__name__)


# ══════════════════════════════════════════════════════════
# Metrics
# ══════════════════════════════════════════════════════════


def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    return white_space_fix(remove_articles(remove_punc(s.lower())))


def compute_f1(prediction: str, gold: str) -> float:
    """Compute word-level F1 between prediction and gold answer."""
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(gold).split()

    if not gold_tokens:
        return 1.0 if not pred_tokens else 0.0

    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def compute_exact_match(prediction: str, gold: str) -> float:
    """Compute exact match (after normalization)."""
    return float(normalize_answer(prediction) == normalize_answer(gold))


def evaluate_benchmark(
    index_dir: str,
    benchmark_path: str,
    config: dict,
    output_path: str | None = None,
    max_samples: int | None = None,
) -> dict:
    """
    Evaluate HyP-DLM on a benchmark dataset.

    Benchmark format (JSON):
    [
        {"question": "...", "answer": "...", "supporting_facts": [...]},
        ...
    ]
    """
    from scripts.query import run_query

    logger.step("Evaluate", "=" * 60)
    logger.step("Evaluate", f"Benchmark: {benchmark_path}")
    logger.step("Evaluate", "=" * 60)

    # Load benchmark
    with open(benchmark_path) as f:
        data = json.load(f)

    if isinstance(data, dict):
        # Some formats have a "data" key
        data = data.get("data", data.get("questions", [data]))

    if max_samples:
        data = data[:max_samples]

    logger.step("Evaluate", f"Evaluating {len(data)} samples")

    results = []
    total_f1 = 0.0
    total_em = 0.0

    with get_progress() as progress:
        task = progress.add_task("Evaluating...", total=len(data))

        for i, sample in enumerate(data):
            question = sample.get("question", "")
            gold_answer = sample.get("answer", "")

            try:
                result = run_query(question, index_dir, config)
                predicted = result.get("answer", "")
            except Exception as e:
                logger.warning(f"  Sample {i}: Error — {e}")
                predicted = ""

            f1 = compute_f1(predicted, gold_answer)
            em = compute_exact_match(predicted, gold_answer)

            total_f1 += f1
            total_em += em

            results.append({
                "question": question,
                "gold_answer": gold_answer,
                "predicted_answer": predicted,
                "f1": f1,
                "exact_match": em,
                "route": result.get("route", "unknown"),
            })

            logger.debug(
                f"  Sample {i+1}/{len(data)}: F1={f1:.4f}, EM={em:.1f} "
                f"| pred='{predicted[:80]}' | gold='{gold_answer}'"
            )

            progress.advance(task)

    n = len(data)
    avg_f1 = total_f1 / n if n > 0 else 0
    avg_em = total_em / n if n > 0 else 0

    # Route distribution
    route_dist: dict[str, int] = {}
    for r in results:
        route_dist[r["route"]] = route_dist.get(r["route"], 0) + 1

    evaluation = {
        "benchmark": benchmark_path,
        "num_samples": n,
        "avg_f1": avg_f1,
        "avg_exact_match": avg_em,
        "route_distribution": route_dist,
        "results": results,
    }

    logger.step("Evaluate", "=" * 60)
    logger.step("Evaluate", "EVALUATION COMPLETE")
    logger.step("Evaluate", "=" * 60)

    logger.summary_table(
        "Evaluation Results",
        [
            {"Metric": "Avg F1", "Value": f"{avg_f1:.4f}"},
            {"Metric": "Avg Exact Match", "Value": f"{avg_em:.4f}"},
            {"Metric": "Samples", "Value": n},
            {"Metric": "Route Distribution", "Value": str(route_dist)},
        ],
    )

    # Save results
    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(evaluation, f, indent=2)
        logger.step("Evaluate", f"Results saved to {output_path}")

    return evaluation


def main():
    parser = argparse.ArgumentParser(description="HyP-DLM: Evaluate on benchmark")
    parser.add_argument("--index_dir", required=True)
    parser.add_argument("--benchmark", required=True)
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--output", default=None)
    parser.add_argument("--max_samples", type=int, default=None)
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    log_level = config.get("logging", {}).get("level", "DEBUG")
    logger.setLevel(log_level)

    evaluate_benchmark(
        args.index_dir, args.benchmark, config, args.output, args.max_samples
    )


if __name__ == "__main__":
    main()
