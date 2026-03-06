"""Thread-safe LLM token usage tracking. Saves per-call records to JSON."""

from __future__ import annotations

import json
import threading
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional


@dataclass
class TokenUsage:
    call_id: str
    phase: str  # "indexing" | "retrieval" | "generation"
    doc_or_query_id: str
    model: str
    input_tokens: int
    output_tokens: int
    total_tokens: int
    timestamp: str


class TokenTracker:
    """Record LLM token usage per call. Thread-safe."""

    def __init__(self, metrics_dir: str):
        self._metrics_dir = Path(metrics_dir)
        self._usages: list[TokenUsage] = []
        self._lock = threading.Lock()

    def record(
        self,
        phase: str,
        doc_or_query_id: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> None:
        """Record a single LLM call's token usage."""
        usage = TokenUsage(
            call_id=str(uuid.uuid4())[:8],
            phase=phase,
            doc_or_query_id=doc_or_query_id,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            timestamp=datetime.now().isoformat(),
        )
        with self._lock:
            self._usages.append(usage)

    def save(self, dataset_name: str) -> Path:
        """Save all token records to metrics/{dataset_name}/token-usage.json"""
        out_dir = self._metrics_dir / dataset_name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "token-usage.json"

        with self._lock:
            records = [asdict(u) for u in self._usages]

        with open(out_path, "w") as f:
            json.dump(records, f, indent=2)
        return out_path

    def summary(self) -> dict:
        """Return avg tokens per doc (indexing), per query (retrieval+gen)."""
        with self._lock:
            records = list(self._usages)

        result = {}
        for phase in ("indexing", "retrieval", "generation"):
            phase_records = [r for r in records if r.phase == phase]
            if not phase_records:
                result[phase] = {"avg_input": 0, "avg_output": 0, "avg_total": 0, "count": 0}
                continue
            n = len(phase_records)
            result[phase] = {
                "avg_input": sum(r.input_tokens for r in phase_records) / n,
                "avg_output": sum(r.output_tokens for r in phase_records) / n,
                "avg_total": sum(r.total_tokens for r in phase_records) / n,
                "total_input": sum(r.input_tokens for r in phase_records),
                "total_output": sum(r.output_tokens for r in phase_records),
                "count": n,
            }
        return result
