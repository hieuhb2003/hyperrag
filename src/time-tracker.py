"""Thread-safe timing tracker for pipeline phases. Saves to JSON."""

from __future__ import annotations

import json
import threading
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path


class TimeTracker:
    """Track execution time per phase per item. Thread-safe."""

    def __init__(self, metrics_dir: str):
        self._metrics_dir = Path(metrics_dir)
        self._timings: list[dict] = []
        self._lock = threading.Lock()

    @contextmanager
    def track(self, phase: str, item_id: str):
        """Context manager to time a phase for a specific item.

        Usage:
            with time_tracker.track("coref", "doc_001"):
                do_coref(doc)
        """
        start = time.perf_counter()
        yield
        elapsed = time.perf_counter() - start
        with self._lock:
            self._timings.append({
                "phase": phase,
                "item_id": item_id,
                "duration_s": round(elapsed, 4),
                "timestamp": datetime.now().isoformat(),
            })

    def save(self, dataset_name: str) -> Path:
        """Save all timing records to metrics/{dataset_name}/timing.json"""
        out_dir = self._metrics_dir / dataset_name
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "timing.json"

        with self._lock:
            records = list(self._timings)

        with open(out_path, "w") as f:
            json.dump(records, f, indent=2)
        return out_path

    def summary(self) -> dict:
        """Avg time per doc (indexing phases), per query (retrieval+gen)."""
        with self._lock:
            records = list(self._timings)

        # Group by phase
        phases: dict[str, list[float]] = {}
        for r in records:
            phases.setdefault(r["phase"], []).append(r["duration_s"])

        result = {}
        for phase, durations in phases.items():
            result[phase] = {
                "avg_s": round(sum(durations) / len(durations), 4),
                "total_s": round(sum(durations), 4),
                "count": len(durations),
            }
        return result
