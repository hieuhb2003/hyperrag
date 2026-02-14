"""
Centralized logging for HyP-DLM with debug/progress tracking.

Usage:
    from src.utils.logger import get_logger
    logger = get_logger(__name__)
    logger.step("Semantic Chunking", "Created 150 chunks from 10 documents")
    logger.matrix("Incidence Matrix H", H)
"""

import json
import logging
import sys
import time
from functools import wraps
from pathlib import Path
from typing import Any, Optional

import numpy as np
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

console = Console()

# ── Custom log levels ──
STEP_LEVEL = 25  # Between INFO(20) and WARNING(30)
logging.addLevelName(STEP_LEVEL, "STEP")


class HypDLMLogger(logging.Logger):
    """Extended logger with step tracking, matrix stats, and debug toolkit."""

    def __init__(self, name: str, level: int = logging.DEBUG):
        super().__init__(name, level)
        self._step_counter = 0
        self._timers: dict[str, float] = {}
        # Debug state
        self._debug_enabled = False
        self._debug_output_dir: Optional[str] = None
        self._debug_timings: list[dict] = []
        self._debug_max_samples = 3

    def enable_debug(self, output_dir: str, max_samples: int = 3) -> None:
        """Enable debug mode with checkpoint output directory."""
        self._debug_enabled = True
        self._debug_output_dir = output_dir
        self._debug_max_samples = max_samples
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        self.info(f"Debug mode enabled. Output: {output_dir}")

    @property
    def is_debug(self) -> bool:
        return self._debug_enabled

    # ── Step-level logging (shows progress to user) ──

    def step(self, phase: str, message: str, **kwargs: Any) -> None:
        self._step_counter += 1
        text = f"[STEP {self._step_counter}] [{phase}] {message}"
        if kwargs:
            details = " | ".join(f"{k}={v}" for k, v in kwargs.items())
            text += f"  ({details})"
        self.log(STEP_LEVEL, text)
        console.print(
            Panel(text, title=f"Step {self._step_counter}", border_style="cyan")
        )

    # ── Matrix statistics ──

    def matrix(self, name: str, mat: Any) -> None:
        """Log sparse/dense matrix statistics."""
        from scipy import sparse

        info_lines = [f"  Matrix: {name}"]
        if sparse.issparse(mat):
            info_lines.append(f"  Shape: {mat.shape}")
            info_lines.append(f"  NNZ: {mat.nnz}")
            density = mat.nnz / (mat.shape[0] * mat.shape[1]) if mat.shape[0] * mat.shape[1] > 0 else 0
            info_lines.append(f"  Density: {density:.6f}")
            info_lines.append(f"  Format: {mat.format}")
        elif isinstance(mat, np.ndarray):
            info_lines.append(f"  Shape: {mat.shape}")
            info_lines.append(f"  Dtype: {mat.dtype}")
            info_lines.append(f"  Min: {mat.min():.4f}, Max: {mat.max():.4f}, Mean: {mat.mean():.4f}")
            info_lines.append(f"  Non-zero: {np.count_nonzero(mat)}")
        else:
            info_lines.append(f"  Type: {type(mat)}")

        text = "\n".join(info_lines)
        self.debug(text)
        console.print(Panel(text, title=f"Matrix: {name}", border_style="green"))

    # ── Timer utilities ──

    def start_timer(self, label: str) -> None:
        self._timers[label] = time.time()

    def stop_timer(self, label: str) -> float:
        """Stop a named timer and return elapsed seconds. Returns -1 if not found."""
        if label not in self._timers:
            return -1.0
        elapsed = time.time() - self._timers.pop(label)
        self.debug(f"{label}: {elapsed:.2f}s")
        return elapsed

    # ── Summary table ──

    def summary_table(self, title: str, rows: list[dict[str, Any]]) -> None:
        """Print a rich summary table."""
        if not rows:
            return
        table = Table(title=title, show_lines=True)
        for key in rows[0]:
            table.add_column(key, style="cyan")
        for row in rows:
            table.add_row(*[str(v) for v in row.values()])
        console.print(table)

    # ── Convergence log ──

    def convergence(self, step: int, delta: float, active: int, max_score: float) -> None:
        self.debug(
            f"  Propagation step {step}: delta={delta:.6f}, "
            f"active_entities={active}, max_score={max_score:.4f}"
        )

    # ── Debug toolkit methods ──

    @staticmethod
    def _format_item(item: Any) -> str:
        """Format an item for debug display (truncate long strings)."""
        s = str(item)
        return s[:200] + "..." if len(s) > 200 else s

    def debug_sample(self, title: str, items: list, max_items: Optional[int] = None) -> None:
        """Show sample items in a Rich panel. No-op if debug disabled."""
        if not self._debug_enabled:
            return
        n = max_items or self._debug_max_samples
        sample = items[:n]
        lines = [f"  [{i+1}/{len(items)}] {self._format_item(item)}" for i, item in enumerate(sample)]
        if len(items) > n:
            lines.append(f"  ... and {len(items) - n} more")
        console.print(Panel("\n".join(lines), title=f"[DEBUG] {title} ({len(items)} total)", border_style="yellow"))

    def debug_distribution(self, title: str, data: dict[str, int]) -> None:
        """Show distribution as a Rich table. No-op if debug disabled."""
        if not self._debug_enabled:
            return
        table = Table(title=f"[DEBUG] {title}", show_lines=True)
        table.add_column("Category", style="cyan")
        table.add_column("Count", style="green", justify="right")
        table.add_column("Percent", style="yellow", justify="right")
        total = sum(data.values())
        for key, count in sorted(data.items(), key=lambda x: -x[1]):
            pct = f"{count / total * 100:.1f}%" if total > 0 else "0%"
            table.add_row(str(key), str(count), pct)
        table.add_row("TOTAL", str(total), "100%", style="bold")
        console.print(table)

    def debug_checkpoint(self, name: str, data: dict, step_time: Optional[float] = None) -> None:
        """Save a debug checkpoint to output_dir. No-op if debug disabled.

        Handles JSON-serializable data, numpy arrays (.npy), scipy sparse (.npz).
        """
        if not self._debug_enabled or not self._debug_output_dir:
            return

        out_dir = Path(self._debug_output_dir)
        json_data = {}

        for key, value in data.items():
            if isinstance(value, np.ndarray):
                arr_path = out_dir / f"{name}_{key}.npy"
                np.save(arr_path, value)
                json_data[key] = f"<numpy {value.shape} -> {arr_path.name}>"
            elif hasattr(value, 'nnz'):  # scipy sparse
                from scipy import sparse
                mat_path = out_dir / f"{name}_{key}.npz"
                sparse.save_npz(mat_path, value)
                json_data[key] = f"<sparse {value.shape} nnz={value.nnz} -> {mat_path.name}>"
            else:
                try:
                    json.dumps(value)
                    json_data[key] = value
                except (TypeError, ValueError):
                    json_data[key] = str(value)

        json_path = out_dir / f"{name}.json"
        with open(json_path, "w") as f:
            json.dump(json_data, f, indent=2, default=str)

        if step_time is not None:
            self._debug_timings.append({"step": name, "time_s": round(step_time, 3)})

        self.debug(f"Checkpoint saved: {json_path}")

    def debug_matrix_detail(self, name: str, mat: Any) -> None:
        """Detailed matrix analysis. No-op if debug disabled."""
        if not self._debug_enabled:
            return
        from scipy import sparse as sp

        lines = [f"Matrix: {name}"]
        if sp.issparse(mat):
            lines.append(f"  Shape: {mat.shape}")
            lines.append(f"  NNZ: {mat.nnz}")
            total = mat.shape[0] * mat.shape[1]
            density = mat.nnz / total if total > 0 else 0
            lines.append(f"  Density: {density:.6f} ({density*100:.4f}%)")
            lines.append(f"  Format: {mat.format}")
            row_nnz = np.diff(mat.tocsr().indptr)
            lines.append(f"  Row NNZ — min: {row_nnz.min()}, max: {row_nnz.max()}, mean: {row_nnz.mean():.1f}")
            lines.append(f"  Empty rows: {np.sum(row_nnz == 0)}")
            col_nnz = np.diff(mat.tocsc().indptr)
            lines.append(f"  Col NNZ — min: {col_nnz.min()}, max: {col_nnz.max()}, mean: {col_nnz.mean():.1f}")
            lines.append(f"  Empty cols: {np.sum(col_nnz == 0)}")
            if mat.nnz > 0:
                vals = mat.data
                lines.append(f"  Values — min: {vals.min():.4f}, max: {vals.max():.4f}, mean: {vals.mean():.4f}")
        elif isinstance(mat, np.ndarray):
            lines.append(f"  Shape: {mat.shape}")
            lines.append(f"  Dtype: {mat.dtype}")
            lines.append(f"  Min: {mat.min():.4f}, Max: {mat.max():.4f}")
            lines.append(f"  Mean: {mat.mean():.4f}, Std: {mat.std():.4f}")
            lines.append(f"  Non-zero: {np.count_nonzero(mat)} / {mat.size}")
        console.print(Panel("\n".join(lines), title=f"[DEBUG] Matrix: {name}", border_style="magenta"))

    def debug_report(self, extra: Optional[dict] = None) -> None:
        """Write final debug report with all timings. No-op if debug disabled."""
        if not self._debug_enabled or not self._debug_output_dir:
            return
        report = {
            "timings": self._debug_timings,
            "total_time_s": sum(t["time_s"] for t in self._debug_timings),
            "steps": len(self._debug_timings),
        }
        if extra:
            report.update(extra)

        report_path = Path(self._debug_output_dir) / "debug_report.json"
        with open(report_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        console.print(Panel(
            f"Debug report saved: {report_path}\n"
            f"Total steps: {report['steps']}\n"
            f"Total time: {report['total_time_s']:.2f}s",
            title="[DEBUG] Report Complete",
            border_style="bold green",
        ))


def get_logger(name: str, level: Optional[str] = None) -> HypDLMLogger:
    """Get or create a HyP-DLM logger."""
    logging.setLoggerClass(HypDLMLogger)
    logger = logging.getLogger(name)
    if not isinstance(logger, HypDLMLogger):
        # Fallback: re-register
        logger.__class__ = HypDLMLogger
        logger._step_counter = 0
        logger._timers = {}
        logger._debug_enabled = False
        logger._debug_output_dir = None
        logger._debug_timings = []
        logger._debug_max_samples = 3

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        fmt = logging.Formatter(
            "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
            datefmt="%H:%M:%S",
        )
        handler.setFormatter(fmt)
        logger.addHandler(handler)

    if level:
        logger.setLevel(getattr(logging, level.upper(), logging.DEBUG))
    elif logger.level == logging.NOTSET:
        logger.setLevel(logging.DEBUG)

    return logger


def get_progress() -> Progress:
    """Get a Rich progress bar for long operations."""
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        console=console,
    )
