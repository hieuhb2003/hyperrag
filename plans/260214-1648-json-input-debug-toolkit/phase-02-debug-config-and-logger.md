# Phase 02: Debug Config & Logger Enhancements

## Context Links

- [plan.md](./plan.md)
- [logger.py](../../hyp_dlm/src/utils/logger.py) -- main file to modify
- [default.yaml](../../hyp_dlm/config/default.yaml) -- add debug section
- [index_corpus.py](../../hyp_dlm/scripts/index_corpus.py) -- add `--debug` flag
- [query.py](../../hyp_dlm/scripts/query.py) -- add `--debug` flag

## Overview

- **Priority**: P2
- **Status**: complete
- **Effort**: 1h
- **Description**: Add debug infrastructure -- config section, CLI flags, and logger helper methods -- that Phase 03 and Phase 04 will use for detailed intermediate output.

## Key Insights

- `HypDLMLogger` already has `step()`, `matrix()`, `convergence()`, `summary_table()` methods. Debug methods should follow the same pattern (Rich panels/tables, log to handler).
- Debug mode should be a runtime flag, not a config-only setting. The `--debug` CLI flag overrides config.
- Checkpoint saving needs to handle both JSON-serializable data (stats, scores) and non-serializable objects (numpy arrays, sparse matrices). Use JSON for inspectable data, numpy `.npy` for arrays.
- Timing per step already uses `start_timer()`/`stop_timer()`. Debug mode just needs to aggregate and report all timings at the end.

## Requirements

### Functional
1. Add `debug` section to `config/default.yaml` with defaults
2. Add `--debug` CLI flag to `index_corpus.py` and `query.py`
3. Add 4 new logger methods: `debug_sample()`, `debug_distribution()`, `debug_checkpoint()`, `debug_matrix_detail()`
4. Add `debug_report()` method to produce final JSON summary

### Non-Functional
- Debug methods must be no-ops when debug mode is off (check `self._debug_enabled`)
- Zero overhead when debug is disabled -- no data collection, no formatting
- Checkpoint files should be human-inspectable where possible (JSON > pickle)

## Architecture

```
CLI --debug flag
      |
      v
config["debug"]["enabled"] = True   (override)
      |
      v
logger.enable_debug(output_dir)     (sets _debug_enabled, _debug_output_dir)
      |
      v
Throughout pipeline:
  logger.debug_sample(...)           -> Rich panel with sample items
  logger.debug_distribution(...)     -> Rich table with stats
  logger.debug_checkpoint(...)       -> saves file to data/debug/
  logger.debug_matrix_detail(...)    -> detailed matrix analysis
      |
      v
End of pipeline:
  logger.debug_report()              -> writes debug_report.json
```

## Related Code Files

### Files to Modify
- `hyp_dlm/src/utils/logger.py` -- Add 5 new methods to `HypDLMLogger`
- `hyp_dlm/config/default.yaml` -- Add `debug` section
- `hyp_dlm/scripts/index_corpus.py` -- Add `--debug` flag, call `logger.enable_debug()`
- `hyp_dlm/scripts/query.py` -- Add `--debug` flag, call `logger.enable_debug()`

### Files NOT Modified
- No new files created

## Implementation Steps

### Step 1: Add debug section to default.yaml

Add after the `logging` section:

```yaml
# === Debug Mode ===
debug:
  enabled: false
  output_dir: "data/debug"
  save_checkpoints: true
  save_report: true
  max_sample_items: 3
```

### Step 2: Add debug state to HypDLMLogger

Add instance variables and `enable_debug()` method:

```python
class HypDLMLogger(logging.Logger):
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
```

### Step 3: Add debug_sample() method

Shows a sample of items in a Rich panel:

```python
def debug_sample(self, title: str, items: list, max_items: Optional[int] = None) -> None:
    """Show sample items in a Rich panel. No-op if debug disabled."""
    if not self._debug_enabled:
        return

    n = max_items or self._debug_max_samples
    sample = items[:n]
    lines = [f"  [{i+1}/{len(items)}] {self._format_item(item)}" for i, item in enumerate(sample)]
    if len(items) > n:
        lines.append(f"  ... and {len(items) - n} more")

    text = "\n".join(lines)
    console.print(Panel(text, title=f"[DEBUG] {title} ({len(items)} total)", border_style="yellow"))

@staticmethod
def _format_item(item: Any) -> str:
    """Format an item for debug display (truncate long strings)."""
    s = str(item)
    return s[:200] + "..." if len(s) > 200 else s
```

### Step 4: Add debug_distribution() method

Shows distribution stats in a Rich table:

```python
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
```

### Step 5: Add debug_checkpoint() method

Saves intermediate state to disk:

```python
def debug_checkpoint(self, name: str, data: dict, step_time: Optional[float] = None) -> None:
    """Save a debug checkpoint to output_dir. No-op if debug disabled.

    Args:
        name: Checkpoint name (becomes filename)
        data: Dict of data to save. Values can be:
            - JSON-serializable -> saved as JSON
            - numpy array -> saved as .npy
            - scipy sparse -> saved as .npz
        step_time: Optional elapsed time for this step
    """
    if not self._debug_enabled or not self._debug_output_dir:
        return

    import json as json_mod
    from pathlib import Path

    out_dir = Path(self._debug_output_dir)
    json_data = {}
    special_files = []

    for key, value in data.items():
        if isinstance(value, np.ndarray):
            arr_path = out_dir / f"{name}_{key}.npy"
            np.save(arr_path, value)
            special_files.append(str(arr_path))
            json_data[key] = f"<numpy array {value.shape} saved to {arr_path.name}>"
        elif hasattr(value, 'nnz'):  # scipy sparse
            from scipy import sparse
            mat_path = out_dir / f"{name}_{key}.npz"
            sparse.save_npz(mat_path, value)
            special_files.append(str(mat_path))
            json_data[key] = f"<sparse matrix {value.shape} nnz={value.nnz} saved to {mat_path.name}>"
        else:
            try:
                json_mod.dumps(value)
                json_data[key] = value
            except (TypeError, ValueError):
                json_data[key] = str(value)

    # Save JSON checkpoint
    json_path = out_dir / f"{name}.json"
    with open(json_path, "w") as f:
        json_mod.dump(json_data, f, indent=2, default=str)

    # Track timing
    if step_time is not None:
        self._debug_timings.append({"step": name, "time_s": round(step_time, 3)})

    self.debug(f"Checkpoint saved: {json_path}")
```

### Step 6: Add debug_matrix_detail() method

Enhanced matrix analysis beyond current `matrix()`:

```python
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
        # Row stats
        row_nnz = np.diff(mat.tocsr().indptr)
        lines.append(f"  Row NNZ — min: {row_nnz.min()}, max: {row_nnz.max()}, mean: {row_nnz.mean():.1f}, median: {np.median(row_nnz):.0f}")
        lines.append(f"  Empty rows: {np.sum(row_nnz == 0)}")
        # Col stats
        col_nnz = np.diff(mat.tocsc().indptr)
        lines.append(f"  Col NNZ — min: {col_nnz.min()}, max: {col_nnz.max()}, mean: {col_nnz.mean():.1f}, median: {np.median(col_nnz):.0f}")
        lines.append(f"  Empty cols: {np.sum(col_nnz == 0)}")
        # Value stats
        if mat.nnz > 0:
            vals = mat.data
            lines.append(f"  Values — min: {vals.min():.4f}, max: {vals.max():.4f}, mean: {vals.mean():.4f}")
    elif isinstance(mat, np.ndarray):
        lines.append(f"  Shape: {mat.shape}")
        lines.append(f"  Dtype: {mat.dtype}")
        lines.append(f"  Min: {mat.min():.4f}, Max: {mat.max():.4f}")
        lines.append(f"  Mean: {mat.mean():.4f}, Std: {mat.std():.4f}")
        lines.append(f"  Non-zero: {np.count_nonzero(mat)} / {mat.size}")
        lines.append(f"  Norm (L2): {np.linalg.norm(mat):.4f}")

    text = "\n".join(lines)
    console.print(Panel(text, title=f"[DEBUG] Matrix Detail: {name}", border_style="magenta"))
```

### Step 7: Add debug_report() method

Produces final JSON summary:

```python
def debug_report(self, extra: Optional[dict] = None) -> None:
    """Write final debug report with all timings. No-op if debug disabled."""
    if not self._debug_enabled or not self._debug_output_dir:
        return

    import json as json_mod
    from pathlib import Path

    report = {
        "timings": self._debug_timings,
        "total_time_s": sum(t["time_s"] for t in self._debug_timings),
        "steps": len(self._debug_timings),
    }
    if extra:
        report.update(extra)

    report_path = Path(self._debug_output_dir) / "debug_report.json"
    with open(report_path, "w") as f:
        json_mod.dump(report, f, indent=2, default=str)

    console.print(Panel(
        f"Debug report saved: {report_path}\n"
        f"Total steps: {report['steps']}\n"
        f"Total time: {report['total_time_s']:.2f}s",
        title="[DEBUG] Report Complete",
        border_style="bold green",
    ))
```

### Step 8: Add --debug CLI flag to index_corpus.py

In `main()`:

```python
parser.add_argument("--debug", action="store_true", help="Enable debug mode with detailed intermediate output")
```

After loading config:

```python
# Debug mode
debug_config = config.get("debug", {})
if args.debug or debug_config.get("enabled", False):
    debug_output = debug_config.get("output_dir", "data/debug")
    max_samples = debug_config.get("max_sample_items", 3)
    logger.enable_debug(debug_output, max_samples=max_samples)
```

### Step 9: Add --debug CLI flag to query.py

Same pattern as Step 8 in `main()`.

## Todo List

- [ ] Add `debug` section to `config/default.yaml`
- [ ] Add `_debug_enabled`, `_debug_output_dir`, `_debug_timings` to `HypDLMLogger.__init__`
- [ ] Add `enable_debug()` method
- [ ] Add `is_debug` property
- [ ] Add `debug_sample()` method
- [ ] Add `_format_item()` static method
- [ ] Add `debug_distribution()` method
- [ ] Add `debug_checkpoint()` method
- [ ] Add `debug_matrix_detail()` method
- [ ] Add `debug_report()` method
- [ ] Add `--debug` flag to `index_corpus.py` `main()`
- [ ] Add `--debug` flag to `query.py` `main()`
- [ ] Add debug enable logic after config load in both scripts

## Success Criteria

1. `logger.enable_debug("data/debug")` creates the output directory
2. `logger.is_debug` returns correct state
3. All debug methods are silent no-ops when `_debug_enabled = False`
4. `debug_checkpoint()` saves JSON + numpy/sparse files
5. `debug_report()` produces valid `debug_report.json`
6. `--debug` flag works on both scripts
7. Config `debug.enabled: true` also enables debug mode
8. All existing tests pass (no regressions)

## Risk Assessment

| Risk | Impact | Likelihood | Mitigation |
|------|--------|------------|------------|
| Debug output clutters terminal | Low | Medium | Debug panels use distinct yellow/magenta border styles |
| Checkpoint disk usage | Low | Low | Only saved when `save_checkpoints: true` (default) |
| Logger class mutation (re-register) | Medium | Low | `enable_debug()` uses instance vars, safe with `__class__` swap in `get_logger()` |
| Import cycles | Low | Low | Lazy import `scipy.sparse` inside methods, same pattern as existing `matrix()` |

## Security Considerations

- Debug output directory is user-controlled via config; no path traversal risk since `mkdir(parents=True)` is safe
- Checkpoint data comes from internal pipeline state, not user input -- no injection risk
- Debug report JSON uses `default=str` to prevent serialization errors, never `eval()`

## Next Steps

- Phase 03 uses these logger methods in `index_corpus.py`
- Phase 04 uses these logger methods in `query.py`
- Both phases depend on this phase completing first
