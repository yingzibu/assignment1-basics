"""Local experiment tracking and loss curves for CS336."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from datetime import datetime, timezone
import json
import math
import operator
import os
from pathlib import Path
import shutil
import time


def _write_json(path: Path, value) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def read_metrics(path: str | Path) -> list[dict]:
    path = Path(path)
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    records = []
    for index, line in enumerate(lines):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            # A process may stop during its last write.
            if index == len(lines) - 1:
                break
            raise ValueError(f"{path}: malformed JSON at record {index + 1}") from error
        if record.get("split") not in {"train", "validation"}:
            raise ValueError(f"{path}: unknown split at record {index + 1}")
        step = operator.index(record["step"])
        if step < 0 or not math.isfinite(float(record["loss"])):
            raise ValueError(f"{path}: invalid step or loss at record {index + 1}")
        elapsed = record.get("elapsed_seconds")
        if elapsed is not None and (not math.isfinite(float(elapsed)) or elapsed < 0):
            raise ValueError(f"{path}: invalid elapsed_seconds at record {index + 1}")
        records.append(record)
    return records


def summarize_records(records: list[dict]) -> dict:
    validation = [record for record in records if record["split"] == "validation"]
    best = min(validation, key=lambda record: record["loss"]) if validation else None
    times = [record["elapsed_seconds"] for record in records if record.get("elapsed_seconds") is not None]
    return {
        "num_records": len(records),
        "last_logged_step": max((record["step"] for record in records), default=None),
        "elapsed_seconds_last_record": max(times, default=None),
        "best_logged_validation_loss": best["loss"] if best else None,
        "best_logged_validation_step": best["step"] if best else None,
        "records_without_wall_time": sum(record.get("elapsed_seconds") is None for record in records),
    }


class ExperimentLogger:
    """Record optimizer updates and cumulative active-process wall time.

    resume_step selects the checkpoint boundary. Existing logs are backed up
    before records beyond this step are removed. Offline downtime is excluded.
    """

    def __init__(
        self,
        run_dir: str | Path,
        config: dict | None = None,
        *,
        resume_step: int | None = None,
        elapsed_seconds: float = 0.0,
        clock: Callable[[], float] = time.perf_counter,
    ):
        if not math.isfinite(elapsed_seconds) or elapsed_seconds < 0:
            raise ValueError("elapsed_seconds must be finite and nonnegative")
        if resume_step is not None and operator.index(resume_step) < 0:
            raise ValueError("resume_step must be nonnegative")
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.run_dir / "metrics.jsonl"
        self._records = []
        self.backup_path = None
        if self.path.exists():
            if resume_step is None:
                raise FileExistsError(f"{self.path} exists; resume or choose a new run directory")
            records = read_metrics(self.path)
            self._records = [record for record in records if record["step"] <= resume_step]
            self.backup_path = self.path.with_name(f"metrics.before_resume_{time.time_ns()}.jsonl")
            shutil.copy2(self.path, self.backup_path)
            temporary = self.path.with_name(self.path.name + ".tmp")
            temporary.write_text(
                "".join(json.dumps(record, allow_nan=False) + "\n" for record in self._records),
                encoding="utf-8",
            )
            temporary.replace(self.path)
        known_times = [
            record["elapsed_seconds"] for record in self._records
            if record.get("elapsed_seconds") is not None
        ]
        self._offset = max(float(elapsed_seconds), max(known_times, default=0.0))
        self._clock = clock
        self._started = clock()
        self._finished_elapsed = None
        self._last_step = resume_step if resume_step is not None else 0
        self._closed = False
        if config is not None:
            _write_json(self.run_dir / "config.json", config)
        self._stream = self.path.open("a", encoding="utf-8")

    @property
    def elapsed_seconds(self) -> float:
        if self._finished_elapsed is not None:
            return self._finished_elapsed
        return self._offset + max(0.0, self._clock() - self._started)

    def log(self, split: str, step: int, loss: float, **metrics) -> dict:
        if self._closed:
            raise RuntimeError("Logger is closed")
        if split not in {"train", "validation"}:
            raise ValueError("split must be train or validation")
        step = operator.index(step)
        if step < 0 or step < self._last_step:
            raise ValueError("step must be nonnegative and cannot decrease")
        loss = float(loss)
        if not math.isfinite(loss):
            raise ValueError("loss must be finite")
        reserved = {"split", "step", "loss", "elapsed_seconds", "timestamp_utc"}
        if reserved.intersection(metrics):
            raise ValueError("Extra metrics cannot replace reserved fields")
        record = {
            "split": split,
            "step": step,
            "elapsed_seconds": self.elapsed_seconds,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "loss": loss,
            **metrics,
        }
        line = json.dumps(record, ensure_ascii=False, allow_nan=False)
        self._stream.write(line + "\n")
        self._stream.flush()
        self._records.append(record)
        self._last_step = step
        return record

    def close(self, status: str = "completed") -> None:
        if self._closed:
            return
        self._finished_elapsed = self.elapsed_seconds
        self._stream.close()
        self._closed = True
        summary = summarize_records(self._records)
        summary.update({
            "status": status,
            "elapsed_seconds": self._finished_elapsed,
            "run_dir": str(self.run_dir.resolve()),
        })
        _write_json(self.run_dir / "summary.json", summary)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        status = "completed" if exc_type is None else (
            "interrupted" if issubclass(exc_type, KeyboardInterrupt) else "failed"
        )
        self.close(status)
        return False


def plot_runs(runs: list[str | Path], output: str | Path) -> dict[str, str]:
    # Colab passes its inline backend to uv subprocesses, where that backend
    # may be unavailable. Select Agg during import, then restore the environment.
    # Explicit canvases also preserve an already-loaded notebook pyplot backend.
    previous_backend = os.environ.get("MPLBACKEND")
    os.environ["MPLBACKEND"] = "Agg"
    try:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
    finally:
        if previous_backend is None:
            os.environ.pop("MPLBACKEND", None)
        else:
            os.environ["MPLBACKEND"] = previous_backend

    figure = Figure(figsize=(12, 4.5))
    FigureCanvasAgg(figure)
    axes = figure.subplots(1, 2)
    any_records = False
    missing_times = 0
    for index, run in enumerate(runs):
        run = Path(run)
        path = run / "metrics.jsonl" if run.is_dir() else run
        records = read_metrics(path)
        label = run.name if run.is_dir() else run.parent.name
        color = f"C{index % 10}"
        for split in ("train", "validation"):
            selected = [record for record in records if record["split"] == split]
            if not selected:
                continue
            any_records = True
            style = "--" if split == "train" else "-"
            kwargs = {
                "label": f"{label} / {split}", "color": color, "linestyle": style,
                "marker": "o", "markersize": 2,
                "alpha": 0.75 if split == "train" else 1.0,
            }
            axes[0].plot(
                [record["step"] for record in selected],
                [record["loss"] for record in selected], **kwargs,
            )
            timed = [record for record in selected if record.get("elapsed_seconds") is not None]
            missing_times += len(selected) - len(timed)
            if timed:
                axes[1].plot(
                    [record["elapsed_seconds"] / 60 for record in timed],
                    [record["loss"] for record in timed], **kwargs,
                )
    if not any_records:
        raise ValueError("No train or validation records to plot")
    axes[0].set(xlabel="Optimizer updates", ylabel="Cross-entropy loss", title="Loss versus updates")
    axes[1].set(xlabel="Elapsed wall time (minutes)", ylabel="Cross-entropy loss", title="Loss versus wall time")
    for axis in axes:
        axis.grid(alpha=0.25)
        handles, labels = axis.get_legend_handles_labels()
        if handles:
            axis.legend(fontsize=8)
    if not axes[1].lines:
        axes[1].text(
            0.5, 0.5, "Wall-clock timestamps unavailable",
            ha="center", va="center", transform=axes[1].transAxes,
        )
    if missing_times:
        figure.text(
            0.5, 0.01,
            f"{missing_times} legacy records have no wall-clock timestamp; the time panel uses timed records.",
            ha="center", fontsize=8,
        )
    figure.tight_layout(rect=(0, 0.05 if missing_times else 0, 1, 1))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    png = output.with_suffix(".png")
    pdf = output.with_suffix(".pdf")
    figure.savefig(png, dpi=180)
    figure.savefig(pdf)
    return {"png": str(png.resolve()), "pdf": str(pdf.resolve())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path, help="Run directories or metrics.jsonl files")
    parser.add_argument("--output", type=Path, default=Path("artifacts/learning_curves"))
    parser.add_argument("--summary", action="store_true", help="Print logged results as JSON")
    args = parser.parse_args()
    if args.summary:
        results = []
        for run in args.runs:
            path = run / "metrics.jsonl" if run.is_dir() else run
            results.append({"run": str(run), **summarize_records(read_metrics(path))})
        print(json.dumps(results, indent=2, allow_nan=False))
    else:
        print(json.dumps(plot_runs(args.runs, args.output), indent=2))


if __name__ == "__main__":
    main()

