"""Small, dependency-free metric summaries for controlled DG comparisons."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .io_utils import read_json, write_json


def _parse_metric_spec(specification: str) -> tuple[str, Path]:
    if "=" not in specification:
        raise ValueError("Each --metric must use NAME=PATH, for example sft=runs/sft/metrics.json.")
    name, raw_path = specification.split("=", maxsplit=1)
    if not name or not raw_path:
        raise ValueError("Each --metric must use a non-empty NAME=PATH pair.")
    path = Path(raw_path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Metric file not found for {name!r}: {path}")
    return name, path


def summarize_metrics(
    metric_specs: list[str], output_dir: str | Path | None = None
) -> dict[str, Any]:
    """Create a comparable summary from independently evaluated methods."""
    if len(metric_specs) < 2:
        raise ValueError("Provide at least two --metric NAME=PATH values for a comparison.")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for specification in metric_specs:
        name, path = _parse_metric_spec(specification)
        if name in seen:
            raise ValueError(f"Duplicate method name in --metric: {name!r}")
        seen.add(name)
        metrics = read_json(path)
        rows.append(
            {
                "method": name,
                "metric_path": str(path),
                "num_samples": int(metrics["num_samples"]),
                "accuracy": float(metrics["accuracy"]),
                "macro_class_accuracy": float(metrics["macro_class_accuracy"]),
                "adapter_path": metrics.get("adapter_path"),
            }
        )
    sample_counts = {row["num_samples"] for row in rows}
    if len(sample_counts) != 1:
        raise ValueError(
            "Compared methods were evaluated on different sample counts: "
            f"{sorted(sample_counts)}. Re-run evaluation with one shared protocol."
        )
    best = max(rows, key=lambda row: row["accuracy"])
    summary = {
        "num_compared_methods": len(rows),
        "num_samples_per_method": rows[0]["num_samples"],
        "best_accuracy_method": best["method"],
        "methods": rows,
    }
    if output_dir is not None:
        directory = Path(output_dir).expanduser()
        write_json(directory / "comparison.json", summary)
        markdown = [
            "| Method | Samples | Accuracy | Macro class accuracy | Adapter |",
            "|---|---:|---:|---:|---|",
        ]
        for row in rows:
            adapter = str(row["adapter_path"] or "base model").replace("|", "\\|")
            markdown.append(
                f"| {row['method']} | {row['num_samples']} | {row['accuracy']:.4f} | "
                f"{row['macro_class_accuracy']:.4f} | {adapter} |"
            )
        report_path = directory / "comparison.md"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text("\n".join(markdown) + "\n", encoding="utf-8")
    return summary


def aggregate_seed_metrics(
    metric_specs: list[str], output_dir: str | Path | None = None
) -> dict[str, Any]:
    """Aggregate ``METHOD=metrics.json`` files from multiple random seeds.

    The method name must be stable across seeds, for example
    ``seed42_crlora=...`` and ``seed43_crlora=...`` are not accepted as one
    group. Use ``METHOD@SEED=PATH`` or ``METHOD:SEED=PATH``.
    """
    if not metric_specs:
        raise ValueError("Provide at least one --metric METHOD@SEED=PATH value.")
    grouped: dict[str, list[dict[str, Any]]] = {}
    for specification in metric_specs:
        name, path = _parse_metric_spec(specification)
        if "@" in name:
            method, seed = name.rsplit("@", maxsplit=1)
        elif ":" in name:
            method, seed = name.rsplit(":", maxsplit=1)
        else:
            raise ValueError(
                f"Seed metric name {name!r} must use METHOD@SEED=PATH, e.g. crlora@42=..."
            )
        if not method or not seed:
            raise ValueError(f"Invalid method/seed metric name: {name!r}")
        metrics = read_json(path)
        grouped.setdefault(method, []).append(
            {
                "seed": seed,
                "metric_path": str(path),
                "num_samples": int(metrics["num_samples"]),
                "accuracy": float(metrics["accuracy"]),
                "macro_class_accuracy": float(metrics["macro_class_accuracy"]),
            }
        )
    sample_counts = {
        row["num_samples"] for rows in grouped.values() for row in rows
    }
    if len(sample_counts) != 1:
        raise ValueError(
            "Seed runs were evaluated on different sample counts: "
            f"{sorted(sample_counts)}"
        )
    aggregate: list[dict[str, Any]] = []
    for method, rows in grouped.items():
        values = [row["accuracy"] for row in rows]
        macro_values = [row["macro_class_accuracy"] for row in rows]
        mean = sum(values) / len(values)
        macro_mean = sum(macro_values) / len(macro_values)
        variance = sum((value - mean) ** 2 for value in values) / max(1, len(values) - 1)
        macro_variance = sum((value - macro_mean) ** 2 for value in macro_values) / max(
            1, len(macro_values) - 1
        )
        aggregate.append(
            {
                "method": method,
                "num_seeds": len(rows),
                "seeds": [row["seed"] for row in rows],
                "num_samples": rows[0]["num_samples"],
                "accuracy_mean": mean,
                "accuracy_std": variance**0.5,
                "macro_class_accuracy_mean": macro_mean,
                "macro_class_accuracy_std": macro_variance**0.5,
                "runs": rows,
            }
        )
    aggregate.sort(key=lambda row: row["accuracy_mean"], reverse=True)
    first_run = next(iter(grouped.values()))[0]
    result = {
        "num_methods": len(aggregate),
        "num_samples_per_run": first_run["num_samples"],
        "methods": aggregate,
    }
    if output_dir is not None:
        directory = Path(output_dir).expanduser()
        write_json(directory / "seed_aggregate.json", result)
        markdown = [
            "| Method | Seeds | Accuracy mean | Accuracy std | "
            "Macro accuracy mean | Macro accuracy std |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for row in aggregate:
            markdown.append(
                f"| {row['method']} | {row['num_seeds']} | "
                f"{row['accuracy_mean']:.4f} | {row['accuracy_std']:.4f} | "
                f"{row['macro_class_accuracy_mean']:.4f} | "
                f"{row['macro_class_accuracy_std']:.4f} |"
            )
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "seed_aggregate.md").write_text(
            "\n".join(markdown) + "\n", encoding="utf-8"
        )
    return result
