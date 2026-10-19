from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .io_utils import read_jsonl
from .prompts import grpo_prompt, sft_completion, sft_prompt


def load_class_names(source_manifest: str | Path) -> list[str]:
    return sorted({str(row["class_name"]) for row in read_jsonl(source_manifest)})


def build_sft_dataset(
    source_manifest: str | Path,
    class_names: Sequence[str],
    use_evidence: bool,
    pair_manifest: str | Path | None = None,
    include_fourier: bool = False,
):
    try:
        from datasets import Dataset, Image
        from datasets import Sequence as ImageSequence
    except ImportError as exc:
        raise RuntimeError(
            "Install the training dependencies with `pip install -e .[train]`."
        ) from exc
    rows: list[dict[str, Any]] = []
    for row in read_jsonl(source_manifest):
        rows.append(
            {
                "images": [str(row["image_path"])],
                "prompt": sft_prompt(
                    class_names=class_names,
                    use_evidence=use_evidence,
                ),
                "completion": sft_completion(
                    class_name=str(row["class_name"]),
                    evidence=list(row.get("evidence", [])),
                    use_evidence=use_evidence,
                ),
                "sample_id": str(row["sample_id"]),
                "class_name": str(row["class_name"]),
                "domain": str(row["domain"]),
            }
        )
    if include_fourier:
        if pair_manifest is None:
            raise ValueError("Pair-SFT requires data.pair_manifest.")
        for row in read_jsonl(pair_manifest):
            if not bool(row.get("materialized", False)):
                raise ValueError(
                    "Pair-SFT currently requires materialized Fourier images. Set "
                    "fourier.materialize=true for this ablation. Independent PairGRPO supports "
                    "online Fourier with materialize=false."
                )
            rows.append(
                {
                    "images": [str(row["styled_path"])],
                    "prompt": sft_prompt(class_names=class_names, use_evidence=use_evidence),
                    "completion": sft_completion(
                        class_name=str(row["class_name"]),
                        evidence=list(row.get("evidence", [])),
                        use_evidence=use_evidence,
                    ),
                    "sample_id": f"fourier:{row['pair_id']}",
                    "class_name": str(row["class_name"]),
                    "domain": f"fourier:{row['source_domain']}",
                }
            )
    return Dataset.from_list(rows).cast_column("images", ImageSequence(Image()))


def build_grpo_dataset(
    pair_manifest: str | Path,
    class_names: Sequence[str],
    use_evidence: bool,
):
    try:
        from datasets import Dataset, Image
    except ImportError as exc:
        raise RuntimeError(
            "Install the training dependencies with `pip install -e .[train]`."
        ) from exc
    rows: list[dict[str, Any]] = []
    for row in read_jsonl(pair_manifest):
        rows.append(
            {
                # Views remain separate. The trainer performs two independent single-image
                # rollouts and couples them only when computing the terminal reward.
                "original_image": str(row["original_path"]),
                "styled_image": str(row["styled_path"]),
                "prompt": grpo_prompt(class_names=class_names, use_evidence=use_evidence),
                "pair_id": str(row["pair_id"]),
                "class_name": str(row["class_name"]),
                "evidence": list(row.get("evidence", [])),
                "source_domain": str(row["source_domain"]),
                "donor_domain": str(row["donor_domain"]),
                "alpha": float(row["alpha"]),
            }
        )
    dataset = Dataset.from_list(rows)
    dataset = dataset.cast_column("original_image", Image())
    return dataset.cast_column("styled_image", Image())
