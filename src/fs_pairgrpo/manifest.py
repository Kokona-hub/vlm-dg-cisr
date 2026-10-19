from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from .datasets import balanced_limit, read_domainnet_split, scan_folder_dataset
from .evidence import load_evidence_bank
from .fourier import build_style_pairs
from .io_utils import read_jsonl, read_yaml, write_json, write_jsonl
from .schema import ImageSample


def validate_domain_protocol(config: dict[str, Any]) -> None:
    dataset = config["dataset"]
    sources = list(dataset["source_domains"])
    targets = list(dataset.get("target_domains", []))
    if len(sources) not in {2, 3}:
        raise ValueError(f"FS-PairGRPO requires exactly 2 or 3 source domains; got {sources}.")
    if len(set(sources)) != len(sources):
        raise ValueError(f"Source domains must be unique; got {sources}.")
    overlap = sorted(set(sources) & set(targets))
    if overlap:
        raise ValueError(f"Target domains must be unseen during training; overlap: {overlap}.")
    if len(targets) != 1:
        raise ValueError(f"FS-PairGRPO requires exactly one unseen target domain; got {targets}.")


def audit_training_manifests(config: dict[str, Any], require_pairs: bool) -> dict[str, int]:
    """Fail closed if a training manifest contains target or undeclared-domain data."""
    validate_domain_protocol(config)
    declared_sources = set(config["dataset"]["source_domains"])
    targets = set(config["dataset"]["target_domains"])
    # read_jsonl yields a streaming generator; materialize it here because the
    # audit reports counts and iterates over the rows more than once.
    source_rows = list(read_jsonl(config["data"]["source_manifest"]))
    if not source_rows:
        raise ValueError("The source training manifest is empty.")
    observed_sources = {str(row["domain"]) for row in source_rows}
    forbidden = observed_sources - declared_sources
    if forbidden or observed_sources & targets:
        raise ValueError(f"Source manifest contains forbidden domains: {sorted(forbidden)}")
    if observed_sources != declared_sources:
        raise ValueError(
            "Source manifest does not contain every declared source domain; "
            f"observed={sorted(observed_sources)}, declared={sorted(declared_sources)}"
        )

    label_names: dict[int, set[str]] = {}
    for row in source_rows:
        label_names.setdefault(int(row["label_id"]), set()).add(str(row["class_name"]))
    inconsistent = {label: sorted(names) for label, names in label_names.items() if len(names) != 1}
    if inconsistent:
        raise ValueError(f"A label ID maps to multiple class names: {inconsistent}")

    pair_count = 0
    if require_pairs:
        pair_rows = list(read_jsonl(config["data"]["pair_manifest"]))
        if not pair_rows:
            raise ValueError("The Fourier pair manifest is empty.")
        pair_count = len(pair_rows)
        for row in pair_rows:
            source_domain = str(row["source_domain"])
            donor_domain = str(row["donor_domain"])
            if source_domain not in declared_sources or donor_domain not in declared_sources:
                raise ValueError(
                    f"Pair {row.get('pair_id')} contains an undeclared or target domain: "
                    f"{source_domain}, {donor_domain}"
                )
            if source_domain == donor_domain:
                raise ValueError(f"Pair {row.get('pair_id')} does not cross source domains.")
            original_path = Path(row["original_path"])
            donor_path = Path(row["donor_path"])
            if not original_path.is_file() or not donor_path.is_file():
                raise FileNotFoundError(
                    f"Pair {row.get('pair_id')} is missing its source or donor image."
                )
            if bool(row.get("materialized", False)) and not Path(row["styled_path"]).is_file():
                raise FileNotFoundError(
                    f"Pair {row.get('pair_id')} declares a materialized image that is missing: "
                    f"{row['styled_path']}"
                )
    return {"source_samples": len(source_rows), "fourier_pairs": pair_count}


def load_samples_from_config(config: dict[str, Any], split: str) -> list[ImageSample]:
    validate_domain_protocol(config)
    dataset = config["dataset"]
    evidence_bank = load_evidence_bank(dataset.get("evidence_bank"))
    dataset_type = str(dataset.get("type", "domainnet")).lower()
    domains = list(
        dataset["source_domains"] if split == "train" else dataset.get("target_domains", [])
    )
    if dataset_type == "domainnet":
        samples = read_domainnet_split(
            root=dataset["root"],
            splits_root=dataset["splits_root"],
            domains=domains,
            split=split,
            evidence_bank=evidence_bank,
            require_images=bool(dataset.get("require_images", True)),
        )
    elif dataset_type == "folder":
        samples = scan_folder_dataset(
            root=dataset["root"],
            domains=domains,
            split=split,
            evidence_bank=evidence_bank,
            max_classes=dataset.get("max_classes"),
        )
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")
    max_classes = dataset.get("max_classes")
    if max_classes is not None:
        max_classes = int(max_classes)
        if max_classes < 1:
            raise ValueError("dataset.max_classes must be positive when provided.")
        selected_classes = set(sorted({sample.class_name for sample in samples})[:max_classes])
        samples = [sample for sample in samples if sample.class_name in selected_classes]
    return balanced_limit(
        samples,
        max_per_domain_class=dataset.get("max_per_domain_class"),
        # Keep the sampled image subset fixed across random-seed runs unless
        # the experiment explicitly asks for a different selection seed.
        seed=int(dataset.get("selection_seed", config.get("seed", 42))),
    )


def build_manifests(config_path: str | Path, dry_run: bool = False) -> dict[str, Any]:
    config_path = Path(config_path).expanduser().resolve()
    config = read_yaml(config_path)
    validate_domain_protocol(config)
    samples = load_samples_from_config(config, split="train")
    if not samples:
        raise ValueError("No source samples were found for the configured source domains.")
    domain_label_sets = {
        domain: {sample.label_id for sample in samples if sample.domain == domain}
        for domain in config["dataset"]["source_domains"]
    }
    empty_domains = [domain for domain, labels in domain_label_sets.items() if not labels]
    if empty_domains:
        raise ValueError(f"No source samples were found for domains: {empty_domains}")
    shared_labels = set.intersection(*domain_label_sets.values())
    union_labels = set.union(*domain_label_sets.values())
    if shared_labels != union_labels:
        missing = {
            domain: sorted(union_labels - labels)
            for domain, labels in domain_label_sets.items()
            if labels != union_labels
        }
        raise ValueError(
            "Source domains must share the same closed-set label IDs before Fourier pairing; "
            f"missing labels by domain: {missing}"
        )
    output = config["output"]
    manifest_dir = Path(output["manifest_dir"]).expanduser().resolve()
    style = config["fourier"]
    pairs = build_style_pairs(
        samples=samples,
        output_dir=output["styled_image_dir"],
        views_per_sample=int(style.get("views_per_sample", 1)),
        interpolation_range=tuple(float(x) for x in style.get("interpolation_range", [0.3, 1.0])),
        extrapolation_range=tuple(float(x) for x in style.get("extrapolation_range", [1.0, 1.2])),
        extrapolation_probability=float(style.get("extrapolation_probability", 0.5)),
        low_frequency_ratio=float(style.get("low_frequency_ratio", 0.08)),
        seed=int(config.get("seed", 42)),
        materialize=bool(style.get("materialize", True)) and not dry_run,
        overwrite=bool(style.get("overwrite", False)),
    )
    manifest_dir.mkdir(parents=True, exist_ok=True)
    source_path = manifest_dir / "source_train.jsonl"
    pair_path = manifest_dir / "fourier_pairs.jsonl"
    write_jsonl(source_path, (sample.to_dict() for sample in samples))
    write_jsonl(pair_path, (pair.to_dict() for pair in pairs))
    summary = {
        "config": str(config_path),
        "dry_run": dry_run,
        "source_domains": list(config["dataset"]["source_domains"]),
        "target_domains": list(config["dataset"].get("target_domains", [])),
        "num_source_samples": len(samples),
        "num_fourier_pairs": len(pairs),
        "domain_counts": dict(Counter(sample.domain for sample in samples)),
        "class_count": len({sample.class_name for sample in samples}),
        "source_manifest": str(source_path),
        "pair_manifest": str(pair_path),
    }
    write_json(manifest_dir / "manifest_summary.json", summary)
    return summary
