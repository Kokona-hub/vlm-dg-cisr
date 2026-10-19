from pathlib import Path

import yaml
from PIL import Image

from fs_pairgrpo.io_utils import read_jsonl
from fs_pairgrpo.manifest import build_manifests


def _write_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (20, 16), color).save(path)


def test_folder_prepare_materializes_cross_domain_pairs(tmp_path):
    root = tmp_path / "dataset"
    for domain, offset in (("photo", 0), ("art", 30)):
        _write_image(root / domain / "horse" / "1.jpg", (180 + offset, 80, 40))
        _write_image(root / domain / "dog" / "1.jpg", (40, 120 + offset, 180))

    config = {
        "seed": 7,
        "dataset": {
            "type": "folder",
            "root": str(root),
            "source_domains": ["photo", "art"],
            "target_domains": ["sketch"],
        },
        "fourier": {
            "views_per_sample": 1,
            "low_frequency_ratio": 0.08,
            "interpolation_range": [0.5, 0.8],
            "extrapolation_range": [1.0, 1.1],
            "extrapolation_probability": 0.5,
        },
        "output": {
            "manifest_dir": str(tmp_path / "manifests"),
            "styled_image_dir": str(tmp_path / "styled"),
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    summary = build_manifests(config_path)
    pairs = read_jsonl(tmp_path / "manifests" / "fourier_pairs.jsonl")

    assert summary["num_source_samples"] == 4
    assert summary["num_fourier_pairs"] == 4
    assert all(row["source_domain"] != row["donor_domain"] for row in pairs)
    assert all(Path(row["styled_path"]).is_file() for row in pairs)


def test_folder_scan_limits_classes_before_returning(tmp_path):
    from fs_pairgrpo.datasets import scan_folder_dataset

    root = tmp_path / "dataset"
    for domain in ("photo", "art"):
        for class_name in ("ant", "horse", "zebra"):
            _write_image(root / domain / class_name / "1.jpg", (100, 120, 140))

    samples = scan_folder_dataset(root, ["photo", "art"], max_classes=2)

    assert {sample.class_name for sample in samples} == {"ant", "horse"}


def test_prepare_can_write_online_pairs_without_image_inodes(tmp_path):
    root = tmp_path / "dataset"
    for domain, offset in (("photo", 0), ("art", 30)):
        _write_image(root / domain / "horse" / "1.jpg", (180 + offset, 80, 40))
    config = {
        "seed": 7,
        "dataset": {
            "type": "folder",
            "root": str(root),
            "source_domains": ["photo", "art"],
            "target_domains": ["sketch"],
        },
        "fourier": {
            "materialize": False,
            "views_per_sample": 1,
            "low_frequency_ratio": 0.08,
            "interpolation_range": [0.5, 0.8],
            "extrapolation_range": [1.0, 1.1],
            "extrapolation_probability": 0.5,
        },
        "output": {
            "manifest_dir": str(tmp_path / "manifests"),
            "styled_image_dir": str(tmp_path / "styled"),
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    summary = build_manifests(config_path)
    pairs = read_jsonl(tmp_path / "manifests" / "fourier_pairs.jsonl")

    assert summary["num_fourier_pairs"] == 2
    assert not (tmp_path / "styled").exists()
    assert all(not row["materialized"] for row in pairs)
    assert all(Path(row["donor_path"]).is_file() for row in pairs)
