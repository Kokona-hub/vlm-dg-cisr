import csv
from pathlib import Path

import pytest

from fs_pairgrpo.datasets import load_image_list, select_samples_by_image_list
from fs_pairgrpo.schema import ImageSample


def _sample(path: Path, domain: str = "real") -> ImageSample:
    return ImageSample(
        sample_id=path.stem,
        domain=domain,
        split="test",
        image_path=path,
        label_id=0,
        class_name="the eiffel tower",
    )


def _write_csv(path: Path, rows: list[list[object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["index", "image_path", "true_label", "raw_output", "pred_label", "is_correct"]
        )
        writer.writerows(rows)


def test_csv_image_list_matches_across_dataset_roots(tmp_path):
    list_path = tmp_path / "wrong.csv"
    _write_csv(
        list_path,
        [
            [160, "/root/autodl-tmp/data/DomainNet/real/The_Eiffel_Tower/real_309_000160.jpg", "The_Eiffel_Tower", "church", "church", 0],
            [169, "/root/autodl-tmp/data/DomainNet/real/The_Eiffel_Tower/real_309_000169.jpg", "The_Eiffel_Tower", "waterslide", "waterslide", 0],
            [3, "/root/autodl-tmp/data/DomainNet/real/laptop/real_001_000003.jpg", "laptop", "tv", "tv", 0],
        ],
    )
    selection = load_image_list(list_path)
    assert selection.entry_count == 3

    server_root = tmp_path / "big-disk" / "domain_net"
    samples = [
        _sample(server_root / "real" / "The_Eiffel_Tower" / "real_309_000160.jpg"),
        _sample(server_root / "real" / "The_Eiffel_Tower" / "real_309_000169.jpg"),
        _sample(server_root / "real" / "The_Eiffel_Tower" / "real_309_000170.jpg"),
        _sample(server_root / "real" / "laptop" / "real_001_000003.jpg"),
    ]

    selected, unmatched = select_samples_by_image_list(samples, selection)

    assert unmatched == 0
    assert {sample.image_path.name for sample in selected} == {
        "real_309_000160.jpg",
        "real_309_000169.jpg",
        "real_001_000003.jpg",
    }


def test_text_image_list_accepts_bare_filenames(tmp_path):
    list_path = tmp_path / "subset.txt"
    list_path.write_text("real_309_000160.jpg\n\nreal_309_999999.jpg\n", encoding="utf-8")
    selection = load_image_list(list_path)

    samples = [
        _sample(tmp_path / "domain_net" / "real" / "The_Eiffel_Tower" / "real_309_000160.jpg"),
        _sample(tmp_path / "domain_net" / "real" / "The_Eiffel_Tower" / "real_309_000170.jpg"),
    ]

    selected, unmatched = select_samples_by_image_list(samples, selection)

    assert len(selected) == 1
    assert unmatched == 1


def test_csv_without_image_path_column_is_rejected(tmp_path):
    list_path = tmp_path / "bad.csv"
    list_path.write_text("index,label\n1,cat\n", encoding="utf-8")

    with pytest.raises(ValueError, match="image path column"):
        load_image_list(list_path)


def test_empty_image_list_is_rejected(tmp_path):
    list_path = tmp_path / "empty.txt"
    list_path.write_text("\n\n", encoding="utf-8")

    with pytest.raises(ValueError, match="no usable image paths"):
        load_image_list(list_path)
