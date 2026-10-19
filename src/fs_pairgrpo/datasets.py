from __future__ import annotations

import csv
import hashlib
import random
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from .evidence import evidence_for_class
from .schema import ImageSample, normalize_class_name

DOMAINNET_DOMAINS = ("clipart", "infograph", "painting", "quickdraw", "real", "sketch")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Column names accepted when the image list is a CSV (for example a
# ``predictions.csv`` exported from an earlier evaluation run).
IMAGE_LIST_CSV_COLUMNS = ("image_path", "path", "file", "filename", "image")


def _stable_id(*parts: str) -> str:
    text = "|".join(parts)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:16]


def _class_from_relative_path(relative_path: str) -> str:
    parts = relative_path.replace("\\", "/").split("/")
    if len(parts) < 2:
        raise ValueError(f"Cannot infer class name from {relative_path!r}")
    return normalize_class_name(parts[-2])


def read_domainnet_split(
    root: str | Path,
    splits_root: str | Path,
    domains: Iterable[str],
    split: str,
    evidence_bank: dict[str, tuple[str, ...]] | None = None,
    require_images: bool = True,
) -> list[ImageSample]:
    root = Path(root).expanduser().resolve()
    splits_root = Path(splits_root).expanduser().resolve()
    evidence_bank = evidence_bank or {}
    samples: list[ImageSample] = []
    for domain in domains:
        if domain not in DOMAINNET_DOMAINS:
            raise ValueError(f"Unknown DomainNet domain: {domain}")
        split_path = splits_root / f"{domain}_{split}.txt"
        if not split_path.exists():
            raise FileNotFoundError(f"Missing DomainNet split: {split_path}")
        with split_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    relative_path, label_text = line.rsplit(maxsplit=1)
                except ValueError as exc:
                    raise ValueError(f"Malformed split line {split_path}:{line_number}") from exc
                image_path = root / relative_path
                if require_images and not image_path.exists():
                    raise FileNotFoundError(
                        f"Image referenced by split does not exist: {image_path}"
                    )
                class_name = _class_from_relative_path(relative_path)
                samples.append(
                    ImageSample(
                        sample_id=_stable_id(domain, split, relative_path),
                        domain=domain,
                        split=split,
                        image_path=image_path,
                        label_id=int(label_text),
                        class_name=class_name,
                        evidence=evidence_for_class(
                            class_name, evidence_bank, fallback_to_label=False
                        ),
                    )
                )
    return samples


def scan_folder_dataset(
    root: str | Path,
    domains: Iterable[str],
    split: str = "train",
    evidence_bank: dict[str, tuple[str, ...]] | None = None,
    max_classes: int | None = None,
) -> list[ImageSample]:
    """Scan root/domain/class/image for PACS, OfficeHome, or similarly arranged data."""
    root = Path(root).expanduser().resolve()
    evidence_bank = evidence_bank or {}
    domains = list(domains)
    for domain in domains:
        domain_root = root / domain
        if not domain_root.is_dir():
            raise FileNotFoundError(f"Missing domain directory: {domain_root}")
    class_names = sorted(
        {
            normalize_class_name(path.name)
            for domain in domains
            for path in (root / domain).iterdir()
            if path.is_dir()
        }
    )
    if max_classes is not None:
        if max_classes < 1:
            raise ValueError("max_classes must be positive when provided.")
        class_names = class_names[:max_classes]
    selected_classes = set(class_names)
    label_map = {name: index for index, name in enumerate(class_names)}
    samples: list[ImageSample] = []
    for domain in domains:
        domain_root = root / domain
        for class_dir in sorted(path for path in domain_root.iterdir() if path.is_dir()):
            class_name = normalize_class_name(class_dir.name)
            if class_name not in selected_classes:
                continue
            for image_path in sorted(class_dir.rglob("*")):
                if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_SUFFIXES:
                    continue
                relative_path = image_path.relative_to(root).as_posix()
                samples.append(
                    ImageSample(
                        sample_id=_stable_id(domain, split, relative_path),
                        domain=domain,
                        split=split,
                        image_path=image_path,
                        label_id=label_map[class_name],
                        class_name=class_name,
                        evidence=evidence_for_class(
                            class_name, evidence_bank, fallback_to_label=False
                        ),
                    )
                )
    return samples


def stratified_train_validation_split(
    samples: list[ImageSample],
    validation_fraction: float,
    seed: int,
) -> tuple[list[ImageSample], list[ImageSample]]:
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in [0, 1).")
    if validation_fraction == 0.0:
        return list(samples), []
    grouped: dict[tuple[str, int], list[ImageSample]] = defaultdict(list)
    for sample in samples:
        grouped[(sample.domain, sample.label_id)].append(sample)
    rng = random.Random(seed)
    train: list[ImageSample] = []
    validation: list[ImageSample] = []
    for group in grouped.values():
        rng.shuffle(group)
        count = max(1, round(len(group) * validation_fraction)) if len(group) > 1 else 0
        validation.extend(group[:count])
        train.extend(group[count:])
    return train, validation


def balanced_limit(
    samples: list[ImageSample],
    max_per_domain_class: int | None,
    seed: int,
) -> list[ImageSample]:
    if max_per_domain_class is None:
        return list(samples)
    if max_per_domain_class < 1:
        raise ValueError("max_per_domain_class must be positive.")
    grouped: dict[tuple[str, int], list[ImageSample]] = defaultdict(list)
    for sample in samples:
        grouped[(sample.domain, sample.label_id)].append(sample)
    rng = random.Random(seed)
    selected: list[ImageSample] = []
    for group in grouped.values():
        rng.shuffle(group)
        selected.extend(group[:max_per_domain_class])
    return selected


@dataclass(frozen=True)
class ImageListSelection:
    """Image paths that must be evaluated, keyed for root-independent matching.

    ``keys`` holds the trailing ``domain/class/file`` components of entries that
    carry at least three path components; ``names`` holds bare file names for the
    remaining entries. Comparing on those two short forms keeps a list produced
    on one machine (for example ``/root/autodl-tmp/data/DomainNet/...``) usable
    against a config that mounts the same dataset under a different root.
    """

    keys: frozenset[str]
    names: frozenset[str]

    @property
    def entry_count(self) -> int:
        return len(self.keys) + len(self.names)

    def __bool__(self) -> bool:
        return self.entry_count > 0


def _path_tail_key(parts: list[str]) -> str:
    if len(parts) >= 3:
        return "/".join(parts[-3:])
    return parts[-1] if parts else ""


def _split_path(value: str) -> list[str]:
    cleaned = value.strip().strip('"').strip("'").replace("\\", "/")
    return [part for part in cleaned.split("/") if part and part != "."]


def load_image_list(path: str | Path) -> ImageListSelection:
    """Read the image subset to evaluate from a CSV file or a plain path list."""
    path = Path(path).expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"Image list does not exist: {path}")
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            fieldnames = [name for name in (reader.fieldnames or []) if name]
            column = next(
                (
                    name
                    for name in fieldnames
                    if name.strip().lower() in IMAGE_LIST_CSV_COLUMNS
                ),
                None,
            )
            if column is None:
                raise ValueError(
                    f"CSV image list {path} has no image path column; expected one of "
                    f"{list(IMAGE_LIST_CSV_COLUMNS)}, found {fieldnames}."
                )
            raw_paths = [row.get(column, "") for row in reader]
    else:
        raw_paths = path.read_text(encoding="utf-8-sig").splitlines()

    keys: set[str] = set()
    names: set[str] = set()
    for raw in raw_paths:
        parts = _split_path(str(raw))
        if not parts:
            continue
        if len(parts) >= 3:
            keys.add(_path_tail_key(parts))
        else:
            names.add(parts[-1])
    selection = ImageListSelection(keys=frozenset(keys), names=frozenset(names))
    if not selection:
        raise ValueError(f"Image list {path} contains no usable image paths.")
    return selection


def select_samples_by_image_list(
    samples: Iterable[ImageSample], selection: ImageListSelection
) -> tuple[list[ImageSample], int]:
    """Return only the listed samples plus the count of unmatched list entries."""
    selected: list[ImageSample] = []
    matched: set[str] = set()
    for sample in samples:
        key = _path_tail_key(_split_path(sample.image_path.as_posix()))
        if key and key in selection.keys:
            matched.add(key)
            selected.append(sample)
            continue
        name = sample.image_path.name
        if name in selection.names:
            matched.add(name)
            selected.append(sample)
    return selected, selection.entry_count - len(matched)
