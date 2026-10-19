from __future__ import annotations

from pathlib import Path

from .io_utils import read_json
from .schema import normalize_class_name


def load_evidence_bank(path: str | Path | None) -> dict[str, tuple[str, ...]]:
    if path is None:
        return {}
    raw = read_json(path)
    if not isinstance(raw, dict):
        raise ValueError("Evidence bank must be a JSON object keyed by class name.")
    bank: dict[str, tuple[str, ...]] = {}
    for class_name, value in raw.items():
        if isinstance(value, str):
            items = [value]
        elif isinstance(value, list):
            items = value
        else:
            raise ValueError(f"Evidence for {class_name!r} must be a string or list of strings.")
        normalized_items = tuple(str(item).strip() for item in items if str(item).strip())
        if normalized_items:
            bank[normalize_class_name(str(class_name))] = normalized_items
    return bank


def evidence_for_class(
    class_name: str,
    bank: dict[str, tuple[str, ...]],
    fallback_to_label: bool = True,
) -> tuple[str, ...]:
    normalized = normalize_class_name(class_name)
    if normalized in bank:
        return bank[normalized]
    return (normalized,) if fallback_to_label else ()
