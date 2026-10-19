from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


def normalize_class_name(value: str) -> str:
    return " ".join(value.strip().replace("_", " ").replace("-", " ").lower().split())


@dataclass(frozen=True)
class ImageSample:
    sample_id: str
    domain: str
    split: str
    image_path: Path
    label_id: int
    class_name: str
    evidence: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["image_path"] = str(self.image_path)
        value["evidence"] = list(self.evidence)
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ImageSample:
        return cls(
            sample_id=str(value["sample_id"]),
            domain=str(value["domain"]),
            split=str(value.get("split", "train")),
            image_path=Path(value["image_path"]),
            label_id=int(value["label_id"]),
            class_name=normalize_class_name(str(value["class_name"])),
            evidence=tuple(
                str(item).strip() for item in value.get("evidence", []) if str(item).strip()
            ),
        )


@dataclass(frozen=True)
class StylePair:
    pair_id: str
    source: ImageSample
    donor: ImageSample
    styled_path: Path
    alpha: float
    low_frequency_ratio: float
    seed: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "pair_id": self.pair_id,
            "sample_id": self.source.sample_id,
            "source_domain": self.source.domain,
            "donor_domain": self.donor.domain,
            "class_name": self.source.class_name,
            "label_id": self.source.label_id,
            "evidence": list(self.source.evidence),
            "original_path": str(self.source.image_path),
            "donor_path": str(self.donor.image_path),
            "styled_path": str(self.styled_path),
            "materialized": self.styled_path.is_file(),
            "alpha": self.alpha,
            "low_frequency_ratio": self.low_frequency_ratio,
            "seed": self.seed,
        }
