from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .schema import normalize_class_name


def _tag(text: str, name: str) -> str | None:
    match = re.search(rf"<{name}>\s*(.*?)\s*</{name}>", text, flags=re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else None


def completion_to_text(completion: Any) -> str:
    if isinstance(completion, str):
        return completion
    if isinstance(completion, list):
        chunks: list[str] = []
        for message in completion:
            if not isinstance(message, dict):
                chunks.append(str(message))
                continue
            content = message.get("content", "")
            if isinstance(content, str):
                chunks.append(content)
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        chunks.append(str(part.get("text", "")))
        return "".join(chunks)
    return str(completion)


@dataclass(frozen=True)
class PairPrediction:
    answer_1: str | None
    answer_2: str | None
    evidence_1: str | None
    evidence_2: str | None

    @property
    def format_valid(self) -> bool:
        return self.answer_1 is not None and self.answer_2 is not None


def parse_pair_prediction(completion: Any) -> PairPrediction:
    text = completion_to_text(completion)
    answer_1 = _tag(text, "ANSWER_1")
    answer_2 = _tag(text, "ANSWER_2")
    return PairPrediction(
        answer_1=normalize_class_name(answer_1) if answer_1 else None,
        answer_2=normalize_class_name(answer_2) if answer_2 else None,
        evidence_1=_tag(text, "EVIDENCE_1"),
        evidence_2=_tag(text, "EVIDENCE_2"),
    )


def parse_single_prediction(completion: Any) -> tuple[str | None, str | None]:
    text = completion_to_text(completion)
    answer = _tag(text, "ANSWER")
    return (normalize_class_name(answer) if answer else None, _tag(text, "EVIDENCE"))
