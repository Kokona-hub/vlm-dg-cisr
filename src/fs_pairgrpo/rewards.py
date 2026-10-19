from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from typing import Any

from .parsing import parse_pair_prediction
from .schema import normalize_class_name

STYLE_WORDS = {
    "background",
    "color",
    "colour",
    "lighting",
    "photo",
    "photograph",
    "painting",
    "realistic",
    "rendering",
    "sketch",
    "style",
    "texture",
}
TOKEN_RE = re.compile(r"[\w\u4e00-\u9fff]+", flags=re.IGNORECASE)


def _repeat(values: Sequence[Any], target_length: int) -> list[Any]:
    if len(values) == target_length:
        return list(values)
    if not values or target_length % len(values) != 0:
        raise ValueError(
            f"Cannot align {len(values)} metadata rows with {target_length} completions."
        )
    factor = target_length // len(values)
    return [value for value in values for _ in range(factor)]


def _labels(labels: Sequence[str], count: int) -> list[str]:
    return [normalize_class_name(str(value)) for value in _repeat(labels, count)]


def _evidence_rows(values: Sequence[Iterable[str]], count: int) -> list[tuple[str, ...]]:
    return [tuple(str(item) for item in row) for row in _repeat(values, count)]


def paired_correctness_reward(
    prompts: list[Any],
    completions: list[Any],
    class_name: list[str],
    **_: Any,
) -> list[float]:
    """0, 0.5, or 1.0 according to how many paired views are correct."""
    del prompts
    labels = _labels(class_name, len(completions))
    rewards: list[float] = []
    for completion, label in zip(completions, labels, strict=True):
        prediction = parse_pair_prediction(completion)
        correct_1 = prediction.answer_1 == label
        correct_2 = prediction.answer_2 == label
        rewards.append(0.5 * (float(correct_1) + float(correct_2)))
    return rewards


def all_views_correct_reward(
    prompts: list[Any],
    completions: list[Any],
    class_name: list[str],
    **_: Any,
) -> list[float]:
    """The method's key reward: one only when both views are correct."""
    del prompts
    labels = _labels(class_name, len(completions))
    rewards: list[float] = []
    for completion, label in zip(completions, labels, strict=True):
        prediction = parse_pair_prediction(completion)
        rewards.append(float(prediction.answer_1 == label and prediction.answer_2 == label))
    return rewards


def strict_format_reward(
    prompts: list[Any],
    completions: list[Any],
    **_: Any,
) -> list[float]:
    del prompts
    return [float(parse_pair_prediction(completion).format_valid) for completion in completions]


def _content_tokens(text: str | None) -> set[str]:
    if not text:
        return set()
    return {token.lower() for token in TOKEN_RE.findall(text) if len(token) > 2}


def evidence_quality_reward(
    prompts: list[Any],
    completions: list[Any],
    evidence: list[list[str]],
    class_name: list[str],
    **_: Any,
) -> list[float]:
    """Reward evidence only after both paired predictions are correct."""
    del prompts
    references = _evidence_rows(evidence, len(completions))
    labels = _labels(class_name, len(completions))
    rewards: list[float] = []
    for completion, reference_items, label in zip(completions, references, labels, strict=True):
        prediction = parse_pair_prediction(completion)
        if prediction.answer_1 != label or prediction.answer_2 != label:
            rewards.append(0.0)
            continue
        evidence_1 = _content_tokens(prediction.evidence_1)
        evidence_2 = _content_tokens(prediction.evidence_2)
        reference = _content_tokens(" ".join(reference_items))
        if not evidence_1 or not evidence_2:
            rewards.append(0.0)
            continue
        cross_view_overlap = len(evidence_1 & evidence_2) / max(1, len(evidence_1 | evidence_2))
        reference_overlap = (
            len((evidence_1 | evidence_2) & reference) / max(1, len(reference))
            if reference
            else 0.0
        )
        style_penalty = float(bool((evidence_1 | evidence_2) & STYLE_WORDS))
        score = 0.5 * cross_view_overlap + 0.5 * reference_overlap - 0.5 * style_penalty
        rewards.append(max(0.0, min(1.0, score)))
    return rewards


REWARD_REGISTRY = {
    "paired_correctness": paired_correctness_reward,
    "all_views_correct": all_views_correct_reward,
    "format": strict_format_reward,
    "evidence_quality": evidence_quality_reward,
}


def reward_functions(names: Sequence[str]) -> list[Any]:
    unknown = sorted(set(names) - REWARD_REGISTRY.keys())
    if unknown:
        raise ValueError(f"Unknown reward functions: {unknown}")
    return [REWARD_REGISTRY[name] for name in names]
