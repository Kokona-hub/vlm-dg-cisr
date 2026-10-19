from __future__ import annotations

from collections.abc import Sequence

SYSTEM_PROMPT = (
    "You are a strict visual classifier. Use object shape, parts, and structure rather than "
    "background, color, rendering style, or image domain. Follow the requested output format "
    "exactly."
)


def _candidate_text(class_names: Sequence[str]) -> str:
    return ", ".join(class_names)


def single_image_prompt(class_names: Sequence[str], use_evidence: bool = True) -> str:
    if use_evidence:
        output = (
            "Return exactly: <EVIDENCE>short class-relevant visual evidence</EVIDENCE>"
            "<ANSWER>one candidate label</ANSWER>."
        )
    else:
        output = "Return exactly: <ANSWER>one candidate label</ANSWER>."
    return (
        "Classify the object in the image. Choose exactly one label from the candidate list. "
        f"Candidates: [{_candidate_text(class_names)}]. {output}"
    )


def single_target(class_name: str, evidence: Sequence[str], use_evidence: bool = True) -> str:
    if not use_evidence:
        return f"<ANSWER>{class_name}</ANSWER>"
    text = ", ".join(evidence)
    return f"<EVIDENCE>{text}</EVIDENCE><ANSWER>{class_name}</ANSWER>"


def sft_messages(
    class_names: Sequence[str],
    class_name: str,
    evidence: Sequence[str],
    use_evidence: bool = True,
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": single_image_prompt(class_names, use_evidence)},
        {"role": "assistant", "content": single_target(class_name, evidence, use_evidence)},
    ]


def sft_prompt(class_names: Sequence[str], use_evidence: bool = True) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": single_image_prompt(class_names, use_evidence)},
    ]


def sft_completion(
    class_name: str, evidence: Sequence[str], use_evidence: bool = True
) -> list[dict[str, str]]:
    return [
        {
            "role": "assistant",
            "content": single_target(class_name, evidence, use_evidence),
        }
    ]


def grpo_prompt(class_names: Sequence[str], use_evidence: bool = True) -> list[dict[str, str]]:
    """Return the same single-image contract used by SFT and target evaluation."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": single_image_prompt(class_names, use_evidence)},
    ]
