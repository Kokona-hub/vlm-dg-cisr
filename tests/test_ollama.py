from pathlib import Path

import fs_pairgrpo.ollama_eval as ollama_eval
from fs_pairgrpo.ollama_eval import (
    _build_metrics,
    _chat_with_retries,
    _image_base64,
    _parse_ollama_answer,
)


def test_image_base64_reads_local_image(tmp_path):
    image_path = tmp_path / "sample.jpg"
    image_path.write_bytes(b"fake-image-bytes")
    assert _image_base64(Path(image_path)) == "ZmFrZS1pbWFnZS1ieXRlcw=="


def test_ollama_answer_falls_back_to_a_candidate_label():
    answer, evidence = _parse_ollama_answer("The object is a horse.", ["horse", "dog"])
    assert answer == "horse"
    assert evidence is None


def test_ollama_chat_retries_after_a_transient_failure(monkeypatch, tmp_path):
    attempts = 0

    def fake_chat(**_kwargs):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise RuntimeError("temporary timeout")
        return "<ANSWER>horse</ANSWER>"

    monkeypatch.setattr(ollama_eval, "_chat", fake_chat)
    output, used_attempts = _chat_with_retries(
        base_url="http://127.0.0.1:11435",
        model="qwen3-vl:8b",
        image_path=tmp_path / "sample.jpg",
        prompt="classify",
        timeout=1,
        max_retries=2,
        retry_backoff=0,
    )

    assert output == "<ANSWER>horse</ANSWER>"
    assert used_attempts == 2


def test_partial_metrics_report_failed_and_pending_samples():
    rows = {
        "ok": {
            "sample_id": "ok",
            "domain": "real",
            "class_name": "horse",
            "prediction": "horse",
            "correct": True,
            "error": None,
        },
        "failed": {
            "sample_id": "failed",
            "domain": "real",
            "class_name": "dog",
            "prediction": None,
            "correct": False,
            "error": "timeout",
        },
    }

    metrics = _build_metrics(
        rows,
        model_name="qwen3-vl:8b",
        base_url="http://127.0.0.1:11435",
        total_samples=3,
    )

    assert metrics["num_samples"] == 1
    assert metrics["num_failed"] == 1
    assert metrics["num_pending"] == 1
    assert metrics["complete"] is False
    assert metrics["accuracy"] == 1.0
