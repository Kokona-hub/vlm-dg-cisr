from __future__ import annotations

import base64
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from tqdm import tqdm

from .datasets import ImageListSelection, load_image_list, select_samples_by_image_list
from .io_utils import read_jsonl, read_yaml, write_json
from .manifest import load_samples_from_config, validate_domain_protocol
from .parsing import parse_single_prediction
from .prompts import SYSTEM_PROMPT, single_image_prompt
from .schema import normalize_class_name


def _image_base64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _chat(
    base_url: str,
    model: str,
    image_path: Path,
    prompt: str,
    timeout: float,
) -> str:
    payload = {
        "model": model,
        "stream": False,
        "think": False,
        "options": {"temperature": 0},
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": prompt,
                "images": [_image_base64(image_path)],
            },
        ],
    }
    request = Request(
        f"{base_url.rstrip('/')}/api/chat",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError) as exc:
        raise RuntimeError(f"Ollama request failed at {base_url!r}: {exc}") from exc
    content = result.get("message", {}).get("content")
    if not isinstance(content, str):
        raise RuntimeError(f"Ollama returned no text content: {result}")
    return content


def _chat_with_retries(
    *,
    base_url: str,
    model: str,
    image_path: Path,
    prompt: str,
    timeout: float,
    max_retries: int,
    retry_backoff: float,
) -> tuple[str, int]:
    last_error: RuntimeError | None = None
    total_attempts = max_retries + 1
    for attempt in range(1, total_attempts + 1):
        try:
            return (
                _chat(
                    base_url=base_url,
                    model=model,
                    image_path=image_path,
                    prompt=prompt,
                    timeout=timeout,
                ),
                attempt,
            )
        except RuntimeError as exc:
            last_error = exc
            if attempt < total_attempts:
                time.sleep(retry_backoff * attempt)
    assert last_error is not None
    raise RuntimeError(
        f"Ollama failed after {total_attempts} attempts: {last_error}"
    ) from last_error


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _build_metrics(
    rows_by_id: dict[str, dict[str, Any]],
    *,
    model_name: str,
    base_url: str,
    total_samples: int,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    successful = [row for row in rows_by_id.values() if not row.get("error")]
    failed = [row for row in rows_by_id.values() if row.get("error")]
    domain_total: Counter[str] = Counter()
    domain_correct: Counter[str] = Counter()
    class_total: Counter[str] = Counter()
    class_correct: Counter[str] = Counter()
    for row in successful:
        domain_total[row["domain"]] += 1
        class_total[row["class_name"]] += 1
        if row["correct"]:
            domain_correct[row["domain"]] += 1
            class_correct[row["class_name"]] += 1
    domain_accuracy = {
        domain: domain_correct[domain] / total for domain, total in sorted(domain_total.items())
    }
    class_accuracy = {
        name: class_correct[name] / total for name, total in sorted(class_total.items())
    }
    metrics = {
        "backend": "ollama",
        "model": model_name,
        "base_url": base_url,
        "num_samples": len(successful),
        "num_failed": len(failed),
        "num_pending": max(0, total_samples - len(successful) - len(failed)),
        "total_samples": total_samples,
        "complete": len(successful) == total_samples,
        "accuracy": sum(row["correct"] for row in successful) / max(1, len(successful)),
        "macro_class_accuracy": sum(class_accuracy.values()) / max(1, len(class_accuracy)),
        "domain_accuracy": domain_accuracy,
        "class_accuracy": class_accuracy,
    }
    if extra:
        metrics.update(extra)
    return metrics


def _parse_ollama_answer(raw_output: str, class_names: list[str]) -> tuple[str | None, str | None]:
    answer, evidence = parse_single_prediction(raw_output)
    if answer is not None:
        return answer, evidence
    normalized_output = normalize_class_name(raw_output)
    for candidate in sorted(class_names, key=len, reverse=True):
        normalized_candidate = normalize_class_name(candidate)
        if normalized_candidate and normalized_candidate in normalized_output:
            return normalized_candidate, evidence
    return None, evidence


def _class_names(config: dict[str, Any]) -> list[str]:
    source_manifest = config.get("data", {}).get("source_manifest")
    if source_manifest and Path(source_manifest).is_file():
        from .training_data import load_class_names

        return load_class_names(source_manifest)
    return sorted({sample.class_name for sample in load_samples_from_config(config, "train")})


def run_ollama_evaluation(
    config_path: str | Path,
    model_override: str | None = None,
    base_url_override: str | None = None,
    image_list_override: str | None = None,
) -> dict[str, Any]:
    config = read_yaml(config_path)
    validate_domain_protocol(config)
    ollama_cfg = config.get("ollama", {})
    eval_cfg = config["evaluation"]
    model_name = str(model_override or ollama_cfg.get("model", "qwen3-vl:8b"))
    base_url = str(base_url_override or ollama_cfg.get("base_url", "http://127.0.0.1:11434"))
    timeout = float(ollama_cfg.get("timeout", 300))
    max_retries = max(0, int(ollama_cfg.get("max_retries", 2)))
    retry_backoff = max(0.0, float(ollama_cfg.get("retry_backoff", 5)))
    max_consecutive_failures = max(1, int(ollama_cfg.get("max_consecutive_failures", 3)))
    checkpoint_every = max(1, int(ollama_cfg.get("checkpoint_every", 10)))
    resume = bool(ollama_cfg.get("resume", True))
    use_evidence = bool(config.get("use_evidence", True))
    class_names = _class_names(config)
    samples = load_samples_from_config(config, split=str(eval_cfg.get("split", "test")))
    total_target_samples = len(samples)
    image_list_value = image_list_override or eval_cfg.get("image_list")
    selection: ImageListSelection | None = None
    unmatched_list_entries = 0
    if image_list_value:
        selection = load_image_list(str(image_list_value))
        samples, unmatched_list_entries = select_samples_by_image_list(samples, selection)
        if not samples:
            raise ValueError(
                f"None of the {selection.entry_count} images in {image_list_value!r} matched the "
                f"{total_target_samples} '{eval_cfg.get('split', 'test')}' samples of the "
                f"configured target domain(s) {list(config['dataset'].get('target_domains', []))}."
            )
    max_samples = eval_cfg.get("max_samples")
    if max_samples is not None:
        samples = samples[: int(max_samples)]

    configured_output_dir = str(eval_cfg.get("ollama_output_dir", f"{eval_cfg['output_dir']}_ollama"))
    if image_list_value:
        configured_output_dir = f"{configured_output_dir}_subset"
    output_dir = Path(configured_output_dir)
    extra_metrics = {
        "target_split_num_samples": total_target_samples,
        "image_list": str(image_list_value) if image_list_value else None,
        "image_list_entries": selection.entry_count if selection else None,
        "image_list_unmatched": unmatched_list_entries if selection else None,
        "output_dir": str(output_dir),
    }
    predictions_path = output_dir / "predictions.jsonl"
    metrics_path = output_dir / "metrics.json"
    rows_by_id: dict[str, dict[str, Any]] = {}
    if resume and predictions_path.is_file():
        rows_by_id = {str(row["sample_id"]): row for row in read_jsonl(predictions_path)}
    elif predictions_path.exists():
        predictions_path.unlink()
    selected_ids = {sample.sample_id for sample in samples}
    rows_by_id = {
        sample_id: row for sample_id, row in rows_by_id.items() if sample_id in selected_ids
    }
    completed_ids = {
        sample_id for sample_id, row in rows_by_id.items() if not row.get("error")
    }
    pending_samples = [sample for sample in samples if sample.sample_id not in completed_ids]

    prompt = single_image_prompt(class_names, use_evidence=use_evidence)
    progress = tqdm(
        pending_samples,
        desc=f"Ollama {model_name}",
        unit="image",
        initial=len(completed_ids),
        total=len(samples),
    )
    consecutive_failures = 0
    processed_since_checkpoint = 0
    for sample in progress:
        try:
            raw_output, attempts = _chat_with_retries(
                base_url=base_url,
                model=model_name,
                image_path=sample.image_path,
                prompt=prompt,
                timeout=timeout,
                max_retries=max_retries,
                retry_backoff=retry_backoff,
            )
        except RuntimeError as exc:
            consecutive_failures += 1
            row = {
                "sample_id": sample.sample_id,
                "domain": sample.domain,
                "class_name": sample.class_name,
                "prediction": None,
                "correct": False,
                "evidence": None,
                "raw_output": None,
                "image_path": str(sample.image_path),
                "error": str(exc),
            }
            rows_by_id[sample.sample_id] = row
            _append_jsonl(predictions_path, row)
            write_json(
                metrics_path,
                _build_metrics(
                    rows_by_id,
                    model_name=model_name,
                    base_url=base_url,
                    total_samples=len(samples),
                    extra=extra_metrics,
                ),
            )
            tqdm.write(f"Failed {sample.sample_id}: {exc}")
            if consecutive_failures >= max_consecutive_failures:
                raise RuntimeError(
                    "Ollama failed repeatedly. Partial progress was saved; restart Ollama and "
                    "rerun the same command to resume."
                ) from exc
            continue

        consecutive_failures = 0
        answer, evidence = _parse_ollama_answer(raw_output, class_names)
        row = {
            "sample_id": sample.sample_id,
            "domain": sample.domain,
            "class_name": sample.class_name,
            "prediction": answer,
            "correct": answer == sample.class_name,
            "evidence": evidence,
            "raw_output": raw_output,
            "image_path": str(sample.image_path),
            "attempts": attempts,
            "error": None,
        }
        rows_by_id[sample.sample_id] = row
        completed_ids.add(sample.sample_id)
        _append_jsonl(predictions_path, row)
        processed_since_checkpoint += 1
        if processed_since_checkpoint >= checkpoint_every:
            write_json(
                metrics_path,
                _build_metrics(
                    rows_by_id,
                    model_name=model_name,
                    base_url=base_url,
                    total_samples=len(samples),
                    extra=extra_metrics,
                ),
            )
            processed_since_checkpoint = 0

    metrics = _build_metrics(
        rows_by_id,
        model_name=model_name,
        base_url=base_url,
        total_samples=len(samples),
        extra=extra_metrics,
    )
    write_json(metrics_path, metrics)
    return metrics
