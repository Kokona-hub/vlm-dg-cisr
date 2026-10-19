from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from tqdm import tqdm

from .datasets import ImageListSelection, load_image_list, select_samples_by_image_list
from .io_utils import read_json, read_yaml, write_json, write_jsonl
from .manifest import load_samples_from_config, validate_domain_protocol
from .modeling import load_vlm, load_vlm_with_adapter, processor_init_kwargs
from .parsing import parse_single_prediction
from .prompts import SYSTEM_PROMPT, single_image_prompt


def _adapter_directory_if_valid(value: str | Path) -> Path | None:
    """Return the directory holding ``adapter_config.json``, or None.

    PEFT either writes a named adapter directly into the requested directory or
    nests it one level deeper under its own name, so both layouts are accepted.
    """
    candidate = Path(value).expanduser()
    for possibility in (candidate, candidate / candidate.name):
        if (possibility / "adapter_config.json").is_file():
            return possibility.resolve()
    return None


def _recorded_shared_adapter_paths(experiment_root: Path) -> list[Path]:
    """Collect Shared-LoRA paths recorded by completed CRLoRA-GRPO runs."""
    resolved: list[Path] = []
    if not experiment_root.is_dir():
        return resolved
    for ledger in sorted(experiment_root.glob("*/adapter_paths.json")):
        shared_path = read_json(ledger).get("shared_adapter_path")
        directory = _adapter_directory_if_valid(shared_path) if shared_path else None
        if directory is not None and directory not in resolved:
            resolved.append(directory)
    return resolved


def resolve_adapter_path(config: dict[str, Any], value: str | Path) -> tuple[Path, Path | None]:
    """Resolve an adapter path, falling back to a recorded Shared LoRA.

    Returns the directory to load and, when it was discovered rather than
    configured, the path that was actually used.
    """
    configured = Path(value).expanduser()
    resolved = _adapter_directory_if_valid(configured)
    if resolved is not None:
        return resolved, None

    crlora_output_dir = str(config.get("crlora_grpo", {}).get("output_dir") or "")
    discovered: list[Path] = []
    if crlora_output_dir:
        # The experiment root holds one subdirectory per CRLoRA variant, so a
        # stale evaluation.adapter_path still points at a discoverable adapter.
        discovered = _recorded_shared_adapter_paths(
            Path(crlora_output_dir).expanduser().parent
        )
    if len(discovered) == 1:
        return discovered[0], configured
    hint = (
        "Discovered Shared-LoRA adapters from adapter_paths.json: "
        + ", ".join(str(item) for item in discovered)
        if discovered
        else "No adapter_paths.json with a usable shared_adapter_path was found next to the "
        "configured crlora_grpo.output_dir."
    )
    raise FileNotFoundError(
        f"No adapter_config.json under {configured.resolve()}.\n"
        f"{hint}\n"
        "Finish the training stage that produces this adapter, pass the concrete directory with "
        "`--adapter-path`, or run the base model with `--base-model`."
    )


@torch.inference_mode()
def run_evaluation(
    config_path: str | Path,
    adapter_override: str | None = None,
    output_dir_override: str | None = None,
    image_list_override: str | None = None,
) -> dict[str, Any]:
    try:
        from transformers import AutoProcessor
    except ImportError as exc:
        raise RuntimeError("Install server dependencies before evaluation.") from exc

    config = read_yaml(config_path)
    validate_domain_protocol(config)
    model_cfg = config["model"]
    eval_cfg = config["evaluation"]
    samples = load_samples_from_config(config, split=str(eval_cfg.get("split", "test")))
    class_names = sorted({sample.class_name for sample in samples})
    # Closed-set DG assumes source and target share the same label space.
    # Lock candidates to the source manifest when it is available.
    source_manifest = config.get("data", {}).get("source_manifest")
    if source_manifest:
        from .training_data import load_class_names

        class_names = load_class_names(source_manifest)

    # The label space above always comes from the full target split; only the
    # evaluated image subset is narrowed, so candidate labels stay comparable
    # with a full-domain run on the same adapter.
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

    configured_adapter = eval_cfg.get("adapter_path")
    adapter_path = adapter_override if adapter_override is not None else configured_adapter
    substituted_from: str | None = None
    if adapter_path:
        resolved_adapter, substituted_from = resolve_adapter_path(config, adapter_path)
        if substituted_from is not None:
            tqdm.write(
                f"Configured adapter path {substituted_from} has no adapter_config.json; "
                f"using the recorded Shared LoRA at {resolved_adapter} instead."
            )
        adapter_path = resolved_adapter
        model = load_vlm_with_adapter(
            base_model_path=str(model_cfg["name_or_path"]),
            adapter_path=str(adapter_path),
            model_config=model_cfg,
            trainable=False,
        )
    else:
        model = load_vlm(str(model_cfg["name_or_path"]), model_cfg)
    model.eval()
    processor = AutoProcessor.from_pretrained(
        model_cfg["name_or_path"], **processor_init_kwargs(model_cfg)
    )
    device = next(model.parameters()).device
    predictions: list[dict[str, Any]] = []
    max_samples = eval_cfg.get("max_samples")
    if max_samples is not None:
        samples = samples[: int(max_samples)]
    for sample in tqdm(samples, desc="Evaluating target domain", unit="image"):
        with Image.open(sample.image_path) as image:
            image = image.convert("RGB")
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {
                            "type": "text",
                            "text": single_image_prompt(
                                class_names, use_evidence=bool(config.get("use_evidence", True))
                            ),
                        },
                    ],
                },
            ]
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = processor(text=[text], images=[image], return_tensors="pt").to(device)
            generated = model.generate(
                **inputs,
                max_new_tokens=int(eval_cfg.get("max_new_tokens", 96)),
                do_sample=False,
            )
            new_tokens = generated[:, inputs["input_ids"].shape[1] :]
            raw_output = processor.batch_decode(new_tokens, skip_special_tokens=True)[0]
        answer, evidence = parse_single_prediction(raw_output)
        predictions.append(
            {
                "sample_id": sample.sample_id,
                "domain": sample.domain,
                "class_name": sample.class_name,
                "prediction": answer,
                "correct": answer == sample.class_name,
                "evidence": evidence,
                "raw_output": raw_output,
                "image_path": str(sample.image_path),
            }
        )
    domain_total: Counter[str] = Counter()
    domain_correct: Counter[str] = Counter()
    class_total: Counter[str] = Counter()
    class_correct: Counter[str] = Counter()
    for row in predictions:
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
        "num_samples": len(predictions),
        "accuracy": sum(row["correct"] for row in predictions) / max(1, len(predictions)),
        "macro_class_accuracy": sum(class_accuracy.values()) / max(1, len(class_accuracy)),
        "domain_accuracy": domain_accuracy,
        "class_accuracy": class_accuracy,
        "adapter_path": str(adapter_path) if adapter_path else None,
        "backend": "huggingface",
        "model": str(model_cfg["name_or_path"]),
        "target_split_num_samples": total_target_samples,
        "image_list": str(image_list_value) if image_list_value else None,
        "image_list_entries": selection.entry_count if selection else None,
        "image_list_unmatched": unmatched_list_entries if selection else None,
        "adapter_path_substituted_from": substituted_from,
    }
    configured_output_dir = str(eval_cfg["output_dir"])
    if image_list_value and output_dir_override is None:
        # Never let a subset run overwrite the full-domain predictions.jsonl.
        configured_output_dir = f"{configured_output_dir}_subset"
    output_dir = Path(output_dir_override or configured_output_dir).expanduser().resolve()
    metrics["output_dir"] = str(output_dir)
    write_jsonl(output_dir / "predictions.jsonl", predictions)
    write_json(output_dir / "metrics.json", metrics)
    return metrics
