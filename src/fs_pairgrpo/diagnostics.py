from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from .io_utils import read_yaml
from .modeling import lora_config, model_init_kwargs


def _target_matches(module_names: list[str], targets: list[str] | str) -> dict[str, list[str]]:
    if isinstance(targets, str):
        return {targets: [name for name in module_names if re.fullmatch(targets, name)]}
    return {
        target: [name for name in module_names if name == target or name.endswith(f".{target}")]
        for target in targets
    }


def diagnose_model(config_path: str | Path) -> dict[str, Any]:
    """Load the configured QLoRA model and audit matched modules before a training run."""
    try:
        import torch
        from peft import get_peft_model, prepare_model_for_kbit_training
        from transformers import AutoModelForImageTextToText
    except ImportError as exc:
        raise RuntimeError("Install server dependencies before running model diagnostics.") from exc

    config = read_yaml(config_path)
    model_cfg = config["model"]
    peft_cfg = lora_config(config["lora"])
    model = AutoModelForImageTextToText.from_pretrained(
        model_cfg["name_or_path"], **model_init_kwargs(model_cfg)
    )
    module_names = [name for name, _ in model.named_modules()]
    targets = peft_cfg.target_modules
    matches = _target_matches(module_names, targets)
    missing = [target for target, names in matches.items() if not names]
    if missing:
        raise ValueError(f"LoRA targets matched no model modules: {missing}")

    if bool(model_cfg.get("load_in_4bit", True)):
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=bool(model_cfg.get("gradient_checkpointing", True)),
        )
    model = get_peft_model(model, peft_cfg)
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    categories = Counter(
        "vision"
        if any(part in name.lower() for part in ("visual", "vision"))
        else "language_or_projector"
        for names in matches.values()
        for name in names
    )
    gpu_memory = []
    for index in range(torch.cuda.device_count()):
        free, capacity = torch.cuda.mem_get_info(index)
        gpu_memory.append(
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "free_gib": round(free / 2**30, 2),
                "capacity_gib": round(capacity / 2**30, 2),
            }
        )
    result = {
        "model": str(model_cfg["name_or_path"]),
        "load_in_4bit": bool(model_cfg.get("load_in_4bit", True)),
        "target_match_counts": {target: len(names) for target, names in matches.items()},
        "target_match_examples": {target: names[:5] for target, names in matches.items()},
        "matched_module_categories": dict(categories),
        "trainable_parameters": trainable,
        "total_parameters": total,
        "trainable_percent": round(100.0 * trainable / max(1, total), 4),
        "gpus": gpu_memory,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result
