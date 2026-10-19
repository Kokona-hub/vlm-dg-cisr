from __future__ import annotations

import inspect
from pathlib import Path

from .io_utils import read_yaml
from .manifest import audit_training_manifests
from .modeling import (
    lora_config,
    model_init_kwargs,
    processor_init_kwargs,
    require_huggingface_training_backend,
)
from .training_data import build_sft_dataset, load_class_names


def _compatible_sft_config_kwargs(
    sft_config_cls, values: dict, warmup_ratio: float, max_steps: int
) -> dict:
    """Keep SFTConfig construction compatible across TRL releases.

    Some server environments expose ``warmup_steps`` but not the
    ``warmup_ratio`` inherited by newer ``TrainingArguments``.  Resolve that
    difference here instead of requiring a dependency upgrade for a training
    configuration that is otherwise unchanged.
    """
    parameters = inspect.signature(sft_config_cls).parameters
    accepts_kwargs = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters.values()
    )
    compatible = values if accepts_kwargs else {
        key: value for key, value in values.items() if key in parameters
    }
    if "warmup_ratio" not in parameters and "warmup_steps" in parameters:
        compatible.pop("warmup_ratio", None)
        compatible["warmup_steps"] = (
            round(max_steps * warmup_ratio) if max_steps > 0 else 0
        )
    return compatible


def run_sft(
    config_path: str | Path,
    include_fourier_override: bool | None = None,
    output_dir_override: str | None = None,
) -> None:
    try:
        from transformers import AutoProcessor
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:
        raise RuntimeError(
            "Install server dependencies before SFT: `pip install -e .[train]`."
        ) from exc

    config = read_yaml(config_path)
    configured_fourier = bool(config["sft"].get("include_fourier", False))
    include_fourier = (
        include_fourier_override
        if include_fourier_override is not None
        else configured_fourier
    )
    audit_training_manifests(config, require_pairs=include_fourier)
    model_cfg = config["model"]
    require_huggingface_training_backend(model_cfg)
    train_cfg = config["sft"]
    source_manifest = config["data"]["source_manifest"]
    class_names = load_class_names(source_manifest)
    dataset = build_sft_dataset(
        source_manifest=source_manifest,
        class_names=class_names,
        use_evidence=bool(config.get("use_evidence", True)),
        pair_manifest=config.get("data", {}).get("pair_manifest"),
        include_fourier=include_fourier,
    )
    processor = AutoProcessor.from_pretrained(
        model_cfg["name_or_path"], **processor_init_kwargs(model_cfg)
    )
    max_steps = int(train_cfg.get("max_steps", -1))
    warmup_ratio = float(train_cfg.get("warmup_ratio", 0.03))
    sft_kwargs = dict(
        output_dir=str(output_dir_override or train_cfg["output_dir"]),
        max_steps=max_steps,
        num_train_epochs=float(train_cfg.get("num_train_epochs", 1.0)),
        per_device_train_batch_size=int(train_cfg.get("per_device_train_batch_size", 1)),
        gradient_accumulation_steps=int(train_cfg.get("gradient_accumulation_steps", 8)),
        learning_rate=float(train_cfg.get("learning_rate", 2e-4)),
        warmup_ratio=warmup_ratio,
        lr_scheduler_type=str(train_cfg.get("lr_scheduler_type", "cosine")),
        logging_steps=int(train_cfg.get("logging_steps", 5)),
        save_steps=int(train_cfg.get("save_steps", 250)),
        save_total_limit=int(train_cfg.get("save_total_limit", 2)),
        bf16=str(model_cfg.get("dtype", "bfloat16")) == "bfloat16",
        fp16=str(model_cfg.get("dtype")) == "float16",
        gradient_checkpointing=bool(train_cfg.get("gradient_checkpointing", True)),
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_length=int(train_cfg.get("max_length", 512)),
        completion_only_loss=True,
        packing=False,
        remove_unused_columns=False,
        report_to=list(train_cfg.get("report_to", ["tensorboard"])),
        seed=int(config.get("seed", 42)),
        data_seed=int(config.get("seed", 42)),
        model_init_kwargs=model_init_kwargs(model_cfg),
    )
    args = SFTConfig(
        **_compatible_sft_config_kwargs(SFTConfig, sft_kwargs, warmup_ratio, max_steps)
    )
    trainer = SFTTrainer(
        model=model_cfg["name_or_path"],
        args=args,
        train_dataset=dataset,
        processing_class=processor,
        peft_config=lora_config(config["lora"]),
    )
    trainer.train(resume_from_checkpoint=train_cfg.get("resume_from_checkpoint"))
    output_dir = str(output_dir_override or train_cfg["output_dir"])
    trainer.save_model(output_dir)
    processor.save_pretrained(output_dir)
