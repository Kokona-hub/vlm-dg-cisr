from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import DataLoader

from .fourier import fourier_style_extrapolation
from .io_utils import read_jsonl, read_yaml, write_json
from .manifest import audit_training_manifests
from .modeling import (
    load_vlm_with_trainable_and_reference_adapter,
    processor_init_kwargs,
    require_huggingface_training_backend,
)
from .outcome_rl import normalized_group_advantages, outcome_rewards
from .prompts import SYSTEM_PROMPT, single_image_prompt
from .training_data import load_class_names


def _messages(image: Image.Image, prompt_text: str) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt_text},
            ],
        },
    ]


def _move_to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device) if hasattr(value, "to") else value
        for key, value in batch.items()
    }


def _prepare_prompt_batch(processor, image: Image.Image, prompt_text: str, count: int, device):
    messages = [_messages(image, prompt_text) for _ in range(count)]
    texts = [
        processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        for message in messages
    ]
    batch = processor(
        text=texts,
        images=[image] * count,
        padding=True,
        return_tensors="pt",
    )
    return _move_to_device(dict(batch), device)


def _completion_mask(completion_ids: torch.Tensor, eos_id: int, pad_id: int) -> torch.Tensor:
    mask = completion_ids.ne(pad_id)
    eos = completion_ids.eq(eos_id)
    if eos.any():
        positions = torch.arange(completion_ids.shape[1], device=completion_ids.device)
        first_eos = torch.where(
            eos.any(dim=1),
            eos.float().argmax(dim=1),
            torch.full((completion_ids.shape[0],), completion_ids.shape[1] - 1, device=eos.device),
        )
        mask = mask & positions.unsqueeze(0).le(first_eos.unsqueeze(1))
    return mask


def _extend_sequence_fields(batch: dict[str, Any], completion_length: int) -> None:
    for name in ("token_type_ids", "mm_token_type_ids"):
        if name in batch:
            value = batch[name]
            batch[name] = torch.cat(
                [value, value.new_zeros((value.shape[0], completion_length))], dim=1
            )


def _sample_rollouts(
    model,
    processor,
    image: Image.Image,
    prompt_text: str,
    count: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor, list[str]]:
    device = next(model.parameters()).device
    prompt_batch = _prepare_prompt_batch(processor, image, prompt_text, count, device)
    prompt_width = prompt_batch["input_ids"].shape[1]
    model.set_adapter("default")
    was_training = model.training
    model.eval()
    with torch.no_grad():
        generated = model.generate(
            **prompt_batch,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
        )
    if was_training:
        model.train()
    completion_ids = generated[:, prompt_width:]
    mask = _completion_mask(
        completion_ids, processor.tokenizer.eos_token_id, processor.tokenizer.pad_token_id
    )
    texts = processor.batch_decode(completion_ids, skip_special_tokens=True)
    return prompt_batch, completion_ids, mask, texts


def _sequence_logps(
    model,
    prompt_batch: dict[str, Any],
    completion_ids: torch.Tensor,
    completion_mask: torch.Tensor,
) -> torch.Tensor:
    prompt_ids = prompt_batch["input_ids"]
    full_ids = torch.cat([prompt_ids, completion_ids], dim=1)
    full_attention = torch.cat(
        [prompt_batch["attention_mask"], completion_mask.long()], dim=1
    )
    forward = {
        key: value
        for key, value in prompt_batch.items()
        if key not in {"input_ids", "attention_mask"}
    }
    _extend_sequence_fields(forward, completion_ids.shape[1])
    keep = int(completion_ids.shape[1])
    try:
        output = model(
            input_ids=full_ids,
            attention_mask=full_attention,
            use_cache=False,
            logits_to_keep=keep,
            **forward,
        )
        logits = output.logits
        if logits.shape[1] != keep:
            raise TypeError("model ignored logits_to_keep")
    except (TypeError, ValueError):
        output = model(
            input_ids=full_ids, attention_mask=full_attention, use_cache=False, **forward
        )
        start = prompt_ids.shape[1] - 1
        logits = output.logits[:, start : start + keep, :]
    selected_logits = logits.gather(-1, completion_ids.unsqueeze(-1)).squeeze(-1)
    token_logps = selected_logits - torch.logsumexp(logits, dim=-1)
    lengths = completion_mask.sum(dim=1).clamp_min(1)
    return (token_logps * completion_mask).sum(dim=1) / lengths


def _policy_and_reference_logps(
    distributed_model, adapter_model, prompt_batch, completion_ids, completion_mask
):
    adapter_model.set_adapter("reference")
    for name, parameter in adapter_model.named_parameters():
        if ".reference." in name:
            parameter.requires_grad_(False)
    with torch.no_grad():
        reference = _sequence_logps(
            distributed_model, prompt_batch, completion_ids, completion_mask
        )
    adapter_model.set_adapter("default")
    policy = _sequence_logps(distributed_model, prompt_batch, completion_ids, completion_mask)
    return policy, reference


def _load_rgb(path: str) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def _load_styled_rgb(row: dict[str, Any]) -> Image.Image:
    styled_path = row.get("styled_path")
    if styled_path and Path(styled_path).is_file():
        return _load_rgb(str(styled_path))
    with Image.open(row["original_path"]) as source, Image.open(row["donor_path"]) as donor:
        return fourier_style_extrapolation(
            source,
            donor,
            alpha=float(row["alpha"]),
            low_frequency_ratio=float(row["low_frequency_ratio"]),
        )


def _collate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return rows


def run_grpo(
    config_path: str | Path,
    mode_override: str | None = None,
    output_dir_override: str | None = None,
) -> None:
    try:
        from accelerate import Accelerator
        from accelerate.utils import set_seed
        from transformers import AutoProcessor, get_scheduler
    except ImportError as exc:
        raise RuntimeError("Install requirements.txt before outcome-RL training.") from exc

    config = read_yaml(config_path)
    set_seed(int(config.get("seed", 42)))
    audit_training_manifests(config, require_pairs=True)
    model_cfg = config["model"]
    require_huggingface_training_backend(model_cfg)
    train_cfg = config["grpo"]
    mode = str(mode_override or train_cfg.get("mode", "independent_pair"))
    if mode not in {"single", "independent_pair"}:
        raise ValueError("grpo.mode must be 'single' or 'independent_pair'.")
    num_generations = int(train_cfg.get("num_generations", 4))
    if num_generations < 2:
        raise ValueError("GRPO requires num_generations >= 2.")

    accelerator = Accelerator(
        gradient_accumulation_steps=int(train_cfg.get("gradient_accumulation_steps", 1))
    )
    processor = AutoProcessor.from_pretrained(
        model_cfg["name_or_path"], **processor_init_kwargs(model_cfg)
    )
    model = load_vlm_with_trainable_and_reference_adapter(
        str(model_cfg["name_or_path"]),
        str(train_cfg["sft_adapter_path"]),
        model_cfg,
    )
    reference_trainable = [
        name
        for name, parameter in model.named_parameters()
        if ".reference." in name and parameter.requires_grad
    ]
    if reference_trainable:
        raise RuntimeError(
            "The frozen SFT reference adapter unexpectedly has trainable parameters: "
            f"{reference_trainable[:5]}"
        )
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("No trainable policy LoRA parameters were found.")
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(train_cfg.get("learning_rate", 1e-6)),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )
    # The manifest reader streams rows; GRPO needs length and shuffled epochs.
    rows = list(read_jsonl(config["data"]["pair_manifest"]))
    loader = DataLoader(
        rows,
        batch_size=int(train_cfg.get("per_device_train_batch_size", 1)),
        shuffle=True,
        collate_fn=_collate,
    )
    epochs = float(train_cfg.get("num_train_epochs", 1.0))
    configured_steps = int(train_cfg.get("max_steps", -1))
    steps_per_epoch = math.ceil(len(loader) / accelerator.gradient_accumulation_steps)
    max_steps = (
        configured_steps
        if configured_steps > 0
        else max(1, math.ceil(epochs * steps_per_epoch))
    )
    scheduler = get_scheduler(
        str(train_cfg.get("lr_scheduler_type", "cosine")),
        optimizer,
        num_warmup_steps=round(max_steps * float(train_cfg.get("warmup_ratio", 0.03))),
        num_training_steps=max_steps,
    )
    model, optimizer, loader, scheduler = accelerator.prepare(model, optimizer, loader, scheduler)
    unwrapped = accelerator.unwrap_model(model)
    class_names = load_class_names(config["data"]["source_manifest"])
    prompt_text = single_image_prompt(
        class_names, use_evidence=bool(config.get("use_evidence", False))
    )
    beta = float(train_cfg.get("beta", 0.01))
    clip_epsilon = float(train_cfg.get("epsilon", 0.2))
    format_bonus = float(train_cfg.get("format_bonus", 0.05))
    log_every = int(train_cfg.get("logging_steps", 1))
    global_step = 0
    output_dir = Path(output_dir_override or train_cfg["output_dir"])
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        write_json(
            output_dir / "run_summary.json",
            {
                "config": str(Path(config_path).resolve()),
                "mode": mode,
                "num_pairs": len(rows),
                "num_generations": num_generations,
                "max_steps": max_steps,
                "trainable_parameters": sum(parameter.numel() for parameter in trainable),
                "reference_adapter": "reference (frozen SFT copy)",
            },
        )
    optimizer.zero_grad(set_to_none=True)
    model.train()

    while global_step < max_steps:
        for batch_rows in loader:
            with accelerator.accumulate(model):
                sample_losses = []
                reward_means = []
                for row in batch_rows:
                    original = _load_rgb(str(row["original_path"]))
                    try:
                        original_rollout = _sample_rollouts(
                            unwrapped, processor, original, prompt_text, num_generations,
                            int(train_cfg.get("max_completion_length", 64)),
                            float(train_cfg.get("temperature", 0.9)),
                            float(train_cfg.get("top_p", 0.95)),
                        )
                    finally:
                        original.close()
                    styled_rollout = None
                    styled_texts = None
                    if mode == "independent_pair":
                        styled = _load_styled_rgb(row)
                        try:
                            styled_rollout = _sample_rollouts(
                                unwrapped, processor, styled, prompt_text, num_generations,
                                int(train_cfg.get("max_completion_length", 64)),
                                float(train_cfg.get("temperature", 0.9)),
                                float(train_cfg.get("top_p", 0.95)),
                            )
                        finally:
                            styled.close()
                        styled_texts = styled_rollout[3]
                    rewards = outcome_rewards(
                        original_rollout[3], str(row["class_name"]), styled_texts, format_bonus
                    )
                    advantages = torch.tensor(
                        normalized_group_advantages(rewards),
                        device=accelerator.device,
                        dtype=torch.float32,
                    )
                    op, oref = _policy_and_reference_logps(
                        model,
                        unwrapped,
                        original_rollout[0],
                        original_rollout[1],
                        original_rollout[2],
                    )
                    policy_logps = op
                    reference_logps = oref
                    if styled_rollout is not None:
                        sp, sref = _policy_and_reference_logps(
                            model,
                            unwrapped,
                            styled_rollout[0],
                            styled_rollout[1],
                            styled_rollout[2],
                        )
                        policy_logps = 0.5 * (op + sp)
                        reference_logps = 0.5 * (oref + sref)
                    ratio = torch.exp(policy_logps - policy_logps.detach())
                    clipped_ratio = ratio.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon)
                    policy_objective = torch.minimum(
                        ratio * advantages, clipped_ratio * advantages
                    )
                    log_ratio = reference_logps - policy_logps
                    kl = torch.exp(log_ratio) - log_ratio - 1.0
                    sample_losses.append((-policy_objective + beta * kl).mean())
                    reward_means.append(sum(rewards) / len(rewards))
                loss = torch.stack(sample_losses).mean()
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(
                        trainable, float(train_cfg.get("max_grad_norm", 1.0))
                    )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if accelerator.sync_gradients:
                global_step += 1
                if accelerator.is_main_process and global_step % log_every == 0:
                    log_row = {
                        "step": global_step,
                        "mode": mode,
                        "loss": float(loss.detach()),
                        "mean_reward": sum(reward_means) / len(reward_means),
                        "learning_rate": scheduler.get_last_lr()[0],
                    }
                    print(
                        json.dumps(log_row),
                        flush=True,
                    )
                    with (output_dir / "train_metrics.jsonl").open(
                        "a", encoding="utf-8"
                    ) as handle:
                        handle.write(json.dumps(log_row) + "\n")
                if global_step >= max_steps:
                    break

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped.set_adapter("default")
        unwrapped.save_pretrained(output_dir, selected_adapters=["default"])
        processor.save_pretrained(output_dir)
