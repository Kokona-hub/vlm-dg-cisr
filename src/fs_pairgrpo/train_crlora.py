"""CISR/TU Consensus-Residual LoRA training with GRPO.

This trainer keeps the original FS-IPGRPO implementation intact.  It adds a
separate path where source-domain policy gradients are collected independently,
then synchronously routed to a Shared LoRA and domain-specific residual LoRAs.
"""

from __future__ import annotations

import json
import math
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from .consensus import (
    adapter_parameter_key,
    cisr_rankwise_route,
    rankwise_gradient_components,
    route_consensus_residual,
    transfer_utility_gated_route,
)
from .io_utils import read_jsonl, read_yaml, write_json
from .manifest import audit_training_manifests
from .modeling import (
    load_vlm_with_consensus_residual_adapters,
    lora_config,
    processor_init_kwargs,
    require_huggingface_training_backend,
    set_active_adapters,
)
from .outcome_rl import normalized_group_advantages, outcome_rewards
from .prompts import single_image_prompt
from .train_grpo import (
    _completion_mask,
    _extend_sequence_fields,
    _load_rgb,
    _load_styled_rgb,
    _prepare_prompt_batch,
)
from .training_data import load_class_names


def _adapter_parameter_map(model, adapter_name: str) -> dict[str, torch.nn.Parameter]:
    parameters: dict[str, torch.nn.Parameter] = {}
    for name, parameter in model.named_parameters():
        key = adapter_parameter_key(name, adapter_name)
        if key is not None:
            if key in parameters:
                raise RuntimeError(f"Duplicate LoRA coordinate key {key!r} for {adapter_name}.")
            parameters[key] = parameter
    if not parameters:
        raise RuntimeError(f"No PEFT parameters were found for adapter {adapter_name!r}.")
    return parameters


def _set_trainable_crlora_parameters(model, adapter_names: list[str]) -> None:
    markers = tuple(f".{name}." for name in adapter_names)
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(any(marker in name for marker in markers))


def _sequence_logps(model, prompt_batch, completion_ids, completion_mask) -> torch.Tensor:
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
    # Recent Transformers causal-LM implementations (including Qwen3-VL)
    # expose ``logits_to_keep``.  Asking the lm head for only the completion
    # positions avoids materialising logits for thousands of image/prompt
    # tokens.  Keep a compatibility fallback for older Transformers/PEFT.
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
            # Some older wrappers accept and ignore the keyword. Reuse the
            # already-computed tensor rather than launching a second forward.
            start = prompt_ids.shape[1] - 1
            logits = logits[:, start : start + keep, :]
    except TypeError:
        output = model(
            input_ids=full_ids, attention_mask=full_attention, use_cache=False, **forward
        )
        start = prompt_ids.shape[1] - 1
        logits = output.logits[:, start : start + keep, :]
    # Avoid ``logits.float()`` + ``log_softmax`` here.  For Qwen3-VL the
    # vocabulary projection is very large; materialising an FP32 copy of the
    # complete [batch, sequence, vocab] tensor adds several GiB at the exact
    # point where autograd builds the LoRA gradients.  The equivalent
    # log-probability identity below keeps the computation in the model's
    # bfloat16 dtype and only gathers the chosen token logits.
    selected_logits = logits.gather(-1, completion_ids.unsqueeze(-1)).squeeze(-1)
    token_logps = selected_logits - torch.logsumexp(logits, dim=-1)
    lengths = completion_mask.sum(dim=1).clamp_min(1)
    return (token_logps * completion_mask).sum(dim=1) / lengths


def _sample_rollouts(
    model,
    processor,
    image,
    prompt_text: str,
    count: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    active_adapters: str | list[str],
):
    device = next(model.parameters()).device
    prompt_batch = _prepare_prompt_batch(processor, image, prompt_text, count, device)
    prompt_width = prompt_batch["input_ids"].shape[1]
    set_active_adapters(model, active_adapters)
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


def _policy_and_reference_logps(
    distributed_model,
    adapter_model,
    prompt_batch,
    completion_ids,
    completion_mask,
    policy_adapters: str | list[str],
    reference_adapter: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    set_active_adapters(adapter_model, reference_adapter)
    with torch.no_grad():
        reference = _sequence_logps(
            distributed_model, prompt_batch, completion_ids, completion_mask
        )
    set_active_adapters(adapter_model, policy_adapters)
    policy = _sequence_logps(distributed_model, prompt_batch, completion_ids, completion_mask)
    return policy, reference


def _row_loss(
    distributed_model,
    adapter_model,
    processor,
    row: dict[str, Any],
    prompt_text: str,
    train_cfg: dict[str, Any],
    policy_adapters: str | list[str],
    reference_adapter: str,
) -> tuple[torch.Tensor, float, bool]:
    """Return loss, mean terminal reward, and whether the rollout group has signal."""
    count = int(train_cfg.get("num_generations", 4))
    original = _load_rgb(str(row["original_path"]))
    try:
        original_rollout = _sample_rollouts(
            adapter_model, processor, original, prompt_text, count,
            int(train_cfg.get("max_completion_length", 64)),
            float(train_cfg.get("temperature", 0.9)), float(train_cfg.get("top_p", 0.95)),
            policy_adapters,
        )
    finally:
        original.close()
    styled = _load_styled_rgb(row)
    try:
        styled_rollout = _sample_rollouts(
            adapter_model, processor, styled, prompt_text, count,
            int(train_cfg.get("max_completion_length", 64)),
            float(train_cfg.get("temperature", 0.9)), float(train_cfg.get("top_p", 0.95)),
            policy_adapters,
        )
    finally:
        styled.close()
    rewards = outcome_rewards(
        original_rollout[3], str(row["class_name"]), styled_rollout[3],
        float(train_cfg.get("format_bonus", 0.05)),
    )
    advantage_values = normalized_group_advantages(rewards)
    advantages = torch.tensor(
        advantage_values,
        device=next(adapter_model.parameters()).device,
        dtype=torch.float32,
    )
    op, oref = _policy_and_reference_logps(
        distributed_model, adapter_model, *original_rollout[:3], policy_adapters, reference_adapter
    )
    sp, sref = _policy_and_reference_logps(
        distributed_model, adapter_model, *styled_rollout[:3], policy_adapters, reference_adapter
    )
    policy_logps = 0.5 * (op + sp)
    reference_logps = 0.5 * (oref + sref)
    epsilon = float(train_cfg.get("epsilon", 0.2))
    ratio = torch.exp(policy_logps - policy_logps.detach())
    clipped_ratio = ratio.clamp(1.0 - epsilon, 1.0 + epsilon)
    policy_objective = torch.minimum(ratio * advantages, clipped_ratio * advantages)
    log_ratio = reference_logps - policy_logps
    kl = torch.exp(log_ratio) - log_ratio - 1.0
    loss = (-policy_objective + float(train_cfg.get("beta", 0.01)) * kl).mean()
    has_advantage_signal = any(abs(value) > 0.0 for value in advantage_values)
    return loss, float(sum(rewards) / len(rewards)), has_advantage_signal


def _row_view_losses(
    distributed_model,
    adapter_model,
    processor,
    row: dict[str, Any],
    prompt_text: str,
    train_cfg: dict[str, Any],
    policy_adapters: str | list[str],
    reference_adapter: str,
) -> tuple[list[torch.Tensor], float, bool]:
    """Build one policy loss per view to keep only one backward graph alive.

    The original implementation averages original/style log-probabilities
    before calling ``autograd.grad``.  With a large VLM that retains both
    vision graphs at once and can exceed 24-GiB cards.  This variant samples
    the same pair but returns two losses; the caller differentiates and
    discards them sequentially, preserving the objective while lowering the
    peak activation memory.
    """
    count = int(train_cfg.get("num_generations", 4))
    original = _load_rgb(str(row["original_path"]))
    try:
        original_rollout = _sample_rollouts(
            adapter_model, processor, original, prompt_text, count,
            int(train_cfg.get("max_completion_length", 64)),
            float(train_cfg.get("temperature", 0.9)), float(train_cfg.get("top_p", 0.95)),
            policy_adapters,
        )
    finally:
        original.close()
    styled = _load_styled_rgb(row)
    try:
        styled_rollout = _sample_rollouts(
            adapter_model, processor, styled, prompt_text, count,
            int(train_cfg.get("max_completion_length", 64)),
            float(train_cfg.get("temperature", 0.9)), float(train_cfg.get("top_p", 0.95)),
            policy_adapters,
        )
    finally:
        styled.close()
    rewards = outcome_rewards(
        original_rollout[3], str(row["class_name"]), styled_rollout[3],
        float(train_cfg.get("format_bonus", 0.05)),
    )
    advantage_values = normalized_group_advantages(rewards)
    advantages = torch.tensor(
        advantage_values, device=next(adapter_model.parameters()).device, dtype=torch.float32
    )
    losses: list[torch.Tensor] = []
    epsilon = float(train_cfg.get("epsilon", 0.2))
    beta = float(train_cfg.get("beta", 0.01))
    for rollout in (original_rollout, styled_rollout):
        policy, reference = _policy_and_reference_logps(
            distributed_model, adapter_model, *rollout[:3], policy_adapters, reference_adapter
        )
        ratio = torch.exp(policy - policy.detach())
        clipped = ratio.clamp(1.0 - epsilon, 1.0 + epsilon)
        objective = torch.minimum(ratio * advantages, clipped * advantages)
        log_ratio = reference - policy
        kl = torch.exp(log_ratio) - log_ratio - 1.0
        losses.append((-objective + beta * kl).mean())
    return losses, float(sum(rewards) / len(rewards)), any(abs(v) > 0.0 for v in advantage_values)


def _row_view_data(
    adapter_model, processor, row: dict[str, Any], prompt_text: str,
    train_cfg: dict[str, Any], policy_adapters: str | list[str],
):
    """Sample the two views and compute rewards without building grad graphs."""
    count = int(train_cfg.get("num_generations", 4))
    def sample(image):
        return _sample_rollouts(
            adapter_model, processor, image, prompt_text, count,
            int(train_cfg.get("max_completion_length", 64)),
            float(train_cfg.get("temperature", 0.9)), float(train_cfg.get("top_p", 0.95)),
            policy_adapters,
        )
    original = _load_rgb(str(row["original_path"]))
    try:
        original_rollout = sample(original)
    finally:
        original.close()
    styled = _load_styled_rgb(row)
    try:
        styled_rollout = sample(styled)
    finally:
        styled.close()
    rewards = outcome_rewards(
        original_rollout[3], str(row["class_name"]), styled_rollout[3],
        float(train_cfg.get("format_bonus", 0.05)),
    )
    advantages = torch.tensor(
        normalized_group_advantages(rewards),
        device=next(adapter_model.parameters()).device, dtype=torch.float32,
    )
    has_signal = any(
        abs(float(value)) > 0.0 for value in advantages.detach().cpu().tolist()
    )
    return (
        (original_rollout, styled_rollout),
        advantages,
        float(sum(rewards) / len(rewards)),
        has_signal,
    )


def _view_loss_from_rollout(
    distributed_model, adapter_model, rollout, advantages,
    policy_adapters, reference_adapter, train_cfg,
):
    policy, reference = _policy_and_reference_logps(
        distributed_model, adapter_model, *rollout[:3], policy_adapters, reference_adapter
    )
    ratio = torch.exp(policy - policy.detach())
    epsilon = float(train_cfg.get("epsilon", 0.2))
    clipped = ratio.clamp(1.0 - epsilon, 1.0 + epsilon)
    objective = torch.minimum(ratio * advantages, clipped * advantages)
    log_ratio = reference - policy
    kl = torch.exp(log_ratio) - log_ratio - 1.0
    return (-objective + float(train_cfg.get("beta", 0.01)) * kl).mean()


def _heldout_behavior_terms(
    distributed_model,
    adapter_model,
    rollout,
    shared_adapter: str,
    reference_adapter: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Freeze behavior/reference log-probs for a held-out GRPO rollout."""
    set_active_adapters(adapter_model, reference_adapter)
    reference = _sequence_logps(distributed_model, *rollout[:3])
    set_active_adapters(adapter_model, shared_adapter)
    behavior = _sequence_logps(distributed_model, *rollout[:3])
    return behavior, reference


def _heldout_surrogate_loss(
    distributed_model,
    adapter_model,
    rollout,
    advantages: torch.Tensor,
    behavior_logps: torch.Tensor,
    reference_logps: torch.Tensor,
    shared_adapter: str,
    train_cfg: dict[str, Any],
) -> torch.Tensor:
    """Evaluate a fixed-behavior GRPO surrogate after a virtual update."""
    set_active_adapters(adapter_model, shared_adapter)
    policy = _sequence_logps(distributed_model, *rollout[:3])
    epsilon = float(train_cfg.get("epsilon", 0.2))
    ratio = torch.exp(policy - behavior_logps)
    objective = torch.minimum(
        ratio * advantages,
        ratio.clamp(1.0 - epsilon, 1.0 + epsilon) * advantages,
    )
    log_ratio = reference_logps - policy
    kl = torch.exp(log_ratio) - log_ratio - 1.0
    return (-objective + float(train_cfg.get("beta", 0.01)) * kl).mean()


def _virtual_transfer_utilities(
    distributed_model,
    adapter_model,
    shared_parameters: dict[str, torch.nn.Parameter],
    source_gradients: list[dict[str, torch.Tensor]],
    source_domains: list[str],
    validation_rollouts,
    validation_advantages: torch.Tensor,
    shared_adapter: str,
    reference_adapter: str,
    train_cfg: dict[str, Any],
) -> dict[str, float]:
    """Measure finite-difference transfer utility on a held-out domain.

    Each candidate source update is temporarily applied to Shared LoRA.  We
    then evaluate fixed held-out GRPO rollouts before and after that virtual
    update. The held-out behavior policy remains fixed, so the comparison is
    a valid finite-difference GRPO surrogate rather than a recomputed
    on-policy ratio that would always equal one. A positive score means the
    candidate lowers held-out loss. Parameters are restored exactly before
    returning; optimizer state is untouched.
    """
    if not validation_rollouts:
        return {domain: 0.0 for domain in source_domains}
    step_size = float(train_cfg.get("transfer_utility_step_size", 0.05))
    if step_size <= 0:
        raise ValueError("transfer_utility_step_size must be positive.")
    was_training = adapter_model.training
    adapter_model.eval()
    try:
        with torch.no_grad():
            behavior_terms = [
                _heldout_behavior_terms(
                    distributed_model,
                    adapter_model,
                    rollout,
                    shared_adapter,
                    reference_adapter,
                )
                for rollout in validation_rollouts
            ]
            baseline = torch.stack(
                [
                    _heldout_surrogate_loss(
                        distributed_model,
                        adapter_model,
                        rollout,
                        validation_advantages,
                        behavior,
                        reference,
                        shared_adapter,
                        train_cfg,
                    )
                    for rollout, (behavior, reference) in zip(
                        validation_rollouts, behavior_terms, strict=True
                    )
                ]
            ).mean()
            baseline_value = float(baseline.detach().cpu())
            denominator = max(abs(baseline_value), 1.0)
            utilities: dict[str, float] = {}
            for domain, gradient in zip(source_domains, source_gradients, strict=True):
                norm_sq = sum(
                    (value.float() * value.float()).sum() for value in gradient.values()
                ).clamp_min(1e-12)
                scale = step_size / float(norm_sq.sqrt().detach().cpu())
                try:
                    for key, parameter in shared_parameters.items():
                        parameter.add_(gradient[key].to(parameter), alpha=-scale)
                    virtual = torch.stack(
                        [
                            _heldout_surrogate_loss(
                                distributed_model,
                                adapter_model,
                                rollout,
                                validation_advantages,
                                behavior,
                                reference,
                                shared_adapter,
                                train_cfg,
                            )
                            for rollout, (behavior, reference) in zip(
                                validation_rollouts, behavior_terms, strict=True
                            )
                        ]
                    ).mean()
                finally:
                    for key, parameter in shared_parameters.items():
                        parameter.add_(gradient[key].to(parameter), alpha=scale)
                utility = (baseline_value - float(virtual.detach().cpu())) / denominator
                utilities[domain] = float(max(-1.0, min(1.0, utility)))
            return utilities
    finally:
        if was_training:
            adapter_model.train()


def _rankwise_counterfactual_utilities(
    source_gradients: list[dict[str, torch.Tensor]],
    source_domains: list[str],
    validation_gradient: dict[str, torch.Tensor] | None,
) -> dict[str, dict[int, float]]:
    """Estimate rank-level leave-out transfer utility without extra forwards.

    For a loss gradient ``g_val`` on the held-out pseudo-target, applying a
    small update ``theta <- theta - eta*g_{d,r}`` changes its loss by
    approximately ``-eta <g_val, g_{d,r}>``.  The cosine is therefore a scale-
    invariant first-order counterfactual utility.  A finite-difference variant
    can be added later, but this form is deterministic, cheap, and does not
    multiply the expensive VLM forward pass by the LoRA rank.
    """
    components = [rankwise_gradient_components(gradient) for gradient in source_gradients]
    num_ranks = len(components[0])
    if validation_gradient is None:
        return {
            domain: {rank: 0.0 for rank in range(num_ranks)}
            for domain in source_domains
        }
    validation_components = rankwise_gradient_components(validation_gradient, num_ranks)
    utilities: dict[str, dict[int, float]] = {}
    for domain, domain_components in zip(source_domains, components, strict=True):
        values: dict[int, float] = {}
        for rank in range(num_ranks):
            source = domain_components[rank]
            heldout = validation_components[rank]
            source_norm = sum(
                (value.float() * value.float()).sum() for value in source.values()
            ).sqrt()
            heldout_norm = sum(
                (value.float() * value.float()).sum() for value in heldout.values()
            ).sqrt()
            denominator = (source_norm * heldout_norm).clamp_min(1e-12)
            dot = sum(
                (source[name].float() * heldout[name].float()).sum() for name in source
            )
            values[rank] = float((dot / denominator).detach().cpu())
        utilities[domain] = values
    return utilities


def _gradient_norm(gradient: dict[str, torch.Tensor]) -> float:
    squared = sum((value.float() * value.float()).sum() for value in gradient.values())
    return float(squared.sqrt())


def _offload_gradient(gradient: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Copy detached routing gradients to host memory before rank expansion."""
    return {key: value.detach().cpu() for key, value in gradient.items()}


def _gradient_for_parameter_groups(
    loss: torch.Tensor,
    groups: dict[str, dict[str, torch.nn.Parameter]],
) -> dict[str, dict[str, torch.Tensor]]:
    """Extract gradients for several adapter groups in one backward traversal."""
    group_items = {group: list(parameters.items()) for group, parameters in groups.items()}
    values = [parameter for items in group_items.values() for _, parameter in items]
    gradients = torch.autograd.grad(loss, values, allow_unused=True)
    output: dict[str, dict[str, torch.Tensor]] = {}
    offset = 0
    for group, items in group_items.items():
        group_output: dict[str, torch.Tensor] = {}
        for key, parameter in items:
            gradient = gradients[offset]
            offset += 1
            group_output[key] = (
                gradient.detach()
                if gradient is not None
                else parameter.detach().new_zeros(parameter.shape)
            )
        output[group] = group_output
    return output


def _choose_inner_and_pseudo_domains(
    source_domains: list[str], step: int, pseudo_rotation: bool
) -> tuple[list[str], str | None]:
    if pseudo_rotation and len(source_domains) >= 3:
        pseudo = source_domains[(step - 1) % len(source_domains)]
        return [domain for domain in source_domains if domain != pseudo], pseudo
    return list(source_domains), None


def _export_adapter(model, output_dir: Path, adapter_name: str) -> Path:
    destination = output_dir / adapter_name
    model.save_pretrained(destination, selected_adapters=[adapter_name])
    # PEFT writes selected named adapters under <destination>/<adapter_name>/.
    # Keep a defensive fallback for PEFT variants that write directly into the
    # requested directory.
    nested = destination / adapter_name
    if (nested / "adapter_config.json").is_file():
        return nested
    if (destination / "adapter_config.json").is_file():
        return destination
    raise RuntimeError(
        f"PEFT did not save adapter_config.json for {adapter_name!r} under {destination}."
    )


def run_crlora_grpo(config_path: str | Path, output_dir_override: str | None = None) -> None:
    """Train CRLoRA-GRPO variants using source-only data and terminal rewards.

    ``routing_mode: cisr_rankwise`` decomposes source gradients by LoRA rank and
    routes only held-out-useful, cross-domain-consistent rank directions to the
    Shared adapter. ``consensus_only``, ``utility_only``, and ``mean_only`` are
    controlled routing ablations. ``routing_mode: domain_utility`` preserves TU
    behavior.
    """
    try:
        from accelerate import Accelerator
        from accelerate.utils import set_seed
        from transformers import AutoProcessor, get_scheduler
    except ImportError as exc:
        raise RuntimeError("Install requirements.txt before CRLoRA-GRPO training.") from exc

    config = read_yaml(config_path)
    set_seed(int(config.get("seed", 42)))
    audit_training_manifests(config, require_pairs=True)
    model_cfg = config["model"]
    require_huggingface_training_backend(model_cfg)
    train_cfg = config.get("crlora_grpo")
    if not isinstance(train_cfg, dict):
        raise ValueError("Missing 'crlora_grpo' configuration block.")
    if int(train_cfg.get("num_generations", 4)) < 2:
        raise ValueError("CRLoRA-GRPO requires crlora_grpo.num_generations >= 2.")
    if int(train_cfg.get("gradient_accumulation_steps", 1)) != 1:
        raise ValueError("CRLoRA-GRPO currently requires gradient_accumulation_steps: 1.")
    if int(train_cfg.get("per_domain_batch_size", 1)) != 1:
        raise ValueError("CRLoRA-GRPO currently requires per_domain_batch_size: 1.")

    # A Qwen3-VL 8B replica plus two-view GRPO activations can exceed a 24 GB
    # RTX 4090 even with 4-bit weights. This opt-in mode keeps the algorithm
    # unchanged but caps activation-heavy rollout settings for a single GPU.
    low_memory = os.environ.get("CRLORA_LOW_MEMORY", "0") == "1"
    ultra_low_memory = os.environ.get("CRLORA_ULTRA_LOW_MEMORY", "0") == "1"
    routing_mode = str(train_cfg.get("routing_mode", "domain_utility")).lower()
    cisr_enabled = routing_mode in {
        "cisr", "cisr_rankwise", "consensus_only", "utility_only", "mean_only"
    }
    if routing_mode not in {
        "domain_utility", "consensus", "cisr", "cisr_rankwise",
        "consensus_only", "utility_only", "mean_only",
    }:
        raise ValueError(
            "crlora_grpo.routing_mode must be one of domain_utility, consensus, "
            "cisr_rankwise, consensus_only, utility_only, mean_only."
        )
    use_rank_consensus = routing_mode not in {"utility_only", "mean_only"}
    use_rank_utility = routing_mode not in {"consensus_only", "mean_only"}
    sequential_grad = (
        low_memory
        or ultra_low_memory
        or bool(train_cfg.get("transfer_utility_gating", False))
        or cisr_enabled
        or os.environ.get("CRLORA_SEQUENTIAL_GRAD", "0") == "1"
    )
    if low_memory or ultra_low_memory:
        train_cfg = dict(train_cfg)
        train_cfg["num_generations"] = min(int(train_cfg.get("num_generations", 4)), 2)
        train_cfg["max_completion_length"] = min(
            int(train_cfg.get("max_completion_length", 96)), 32
        )
        model_cfg = dict(model_cfg)
        model_cfg["max_pixels"] = min(int(model_cfg.get("max_pixels", 401408)), 100352)
        model_cfg["min_pixels"] = min(int(model_cfg.get("min_pixels", 100352)), 50176)
    # If a 24-GiB card still peaks during autograd, this stricter setting
    # trades image resolution and rollout length for a runnable experiment.
    # It does not alter the consensus/residual objective.
    if ultra_low_memory:
        train_cfg["max_completion_length"] = min(
            int(train_cfg.get("max_completion_length", 32)), 16
        )
        model_cfg["max_pixels"] = min(int(model_cfg.get("max_pixels", 100352)), 50176)
        model_cfg["min_pixels"] = min(int(model_cfg.get("min_pixels", 50176)), 32768)

    # The low-memory profiles provide conservative defaults for a 24-GiB card.
    # Allow a user to spend otherwise-unused VRAM incrementally without
    # changing the YAML or disabling the CPU routing memory fix.  These values
    # are applied after the profile caps, so an explicit override is opt-in.
    for environment_name, config_name in (
        ("CRLORA_NUM_GENERATIONS", "num_generations"),
        ("CRLORA_MAX_COMPLETION_LENGTH", "max_completion_length"),
    ):
        raw_value = os.environ.get(environment_name)
        if raw_value:
            train_cfg[config_name] = max(1, int(raw_value))
    for environment_name, config_name in (
        ("CRLORA_MIN_PIXELS", "min_pixels"),
        ("CRLORA_MAX_PIXELS", "max_pixels"),
    ):
        raw_value = os.environ.get(environment_name)
        if raw_value:
            model_cfg[config_name] = max(1, int(raw_value))
    if int(model_cfg.get("min_pixels", 1)) > int(model_cfg.get("max_pixels", 1)):
        raise ValueError("CRLORA_MIN_PIXELS cannot exceed CRLORA_MAX_PIXELS.")

    source_domains = list(config["dataset"]["source_domains"])
    shared_name = str(train_cfg.get("shared_adapter_name", "shared"))
    reference_name = str(train_cfg.get("reference_adapter_name", "reference"))
    prefix = str(train_cfg.get("domain_adapter_prefix", "domain_"))
    domain_adapter_names = {domain: f"{prefix}{domain}" for domain in source_domains}
    # Avoid Accelerate's bf16->fp32 output conversion for the very large VLM
    # logits tensor. The model is still loaded with the configured bfloat16
    # compute dtype; disabling only this wrapper saves several GiB per GPU.
    configured_precision = train_cfg.get("mixed_precision", "no")
    # YAML 1.1 parses an unquoted ``no`` as boolean False. Normalize both
    # spellings before passing the value to Accelerate.
    if configured_precision is False or configured_precision is None:
        configured_precision = "no"
    configured_precision = str(configured_precision).lower()
    if configured_precision == "false":
        configured_precision = "no"
    if configured_precision not in {"no", "fp8", "fp16", "bf16"}:
        raise ValueError(
            "crlora_grpo.mixed_precision must be one of no, fp8, fp16, bf16."
        )
    accelerator = Accelerator(mixed_precision=configured_precision)
    if (low_memory or ultra_low_memory) and accelerator.is_main_process:
        print(
            json.dumps(
                {
                    "low_memory_mode": True,
                    "num_generations": train_cfg["num_generations"],
                    "max_completion_length": train_cfg["max_completion_length"],
                    "max_pixels": model_cfg["max_pixels"],
                    "ultra_low_memory": ultra_low_memory,
                }
            ),
            flush=True,
        )
    processor = AutoProcessor.from_pretrained(
        model_cfg["name_or_path"], **processor_init_kwargs(model_cfg)
    )
    model = load_vlm_with_consensus_residual_adapters(
        str(model_cfg["name_or_path"]), str(train_cfg["sft_adapter_path"]), model_cfg,
        lora_config(config["lora"]), source_domains, shared_name, reference_name, prefix,
    )
    train_adapter_names = [shared_name, *domain_adapter_names.values()]
    _set_trainable_crlora_parameters(model, train_adapter_names)
    shared_parameters = _adapter_parameter_map(model, shared_name)
    domain_parameters = {
        domain: _adapter_parameter_map(model, adapter_name)
        for domain, adapter_name in domain_adapter_names.items()
    }
    expected = set(shared_parameters)
    for domain, parameters in domain_parameters.items():
        shapes_match = all(
            parameters[key].shape == shared_parameters[key].shape for key in expected
        )
        if set(parameters) != expected or not shapes_match:
            raise RuntimeError(
                f"Domain adapter {domain!r} does not share Shared LoRA's parameter coordinates. "
                "Use one common LoRA configuration for every adapter."
            )
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable, lr=float(train_cfg.get("learning_rate", 1e-6)),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )

    rows_by_domain: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in read_jsonl(config["data"]["pair_manifest"]):
        rows_by_domain[str(row["source_domain"])].append(row)
    missing = [domain for domain in source_domains if not rows_by_domain[domain]]
    if missing:
        raise ValueError(f"No Fourier pairs were found for source domains: {missing}")
    randomizer = random.Random(int(config.get("seed", 42)))
    for domain in source_domains:
        randomizer.shuffle(rows_by_domain[domain])
    # Each DDP rank starts at a different source example. The later all-reduce
    # therefore averages distinct per-domain gradients rather than duplicates.
    positions = {domain: accelerator.process_index for domain in source_domains}

    max_steps = int(train_cfg.get("max_steps", -1))
    if max_steps <= 0:
        min_domain_rows = min(len(rows_by_domain[domain]) for domain in source_domains)
        max_steps = max(
            1,
            math.ceil(float(train_cfg.get("num_train_epochs", 1.0)) * min_domain_rows),
        )
    scheduler = get_scheduler(
        str(train_cfg.get("lr_scheduler_type", "cosine")), optimizer,
        num_warmup_steps=round(max_steps * float(train_cfg.get("warmup_ratio", 0.03))),
        num_training_steps=max_steps,
    )
    model, optimizer, scheduler = accelerator.prepare(model, optimizer, scheduler)
    unwrapped = accelerator.unwrap_model(model)
    prompt_text = single_image_prompt(
        load_class_names(config["data"]["source_manifest"]),
        use_evidence=bool(config.get("use_evidence", False)),
    )
    output_dir = Path(output_dir_override or train_cfg["output_dir"])
    if accelerator.is_main_process:
        output_dir.mkdir(parents=True, exist_ok=True)
        if routing_mode == "consensus_only":
            method_name = "CISR-Consensus-Only"
        elif routing_mode == "utility_only":
            method_name = "CISR-Utility-Only"
        elif routing_mode == "mean_only":
            method_name = "Rank1-Mean-Only"
        elif cisr_enabled:
            method_name = "CISR-LoRA-GRPO"
        elif bool(train_cfg.get("transfer_utility_gating", False)):
            method_name = "TU-CRLoRA-GRPO"
        else:
            method_name = "FS-CRLoRA-GRPO"
        write_json(output_dir / "run_summary.json", {
            "config": str(Path(config_path).resolve()),
            "method": method_name,
            "routing_mode": routing_mode,
            "source_domains": source_domains,
            "shared_adapter": shared_name,
            "domain_adapters": domain_adapter_names,
            "pseudo_target_rotation": bool(train_cfg.get("pseudo_target_rotation", True)),
            "orthogonal_residual": bool(train_cfg.get("orthogonal_residual", True)),
            "inference_adapter": "Shared LoRA only",
            "trainable_parameters": sum(parameter.numel() for parameter in trainable),
        })

    def next_row(domain: str) -> dict[str, Any]:
        rows = rows_by_domain[domain]
        row = rows[positions[domain] % len(rows)]
        positions[domain] += 1
        return row

    pseudo_rotation = bool(train_cfg.get("pseudo_target_rotation", True))
    pseudo_weight = float(train_cfg.get("pseudo_target_weight", 1.0))
    domain_weight = float(train_cfg.get("domain_residual_weight", 1.0))
    log_every = int(train_cfg.get("logging_steps", 1))
    model.train()
    for step in range(1, max_steps + 1):
        inner_domains, pseudo_domain = _choose_inner_and_pseudo_domains(
            source_domains, step, pseudo_rotation
        )
        inner_gradients: list[dict[str, torch.Tensor]] = []
        reward_by_domain: dict[str, float] = {}
        signal_by_domain: dict[str, bool] = {}
        for domain in inner_domains:
            row = next_row(domain)
            groups = {"shared": shared_parameters, "domain": domain_parameters[domain]}
            if sequential_grad:
                rollouts, advantages, reward, has_signal = _row_view_data(
                    unwrapped, processor, row, prompt_text, train_cfg,
                    [shared_name, domain_adapter_names[domain]],
                )
                # Accumulate detached gradients from each image view, so the
                # graph for the first view is released before the second
                # backward pass begins.
                gradients = {
                    group: {
                        key: parameter.detach().new_zeros(parameter.shape)
                        for key, parameter in params.items()
                    }
                    for group, params in groups.items()
                }
                for rollout in rollouts:
                    view_loss = _view_loss_from_rollout(
                        model, unwrapped, rollout, advantages,
                        [shared_name, domain_adapter_names[domain]], reference_name, train_cfg,
                    )
                    view_gradients = _gradient_for_parameter_groups(view_loss, groups)
                    for group in gradients:
                        for key in gradients[group]:
                            gradients[group][key].add_(view_gradients[group][key], alpha=0.5)
                    del view_gradients, view_loss
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            else:
                loss, reward, has_signal = _row_loss(
                    model, unwrapped, processor, row, prompt_text, train_cfg,
                    [shared_name, domain_adapter_names[domain]], reference_name,
                )
                gradients = _gradient_for_parameter_groups(loss, groups)
            # Use the source loss gradient with respect to Shared LoRA as the
            # common coordinate for consensus. The routed residual is later
            # copied onto the corresponding Domain LoRA tensors, which have the
            # same shape and target modules by construction.
            inner_gradients.append(
                _offload_gradient(gradients["shared"])
                if cisr_enabled
                else gradients["shared"]
            )
            del gradients
            if cisr_enabled and torch.cuda.is_available():
                # Rollout prompt/completion tensors are no longer needed once
                # their detached gradient has been copied to host memory.
                del rollouts, advantages
                torch.cuda.empty_cache()
            reward_by_domain[domain] = reward
            signal_by_domain[domain] = has_signal
        rank_transfer_utilities: dict[str, dict[int, float]] = {}
        rank_transfer_gates: dict[str, dict[int, float]] = {}
        rank_consensus: dict[int, float] = {}
        pseudo_gradient_update_weight = 0.0
        if cisr_enabled:
            (
                shared_gradient,
                residual_gradients,
                rank_transfer_utilities,
                rank_transfer_gates,
                rank_consensus,
            ) = cisr_rankwise_route(
                inner_gradients,
                inner_domains,
                orthogonalize=bool(train_cfg.get("orthogonal_residual", True)),
                domain_weight=domain_weight,
                temperature=float(train_cfg.get("transfer_utility_temperature", 0.25)),
                threshold=float(train_cfg.get("transfer_utility_threshold", 0.0)),
                use_consensus=use_rank_consensus,
                use_utility=use_rank_utility,
            )
        else:
            shared_gradient, residual_gradients = route_consensus_residual(
                inner_gradients, inner_domains,
                orthogonalize=bool(train_cfg.get("orthogonal_residual", True)),
                domain_weight=domain_weight,
            )
        transfer_utilities: dict[str, float] = {}
        transfer_gates: dict[str, float] = {}
        meta_reward = None
        if pseudo_domain is not None:
            meta_row = next_row(pseudo_domain)
            transfer_gating = bool(train_cfg.get("transfer_utility_gating", False))
            meta_update_weight = float(train_cfg.get("pseudo_target_update_weight", 0.0))
            if sequential_grad:
                meta_rollouts, meta_advantages, meta_reward, meta_has_signal = _row_view_data(
                    unwrapped, processor, meta_row, prompt_text, train_cfg,
                    shared_name,
                )
                meta_gradient = None
                if cisr_enabled or not transfer_gating or meta_update_weight != 0.0:
                    meta_gradient = {
                        key: parameter.detach().new_zeros(parameter.shape)
                        for key, parameter in shared_parameters.items()
                    }
                    for meta_rollout in meta_rollouts:
                        meta_loss = _view_loss_from_rollout(
                            model, unwrapped, meta_rollout, meta_advantages,
                            shared_name, reference_name, train_cfg,
                        )
                        one_grad = _gradient_for_parameter_groups(
                            meta_loss, {"shared": shared_parameters}
                        )["shared"]
                        for key in meta_gradient:
                            meta_gradient[key].add_(one_grad[key], alpha=0.5)
                        del one_grad, meta_loss
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                if cisr_enabled and meta_gradient is not None:
                    meta_gradient = _offload_gradient(meta_gradient)
                    del meta_rollouts, meta_advantages
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
            else:
                meta_loss, meta_reward, meta_has_signal = _row_loss(
                    model, unwrapped, processor, meta_row, prompt_text, train_cfg,
                    shared_name, reference_name,
                )
                meta_gradient = _gradient_for_parameter_groups(
                    meta_loss, {"shared": shared_parameters}
                )["shared"]
            if cisr_enabled:
                rank_transfer_utilities = _rankwise_counterfactual_utilities(
                    inner_gradients, inner_domains, meta_gradient
                )
                (
                    shared_gradient,
                    residual_gradients,
                    rank_transfer_utilities,
                    rank_transfer_gates,
                    rank_consensus,
                ) = cisr_rankwise_route(
                    inner_gradients,
                    inner_domains,
                    validation_gradient=meta_gradient,
                    utility_scores=rank_transfer_utilities,
                    temperature=float(train_cfg.get("transfer_utility_temperature", 0.25)),
                    threshold=float(train_cfg.get("transfer_utility_threshold", 0.0)),
                    orthogonalize=bool(train_cfg.get("orthogonal_residual", True)),
                    domain_weight=domain_weight,
                    use_consensus=use_rank_consensus,
                    use_utility=use_rank_utility,
                )
                if meta_gradient is not None:
                    for key in shared_gradient:
                        shared_gradient[key] = (
                            shared_gradient[key] + meta_update_weight * meta_gradient[key]
                        )
                pseudo_gradient_update_weight = meta_update_weight
            elif transfer_gating:
                # Evaluate each source update by a real temporary Shared-LoRA
                # update on the held-out pseudo-target rollouts.  This finite
                # difference score, rather than a fixed gradient mean, drives
                # the Shared/Domain routing weights.
                transfer_utilities = _virtual_transfer_utilities(
                    model,
                    unwrapped,
                    shared_parameters,
                    inner_gradients,
                    inner_domains,
                    meta_rollouts if sequential_grad else [],
                    meta_advantages if sequential_grad else torch.zeros(
                        int(train_cfg.get("num_generations", 2)),
                        device=next(model.parameters()).device,
                    ),
                    shared_name,
                    reference_name,
                    train_cfg,
                )
                (
                    shared_gradient,
                    residual_gradients,
                    transfer_utilities,
                    transfer_gates,
                ) = transfer_utility_gated_route(
                    inner_gradients,
                    inner_domains,
                    validation_gradient=meta_gradient,
                    utility_scores=transfer_utilities or None,
                    temperature=float(train_cfg.get("transfer_utility_temperature", 0.25)),
                    threshold=float(train_cfg.get("transfer_utility_threshold", 0.0)),
                    orthogonalize=bool(train_cfg.get("orthogonal_residual", True)),
                    domain_weight=domain_weight,
                )
                # Optional first-order meta update.  Keep this at zero for the
                # strict leave-one-domain validation interpretation.
                if meta_gradient is not None:
                    for key in shared_gradient:
                        shared_gradient[key] = (
                            shared_gradient[key] + meta_update_weight * meta_gradient[key]
                        )
                pseudo_gradient_update_weight = meta_update_weight
            else:
                assert meta_gradient is not None
                for key in shared_gradient:
                    shared_gradient[key] = shared_gradient[key] + pseudo_weight * meta_gradient[key]
                pseudo_gradient_update_weight = pseudo_weight
            reward_by_domain[pseudo_domain] = meta_reward
            signal_by_domain[pseudo_domain] = meta_has_signal

        # When every domain in this step has a tied rollout group, the advantage
        # term of the objective is identically zero and the only surviving
        # gradient is beta * KL. AdamW turns that signal-free gradient into a
        # full-size per-parameter step, so it drifts the Shared LoRA off the SFT
        # optimum without teaching anything. Opt in to skipping those steps.
        skip_without_signal = bool(train_cfg.get("skip_update_without_signal", False))
        apply_update = True
        if skip_without_signal:
            # A pseudo-target advantage should trigger an optimizer update only
            # when its gradient is actually added to the update. When its
            # weight is zero, it only scores source directions for routing.
            has_signal = any(signal_by_domain.get(domain, False) for domain in inner_domains)
            if pseudo_domain is not None and pseudo_gradient_update_weight != 0.0:
                has_signal = has_signal or signal_by_domain.get(pseudo_domain, False)
            # The decision must be identical on every rank; otherwise the
            # gradient all-reduce below would deadlock.
            signal_flag = torch.tensor(
                float(has_signal), device=next(model.parameters()).device
            )
            apply_update = bool(accelerator.reduce(signal_flag, reduction="sum").item() > 0)

        if apply_update:
            # Synchronize manually routed gradients across processes before the
            # AdamW update.
            for key, parameter in shared_parameters.items():
                parameter.grad = accelerator.reduce(
                    shared_gradient[key].to(parameter), reduction="mean"
                ).to(parameter)
            for domain, parameters in domain_parameters.items():
                routed = residual_gradients.get(domain)
                for key, parameter in parameters.items():
                    gradient = (
                        parameter.detach().new_zeros(parameter.shape)
                        if routed is None
                        else routed[key]
                    )
                    parameter.grad = accelerator.reduce(
                        gradient.to(parameter), reduction="mean"
                    ).to(parameter)
            accelerator.clip_grad_norm_(trainable, float(train_cfg.get("max_grad_norm", 1.0)))
            optimizer.step()
        # The scheduler tracks attempted steps, not applied updates: max_steps
        # defines the horizon passed to get_scheduler as num_training_steps, so
        # freezing it on skipped steps would leave the cosine schedule unfinished
        # and pin the reported learning rate at its warmup start value.
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        if accelerator.is_main_process and step % log_every == 0:
            log_row = {
                "step": step,
                "inner_domains": inner_domains,
                "pseudo_target_domain": pseudo_domain,
                "mean_reward": sum(reward_by_domain.values()) / len(reward_by_domain),
                "reward_by_domain": reward_by_domain,
                "advantage_signal_by_domain": signal_by_domain,
                "shared_gradient_norm": _gradient_norm(shared_gradient),
                "domain_gradient_norms": {
                    domain: _gradient_norm(gradient)
                    for domain, gradient in residual_gradients.items()
                },
                "transfer_utility_by_domain": transfer_utilities,
                "transfer_gate_by_domain": transfer_gates,
                "rank_transfer_utility": rank_transfer_utilities,
                "rank_transfer_gate": rank_transfer_gates,
                "rank_consensus_score": rank_consensus,
                "selected_shared_rank_fraction": (
                    sum(
                        1 for rank in rank_consensus
                        if (
                            not use_rank_consensus
                            or rank_consensus[rank] > 0.0
                        )
                        and (
                            not use_rank_utility
                            or any(
                                rank_transfer_gates.get(domain, {}).get(rank, 0.0) > 0.5
                                for domain in inner_domains
                            )
                        )
                    ) / max(len(rank_consensus), 1)
                    if cisr_enabled else None
                ),
                "optimizer_updated": apply_update,
                "learning_rate": scheduler.get_last_lr()[0],
            }
            print(json.dumps(log_row), flush=True)
            with (output_dir / "train_metrics.jsonl").open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(log_row) + "\n")

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        set_active_adapters(unwrapped, shared_name)
        shared_path = _export_adapter(unwrapped, output_dir / "adapters", shared_name)
        domain_paths = {
            domain: str(_export_adapter(unwrapped, output_dir / "adapters", adapter_name))
            for domain, adapter_name in domain_adapter_names.items()
        }
        processor.save_pretrained(output_dir)
        write_json(output_dir / "adapter_paths.json", {
            "shared_adapter_path": str(shared_path),
            "domain_adapter_paths": domain_paths,
            "inference": "Use shared_adapter_path only for an unseen target domain.",
        })
