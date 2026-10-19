"""Tensor utilities for consensus-residual LoRA gradient routing.

The operators in this module are deliberately independent of PEFT and the VLM
trainer.  They operate on a common dictionary of tensors so that the routing
rule can be tested without loading a multi-billion-parameter model.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch


def _zeros_like(reference: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: value.new_zeros(value.shape) for name, value in reference.items()}


def consensus_gradient(
    gradients: Sequence[Mapping[str, torch.Tensor]],
    weights: Sequence[float] | None = None,
) -> dict[str, torch.Tensor]:
    """Return a weighted mean in the shared LoRA parameter coordinate system."""
    if not gradients:
        raise ValueError("At least one domain gradient is required.")
    names = set(gradients[0])
    if any(set(item) != names for item in gradients[1:]):
        raise ValueError("All domain gradients must use the same parameter keys.")
    if weights is None:
        weights = [1.0] * len(gradients)
    if len(weights) != len(gradients):
        raise ValueError("Gradient weights must match the number of domains.")
    total = float(sum(weights))
    if total <= 0:
        raise ValueError("Gradient weights must have a positive sum.")
    result = _zeros_like(gradients[0])
    for weight, gradient in zip(weights, gradients, strict=True):
        for name in result:
            result[name].add_(gradient[name], alpha=float(weight) / total)
    return result


def residual_gradients(
    gradients: Sequence[Mapping[str, torch.Tensor]],
    shared: Mapping[str, torch.Tensor],
    orthogonalize: bool = True,
    epsilon: float = 1e-12,
) -> list[dict[str, torch.Tensor]]:
    """Split each domain gradient into a residual and optionally its orthogonal part.

    Orthogonalization is a first-order optimization-direction constraint.  It
    should not be interpreted as a proof that the resulting adapter contains
    only semantic domain knowledge.
    """
    shared_norm_sq = sum((value.float() * value.float()).sum() for value in shared.values())
    denominator = shared_norm_sq.clamp_min(float(epsilon))
    residuals: list[dict[str, torch.Tensor]] = []
    for gradient in gradients:
        residual = {name: gradient[name] - shared[name] for name in shared}
        if orthogonalize and float(shared_norm_sq.detach()) > epsilon:
            dot = sum((residual[name].float() * shared[name].float()).sum() for name in shared)
            scale = dot / denominator
            residual = {
                name: residual[name] - scale.to(residual[name]) * shared[name]
                for name in shared
            }
        residuals.append(residual)
    return residuals


def route_consensus_residual(
    gradients: Sequence[Mapping[str, torch.Tensor]],
    domain_names: Sequence[str],
    orthogonalize: bool = True,
    domain_weight: float = 1.0,
) -> tuple[dict[str, torch.Tensor], dict[str, dict[str, torch.Tensor]]]:
    """Compute the Shared LoRA gradient and one residual per Domain LoRA."""
    if len(gradients) != len(domain_names):
        raise ValueError("Each domain must have exactly one gradient dictionary.")
    shared = consensus_gradient(gradients)
    residuals = residual_gradients(gradients, shared, orthogonalize=orthogonalize)
    routed = {
        domain: {name: value * float(domain_weight) for name, value in residual.items()}
        for domain, residual in zip(domain_names, residuals, strict=True)
    }
    return shared, routed


def transfer_utility_gated_route(
    gradients: Sequence[Mapping[str, torch.Tensor]],
    domain_names: Sequence[str],
    validation_gradient: Mapping[str, torch.Tensor] | None = None,
    utility_scores: Mapping[str, float] | None = None,
    temperature: float = 0.25,
    threshold: float = 0.0,
    orthogonalize: bool = True,
    domain_weight: float = 1.0,
) -> tuple[
    dict[str, torch.Tensor],
    dict[str, dict[str, torch.Tensor]],
    dict[str, float],
    dict[str, float],
]:
    """Route gradients using first-order leave-one-domain transfer utility.

    ``validation_gradient`` is the GRPO loss gradient on a held-out
    (pseudo-target) domain.  For a proposed update ``theta <- theta - eta*g``
    the first-order validation-loss change is ``-eta <g_val, g>``.  Therefore
    ``utility_scores`` may provide finite-difference utility measured by an
    actual temporary parameter update and held-out validation loss.  When it
    is absent, the function falls back to the first-order cosine proxy.
    """
    if len(gradients) != len(domain_names):
        raise ValueError("Each domain must have exactly one gradient dictionary.")
    if not gradients:
        raise ValueError("At least one domain gradient is required.")
    if temperature <= 0:
        raise ValueError("Transfer-utility temperature must be positive.")
    names = set(gradients[0])
    if any(set(item) != names for item in gradients[1:]):
        raise ValueError("All domain gradients must use the same parameter keys.")

    def norm(gradient: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return sum((value.float() * value.float()).sum() for value in gradient.values()).sqrt()

    reference = gradients[0] if validation_gradient is None else validation_gradient
    reference_norm = norm(reference).clamp_min(1e-12)
    utilities: dict[str, float] = {}
    gates: dict[str, float] = {}
    weights: list[float] = []
    for domain, gradient in zip(domain_names, gradients, strict=True):
        if utility_scores is not None:
            if domain not in utility_scores:
                raise ValueError(f"Missing transfer utility for domain {domain!r}.")
            utility_value = float(utility_scores[domain])
            utility = torch.tensor(utility_value, device=reference_norm.device)
        else:
            utility = sum(
                (gradient[name].float() * reference[name].float()).sum() for name in names
            ) / (norm(gradient).clamp_min(1e-12) * reference_norm)
            utility_value = float(utility.detach().cpu())
        gate = float(
            torch.sigmoid((utility - float(threshold)) / float(temperature))
            .detach()
            .cpu()
        )
        utilities[domain] = utility_value
        gates[domain] = gate
        weights.append(gate)

    total = sum(weights)
    if total <= 1e-12:
        weights = [1.0] * len(weights)
        total = float(len(weights))
    shared = _zeros_like(gradients[0])
    for weight, gradient in zip(weights, gradients, strict=True):
        for name in shared:
            shared[name].add_(gradient[name], alpha=float(weight) / total)
    residuals = residual_gradients(gradients, shared, orthogonalize=orthogonalize)
    routed = {
        domain: {name: value * float(domain_weight) for name, value in residual.items()}
        for domain, residual in zip(domain_names, residuals, strict=True)
    }
    return shared, routed, utilities, gates


def _infer_lora_rank(gradient: Mapping[str, torch.Tensor]) -> int:
    """Infer the common LoRA rank from adapter-independent parameter keys."""
    ranks: set[int] = set()
    for name, value in gradient.items():
        if "lora_A." in name and value.ndim >= 1:
            ranks.add(int(value.shape[0]))
        elif "lora_B." in name and value.ndim >= 1:
            ranks.add(int(value.shape[-1]))
    if not ranks:
        return 1
    if len(ranks) != 1:
        raise ValueError(f"LoRA gradients use inconsistent ranks: {sorted(ranks)}")
    return ranks.pop()


def rankwise_gradient_components(
    gradient: Mapping[str, torch.Tensor], rank: int | None = None
) -> list[dict[str, torch.Tensor]]:
    """Split a LoRA gradient into rank-coordinate components.

    A LoRA update is parameterized by ``A`` with shape ``[r, in]`` and ``B``
    with shape ``[out, r]``.  Each component owns one row of ``A`` and the
    matching column of ``B``.  This is a coordinate decomposition (the sum of
    all components is exactly the input gradient), while avoiding construction
    of a dense parameter-space covariance matrix.  Non-LoRA coordinates, if
    present, are assigned to rank zero so the operator remains lossless.
    """
    inferred = _infer_lora_rank(gradient)
    num_ranks = inferred if rank is None else int(rank)
    if num_ranks <= 0 or num_ranks != inferred:
        raise ValueError(f"Expected rank {inferred}, got {num_ranks}.")
    components = [_zeros_like(gradient) for _ in range(num_ranks)]
    for name, value in gradient.items():
        if "lora_A." in name:
            if value.ndim < 1 or value.shape[0] != num_ranks:
                raise ValueError(f"Invalid lora_A shape for {name!r}: {tuple(value.shape)}")
            for index in range(num_ranks):
                components[index][name][index].copy_(value[index])
        elif "lora_B." in name:
            if value.ndim < 1 or value.shape[-1] != num_ranks:
                raise ValueError(f"Invalid lora_B shape for {name!r}: {tuple(value.shape)}")
            for index in range(num_ranks):
                components[index][name][..., index].copy_(value[..., index])
        else:
            components[0][name].copy_(value)
    return components


def rank_consensus_scores(
    components_by_domain: Sequence[Sequence[Mapping[str, torch.Tensor]]],
) -> list[float]:
    """Return cross-domain cosine agreement for every LoRA rank coordinate."""
    if not components_by_domain:
        raise ValueError("At least one domain component sequence is required.")
    num_ranks = len(components_by_domain[0])
    if num_ranks == 0 or any(len(item) != num_ranks for item in components_by_domain):
        raise ValueError("All domains must expose the same non-empty rank decomposition.")

    def cosine(left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]) -> float:
        dot = flatten_dot(left, right)
        left_norm = flatten_dot(left, left).sqrt().clamp_min(1e-12)
        right_norm = flatten_dot(right, right).sqrt().clamp_min(1e-12)
        if float(left_norm.detach()) <= 1e-12 or float(right_norm.detach()) <= 1e-12:
            return 0.0
        return float((dot / (left_norm * right_norm)).detach().cpu())

    if len(components_by_domain) == 1:
        return [1.0] * num_ranks
    scores: list[float] = []
    for rank_index in range(num_ranks):
        values = [item[rank_index] for item in components_by_domain]
        pairwise = [
            cosine(values[left], values[right])
            for left in range(len(values))
            for right in range(left + 1, len(values))
        ]
        scores.append(float(sum(pairwise) / len(pairwise)))
    return scores


def cisr_rankwise_route(
    gradients: Sequence[Mapping[str, torch.Tensor]],
    domain_names: Sequence[str],
    validation_gradient: Mapping[str, torch.Tensor] | None = None,
    utility_scores: Mapping[str, Mapping[int, float] | Sequence[float]] | None = None,
    temperature: float = 0.25,
    threshold: float = 0.0,
    orthogonalize: bool = True,
    domain_weight: float = 1.0,
    use_consensus: bool = True,
    use_utility: bool = True,
) -> tuple[
    dict[str, torch.Tensor],
    dict[str, dict[str, torch.Tensor]],
    dict[str, dict[int, float]],
    dict[str, dict[int, float]],
    dict[int, float],
]:
    """Route LoRA gradients with Counterfactual Invariant Subspace Routing.

    Each domain gradient is decomposed by LoRA rank.  In the full mode, a rank
    receives Shared capacity only when (1) its held-out transfer utility is
    positive and (2) its source-domain components agree.  The two boolean
    switches expose controlled ablations: ``use_consensus=False`` gives the
    utility-only route and ``use_utility=False`` gives the consensus-only
    route.  The remaining rank-coordinate
    residual is sent to that domain's adapter and projected orthogonally to the
    routed Shared component.  ``utility_scores`` are normally measured on a
    pseudo-target rollout; a validation-gradient cosine is used as a cheap
    fallback for tests and ablations.
    """
    if not gradients or len(gradients) != len(domain_names):
        raise ValueError("Each domain must have exactly one gradient dictionary.")
    if temperature <= 0:
        raise ValueError("CISR temperature must be positive.")
    names = set(gradients[0])
    if any(set(item) != names for item in gradients[1:]):
        raise ValueError("All domain gradients must use the same parameter keys.")
    rank_components = [rankwise_gradient_components(item) for item in gradients]
    num_ranks = len(rank_components[0])
    consensus = rank_consensus_scores(rank_components)
    validation_components = (
        rankwise_gradient_components(validation_gradient, num_ranks)
        if validation_gradient is not None
        else None
    )
    utilities: dict[str, dict[int, float]] = {domain: {} for domain in domain_names}
    gates: dict[str, dict[int, float]] = {domain: {} for domain in domain_names}
    for domain_index, domain in enumerate(domain_names):
        for rank_index in range(num_ranks):
            if utility_scores is not None:
                values = utility_scores.get(domain)
                if values is None:
                    raise ValueError(f"Missing CISR utility scores for domain {domain!r}.")
                if isinstance(values, Mapping):
                    raw_value = values.get(rank_index, values.get(str(rank_index)))
                    if raw_value is None:
                        raise ValueError(
                            f"Missing CISR utility for domain {domain!r}, rank {rank_index}."
                        )
                    utility_value = float(raw_value)
                else:
                    utility_value = float(values[rank_index])
            elif validation_components is not None:
                source = rank_components[domain_index][rank_index]
                heldout = validation_components[rank_index]
                utility_value = float(
                    flatten_dot(source, heldout)
                    / (
                        flatten_dot(source, source).sqrt().clamp_min(1e-12)
                        * flatten_dot(heldout, heldout).sqrt().clamp_min(1e-12)
                    )
                )
            else:
                utility_value = 0.0
            gate = (
                float(
                    torch.sigmoid(
                        torch.tensor((utility_value - threshold) / temperature)
                    ).item()
                )
                if use_utility
                else 1.0
            )
            utilities[domain][rank_index] = utility_value
            gates[domain][rank_index] = gate

    shared_components = [_zeros_like(gradients[0]) for _ in range(num_ranks)]
    for rank_index in range(num_ranks):
        consensus_weight = max(consensus[rank_index], 0.0) if use_consensus else 1.0
        weights = [
            gates[domain][rank_index] * consensus_weight for domain in domain_names
        ]
        total = sum(weights)
        if total <= 1e-12:
            # A rank with no cross-domain agreement is deliberately excluded
            # from Shared LoRA; its complete update remains domain residual.
            continue
        for weight, components in zip(weights, rank_components, strict=True):
            for name in names:
                shared_components[rank_index][name].add_(
                    components[rank_index][name], alpha=weight / total
                )
    shared = _zeros_like(gradients[0])
    for component in shared_components:
        for name in names:
            shared[name].add_(component[name])

    routed: dict[str, dict[str, torch.Tensor]] = {}
    for domain_index, domain in enumerate(domain_names):
        residual = _zeros_like(gradients[0])
        for rank_index in range(num_ranks):
            delta = {
                name: rank_components[domain_index][rank_index][name]
                - shared_components[rank_index][name]
                for name in names
            }
            reference = shared_components[rank_index]
            denom = flatten_dot(reference, reference).clamp_min(1e-12)
            if orthogonalize and float(denom.detach()) > 1e-12:
                scale = flatten_dot(delta, reference) / denom
                delta = {
                    name: value - scale.to(value) * reference[name]
                    for name, value in delta.items()
                }
            for name in names:
                residual[name].add_(delta[name], alpha=float(domain_weight))
        routed[domain] = residual
    return shared, routed, utilities, gates, {index: value for index, value in enumerate(consensus)}


def flatten_dot(
    left: Mapping[str, torch.Tensor], right: Mapping[str, torch.Tensor]
) -> torch.Tensor:
    """Return a scalar dot product for diagnostics and unit tests."""
    if set(left) != set(right):
        raise ValueError("Gradient dictionaries must have identical keys.")
    return sum((left[name].float() * right[name].float()).sum() for name in left)


def adapter_parameter_key(name: str, adapter_name: str) -> str | None:
    """Map a PEFT parameter name to an adapter-independent coordinate key."""
    marker = f".{adapter_name}."
    if marker in name:
        return name.replace(marker, ".{adapter}.", 1)
    suffix = f".{adapter_name}"
    if name.endswith(suffix):
        return name[: -len(suffix)] + ".{adapter}"
    return None
