import pytest
import torch

from fs_pairgrpo.consensus import (
    adapter_parameter_key,
    cisr_rankwise_route,
    consensus_gradient,
    flatten_dot,
    rankwise_gradient_components,
    route_consensus_residual,
    transfer_utility_gated_route,
)
from fs_pairgrpo.train_crlora import _choose_inner_and_pseudo_domains


def gradient(first, second):
    return {
        "lora_A.{adapter}.weight": torch.tensor(first, dtype=torch.float32),
        "lora_B.{adapter}.weight": torch.tensor(second, dtype=torch.float32),
    }


def test_consensus_is_the_domain_mean():
    shared = consensus_gradient([gradient([1, 3], [2]), gradient([3, 5], [4])])
    assert torch.equal(shared["lora_A.{adapter}.weight"], torch.tensor([2.0, 4.0]))
    assert torch.equal(shared["lora_B.{adapter}.weight"], torch.tensor([3.0]))


def test_equal_domain_gradients_have_zero_residuals():
    value = gradient([2, -1], [3])
    _, residuals = route_consensus_residual([value, value], ["a", "b"])
    for residual in residuals.values():
        assert flatten_dot(residual, residual).item() == pytest.approx(0.0)


def test_orthogonal_residuals_are_first_order_orthogonal_to_shared_gradient():
    shared, residuals = route_consensus_residual(
        [gradient([3, 1], [2]), gradient([1, -3], [4])], ["a", "b"], orthogonalize=True
    )
    for residual in residuals.values():
        assert flatten_dot(shared, residual).item() == pytest.approx(0.0, abs=1e-6)


def test_adapter_coordinate_key_is_adapter_independent():
    assert (
        adapter_parameter_key("model.layers.0.lora_A.shared.weight", "shared")
        == "model.layers.0.lora_A.{adapter}.weight"
    )
    assert adapter_parameter_key("model.layers.0.lora_A.reference.weight", "shared") is None


def test_three_source_rotation_holds_out_one_source_but_two_source_keeps_both():
    sources = ["clipart", "infograph", "painting"]
    inner, pseudo = _choose_inner_and_pseudo_domains(sources, step=1, pseudo_rotation=True)
    assert pseudo == "clipart"
    assert inner == ["infograph", "painting"]
    inner, pseudo = _choose_inner_and_pseudo_domains(["clipart", "painting"], 1, True)
    assert inner == ["clipart", "painting"]
    assert pseudo is None


def test_transfer_utility_gate_prefers_source_aligned_with_heldout_descent():
    # Gradients are loss gradients.  A source gradient aligned with the
    # held-out loss gradient produces a descent step on the held-out domain
    # under theta <- theta - eta*g and should receive the larger gate.
    good = gradient([1, 0], [0])
    bad = gradient([-1, 0], [0])
    shared, _, utilities, gates = transfer_utility_gated_route(
        [good, bad], ["good", "bad"], validation_gradient=gradient([1, 0], [0])
    )
    assert utilities["good"] > utilities["bad"]
    assert gates["good"] > gates["bad"]
    assert torch.isfinite(shared["lora_A.{adapter}.weight"]).all()


def test_measured_transfer_utilities_override_first_order_proxy():
    positive = gradient([1, 0], [0])
    negative = gradient([-1, 0], [0])
    _, _, utilities, gates = transfer_utility_gated_route(
        [positive, negative],
        ["positive", "negative"],
        utility_scores={"positive": -0.2, "negative": 0.4},
    )
    assert utilities == {"positive": -0.2, "negative": 0.4}
    assert gates["negative"] > gates["positive"]


def _rank_gradient(a_values, b_values):
    return {
        "layer.lora_A.{adapter}.weight": torch.tensor(a_values, dtype=torch.float32),
        "layer.lora_B.{adapter}.weight": torch.tensor(b_values, dtype=torch.float32),
    }


def test_rankwise_components_reconstruct_the_full_lora_gradient():
    value = _rank_gradient([[1, 2, 3], [4, 5, 6]], [[1, 2], [3, 4], [5, 6]])
    components = rankwise_gradient_components(value)
    assert len(components) == 2
    for name in value:
        reconstructed = sum(
            (component[name] for component in components), torch.zeros_like(value[name])
        )
        assert torch.equal(reconstructed, value[name])
    assert torch.equal(components[0]["layer.lora_A.{adapter}.weight"][1], torch.zeros(3))
    assert torch.equal(components[1]["layer.lora_B.{adapter}.weight"][:, 0], torch.zeros(3))


def test_cisr_rankwise_route_prefers_positive_rank_transfer_utility():
    first = _rank_gradient([[1, 0], [1, 0]], [[1, 0], [1, 0]])
    second = _rank_gradient([[1, 0], [-1, 0]], [[1, 0], [-1, 0]])
    shared, residuals, utilities, gates, consensus = cisr_rankwise_route(
        [first, second],
        ["a", "b"],
        utility_scores={"a": {0: 0.8, 1: -0.8}, "b": {0: 0.8, 1: -0.8}},
    )
    assert gates["a"][0] > gates["a"][1]
    assert utilities["b"][0] == pytest.approx(0.8)
    assert 0 in consensus and 1 in consensus
    assert torch.isfinite(shared["layer.lora_A.{adapter}.weight"]).all()
    assert torch.equal(
        shared["layer.lora_A.{adapter}.weight"][1], torch.zeros(2)
    )
    assert flatten_dot(shared, residuals["a"]).item() == pytest.approx(0.0, abs=1e-6)


def test_rankwise_ablation_switches_disable_the_selected_factor():
    first = _rank_gradient([[1, 0], [1, 0]], [[1, 0], [1, 0]])
    second = _rank_gradient([[1, 0], [-1, 0]], [[1, 0], [-1, 0]])
    utilities = {"a": {0: 0.8, 1: -0.8}, "b": {0: 0.8, 1: -0.8}}

    _, _, _, consensus_only_gates, _ = cisr_rankwise_route(
        [first, second], ["a", "b"], utility_scores=utilities, use_utility=False
    )
    assert consensus_only_gates["a"] == {0: 1.0, 1: 1.0}

    _, _, _, utility_only_gates, _ = cisr_rankwise_route(
        [first, second], ["a", "b"], utility_scores=utilities, use_consensus=False
    )
    assert utility_only_gates["a"][0] > utility_only_gates["a"][1]


def test_rank1_mean_only_route_reduces_to_direct_domain_mean():
    first = _rank_gradient([[1.0, 2.0]], [[3.0], [4.0]])
    second = _rank_gradient([[5.0, 6.0]], [[7.0], [8.0]])
    expected = consensus_gradient([first, second])

    shared, residuals, _, gates, _ = cisr_rankwise_route(
        [first, second],
        ["a", "b"],
        utility_scores={"a": {0: -0.9}, "b": {0: 0.9}},
        use_consensus=False,
        use_utility=False,
    )

    for name in expected:
        assert torch.equal(shared[name], expected[name])
    assert gates == {"a": {0: 1.0}, "b": {0: 1.0}}
    for residual in residuals.values():
        assert flatten_dot(shared, residual).item() == pytest.approx(0.0, abs=1e-6)
