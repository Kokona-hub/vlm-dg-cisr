import pytest

from fs_pairgrpo.outcome_rl import normalized_group_advantages, outcome_rewards
from fs_pairgrpo.prompts import grpo_prompt


def test_independent_pair_reward_requires_both_views_for_maximum():
    originals = ["<ANSWER>airplane</ANSWER>", "<ANSWER>airplane</ANSWER>", "bad"]
    styled = ["<ANSWER>airplane</ANSWER>", "<ANSWER>dog</ANSWER>", "bad"]

    rewards = outcome_rewards(originals, "airplane", styled, format_bonus=0.0)

    assert rewards == [2.0, 0.5, 0.0]


def test_same_wrong_label_gets_no_pair_reward():
    rewards = outcome_rewards(
        ["<ANSWER>dog</ANSWER>"],
        "airplane",
        ["<ANSWER>dog</ANSWER>"],
        format_bonus=0.0,
    )
    assert rewards == [0.0]


def test_group_advantages_center_rewards_and_handle_ties():
    advantages = normalized_group_advantages([2.0, 0.5, 0.0])
    assert sum(advantages) == pytest.approx(0.0, abs=1e-6)
    assert advantages[0] > 0 > advantages[-1]
    assert normalized_group_advantages([1.0, 1.0]) == [0.0, 0.0]


def test_grpo_uses_the_single_image_contract_without_pair_disclosure():
    prompt = grpo_prompt(["airplane", "dog"], use_evidence=False)
    text = prompt[-1]["content"]
    assert "<ANSWER>" in text
    assert "ANSWER_1" not in text
    assert "same semantic label" not in text
