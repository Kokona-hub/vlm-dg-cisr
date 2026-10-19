from fs_pairgrpo.parsing import parse_single_prediction
from fs_pairgrpo.rewards import (
    all_views_correct_reward,
    evidence_quality_reward,
    paired_correctness_reward,
    strict_format_reward,
)


def test_pair_parsing_and_rewards():
    completions = [
        (
            "<EVIDENCE_1>mane and hooves</EVIDENCE_1><ANSWER_1>horse</ANSWER_1>"
            "<EVIDENCE_2>long face and mane</EVIDENCE_2><ANSWER_2>horse</ANSWER_2>"
        ),
        (
            "<EVIDENCE_1>mane</EVIDENCE_1><ANSWER_1>horse</ANSWER_1>"
            "<EVIDENCE_2>background</EVIDENCE_2><ANSWER_2>dog</ANSWER_2>"
        ),
    ]
    labels = ["horse", "horse"]
    assert paired_correctness_reward([], completions, labels) == [1.0, 0.5]
    assert all_views_correct_reward([], completions, labels) == [1.0, 0.0]
    assert strict_format_reward([], completions) == [1.0, 1.0]
    evidence_scores = evidence_quality_reward(
        [],
        completions,
        [["mane", "long face", "hooves"], ["mane", "long face", "hooves"]],
        labels,
    )
    assert evidence_scores[0] > evidence_scores[1]
    assert evidence_scores[1] == 0.0


def test_metadata_is_repeated_for_multiple_generations():
    completions = [
        "<ANSWER_1>horse</ANSWER_1><ANSWER_2>horse</ANSWER_2>",
        "<ANSWER_1>dog</ANSWER_1><ANSWER_2>dog</ANSWER_2>",
        "<ANSWER_1>cat</ANSWER_1><ANSWER_2>cat</ANSWER_2>",
        "<ANSWER_1>cat</ANSWER_1><ANSWER_2>dog</ANSWER_2>",
    ]
    # Two prompts, two generations each. TRL may repeat metadata before calling rewards.
    assert paired_correctness_reward([], completions, ["horse", "cat"]) == [1.0, 0.0, 1.0, 0.5]


def test_single_prediction():
    answer, evidence = parse_single_prediction(
        "<EVIDENCE>two wheels and pedals</EVIDENCE><ANSWER>Bicycle</ANSWER>"
    )
    assert answer == "bicycle"
    assert evidence == "two wheels and pedals"
