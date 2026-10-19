import pytest

from fs_pairgrpo.io_utils import write_json
from fs_pairgrpo.reporting import aggregate_seed_metrics, summarize_metrics


def metric(path, accuracy, samples=100):
    write_json(
        path,
        {
            "num_samples": samples,
            "accuracy": accuracy,
            "macro_class_accuracy": accuracy - 0.01,
            "adapter_path": None,
        },
    )


def test_summary_writes_json_and_markdown(tmp_path):
    zero = tmp_path / "zero.json"
    sft = tmp_path / "sft.json"
    metric(zero, 0.4)
    metric(sft, 0.5)
    result = summarize_metrics(
        [f"zero={zero}", f"sft={sft}"], output_dir=tmp_path / "summary"
    )
    assert result["best_accuracy_method"] == "sft"
    assert (tmp_path / "summary" / "comparison.json").is_file()
    assert "| sft |" in (tmp_path / "summary" / "comparison.md").read_text()


def test_summary_rejects_mismatched_sample_counts(tmp_path):
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    metric(first, 0.4, samples=100)
    metric(second, 0.5, samples=90)
    with pytest.raises(ValueError, match="different sample counts"):
        summarize_metrics([f"first={first}", f"second={second}"])


def test_aggregate_seed_metrics_reports_mean_and_std(tmp_path):
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    metric(first, 0.4)
    metric(second, 0.6)
    result = aggregate_seed_metrics(
        [f"crlora@42={first}", f"crlora@43={second}"],
        output_dir=tmp_path / "aggregate",
    )
    row = result["methods"][0]
    assert row["accuracy_mean"] == pytest.approx(0.5)
    assert row["accuracy_std"] == pytest.approx(0.141421, rel=1e-3)
    assert (tmp_path / "aggregate" / "seed_aggregate.md").is_file()
