import pytest

from fs_pairgrpo.io_utils import write_jsonl
from fs_pairgrpo.manifest import audit_training_manifests, validate_domain_protocol


def config(sources, targets):
    return {"dataset": {"source_domains": sources, "target_domains": targets}}


def test_accepts_two_or_three_disjoint_source_domains():
    validate_domain_protocol(config(["real", "clipart"], ["sketch"]))
    validate_domain_protocol(config(["real", "clipart", "painting"], ["sketch"]))


def test_rejects_target_leakage():
    with pytest.raises(ValueError, match="unseen"):
        validate_domain_protocol(config(["real", "sketch"], ["sketch"]))


def test_rejects_standard_five_source_setting():
    with pytest.raises(ValueError, match="2 or 3"):
        validate_domain_protocol(
            config(["real", "clipart", "painting", "quickdraw", "infograph"], ["sketch"])
        )


def test_rejects_multiple_targets():
    with pytest.raises(ValueError, match="exactly one"):
        validate_domain_protocol(config(["real", "clipart"], ["sketch", "painting"]))


def test_manifest_audit_rejects_target_leakage(tmp_path):
    source_manifest = tmp_path / "source.jsonl"
    write_jsonl(
        source_manifest,
        [
            {"domain": "real", "label_id": 0, "class_name": "horse"},
            {"domain": "sketch", "label_id": 0, "class_name": "horse"},
        ],
    )
    value = config(["real", "clipart"], ["sketch"])
    value["data"] = {"source_manifest": str(source_manifest)}
    with pytest.raises(ValueError, match="forbidden"):
        audit_training_manifests(value, require_pairs=False)

