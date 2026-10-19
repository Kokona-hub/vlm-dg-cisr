import json
from pathlib import Path

import pytest

from fs_pairgrpo.evaluate import resolve_adapter_path


def _make_adapter(directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "adapter_config.json").write_text("{}\n", encoding="utf-8")
    return directory


def _config(root: Path, variant: str = "crlora_grpo") -> dict:
    return {"crlora_grpo": {"output_dir": str(root / variant)}}


def test_valid_adapter_directory_is_used_as_is(tmp_path):
    adapter = _make_adapter(tmp_path / "crlora_grpo" / "adapters" / "shared" / "shared")

    resolved, substituted = resolve_adapter_path(_config(tmp_path), adapter)

    assert resolved == adapter.resolve()
    assert substituted is None


def test_nested_peft_layout_is_accepted(tmp_path):
    # PEFT variants write <destination>/<adapter_name>/adapter_config.json, so a
    # config pointing at <destination> must still resolve.
    nested = _make_adapter(tmp_path / "crlora_grpo" / "adapters" / "shared" / "shared")

    resolved, substituted = resolve_adapter_path(
        _config(tmp_path), nested.parent
    )

    assert resolved == nested.resolve()
    assert substituted is None


def test_recorded_shared_adapter_is_discovered_for_a_stale_path(tmp_path):
    root = tmp_path / "runs" / "experiment"
    shared = _make_adapter(root / "tu_crlora_grpo" / "adapters" / "shared" / "shared")
    (root / "tu_crlora_grpo").mkdir(parents=True, exist_ok=True)
    (root / "tu_crlora_grpo" / "adapter_paths.json").write_text(
        json.dumps({"shared_adapter_path": str(shared)}), encoding="utf-8"
    )
    stale = root / "crlora_grpo" / "adapters" / "shared" / "shared"

    resolved, substituted = resolve_adapter_path(_config(tmp_path, "runs/experiment/crlora_grpo"), stale)

    assert resolved == shared.resolve()
    assert substituted == stale


def test_ambiguous_discovered_adapters_fail_loudly(tmp_path):
    root = tmp_path / "runs" / "experiment"
    for variant in ("tu_crlora_grpo", "sft_crlora_grpo"):
        shared = _make_adapter(root / variant / "adapters" / "shared" / "shared")
        (root / variant / "adapter_paths.json").write_text(
            json.dumps({"shared_adapter_path": str(shared)}), encoding="utf-8"
        )
    # The configured directory was never produced, and two other variants record
    # a Shared LoRA, so guessing would risk evaluating the wrong adapter.
    stale = root / "crlora_grpo" / "adapters" / "shared" / "shared"

    with pytest.raises(FileNotFoundError, match="--adapter-path"):
        resolve_adapter_path(_config(tmp_path, "runs/experiment/crlora_grpo"), stale)


def test_missing_adapter_reports_the_resolved_path(tmp_path):
    stale = tmp_path / "runs" / "experiment" / "crlora_grpo" / "adapters" / "shared" / "shared"

    with pytest.raises(FileNotFoundError) as error:
        resolve_adapter_path(_config(tmp_path, "runs/experiment/crlora_grpo"), stale)

    assert str(stale.resolve()) in str(error.value)
