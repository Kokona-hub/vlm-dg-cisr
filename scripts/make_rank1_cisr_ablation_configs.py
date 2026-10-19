#!/usr/bin/env python3
"""Generate a controlled rank-1 routing ablation suite from a base config."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import yaml


MODES = {
    "mean_only": "mean_only",
    "consensus_only": "consensus_only",
    "utility_only": "utility_only",
    "full_cisr": "cisr_rankwise",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-config",
        default="configs/domainnet_3src012_to_4_qwen3vl8b_cisr_allclass10_medium.yaml",
    )
    parser.add_argument(
        "--config-dir",
        default="configs/domainnet_rank1_rotation_ablation",
    )
    parser.add_argument(
        "--run-root",
        default="runs/domainnet_3src012_to_4_allclass10_rank1",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    base_path = Path(args.base_config)
    config_dir = Path(args.config_dir)
    run_root = Path(args.run_root)
    base = yaml.safe_load(base_path.read_text(encoding="utf-8"))

    sft_config = copy.deepcopy(base)
    sft_config["seed"] = args.seed
    sft_config["lora"]["r"] = 1
    sft_config["lora"]["alpha"] = 2
    sft_config["sft"]["output_dir"] = str(run_root / "sft_adapter")
    (config_dir / "rank1_sft.yaml").parent.mkdir(parents=True, exist_ok=True)
    (config_dir / "rank1_sft.yaml").write_text(
        yaml.safe_dump(sft_config, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )

    for name, mode in MODES.items():
        config = copy.deepcopy(base)
        config["seed"] = args.seed
        config["lora"]["r"] = 1
        # Preserve the base alpha/r scaling (32/16 = 2) at rank one.
        config["lora"]["alpha"] = 2
        config["sft"]["output_dir"] = str(run_root / "sft_adapter")
        config["crlora_grpo"]["sft_adapter_path"] = str(run_root / "sft_adapter")
        config["crlora_grpo"]["output_dir"] = str(run_root / name)
        config["crlora_grpo"]["routing_mode"] = mode
        config["crlora_grpo"]["pseudo_target_rotation"] = True
        config["crlora_grpo"]["pseudo_target_update_weight"] = 0.0
        config["crlora_grpo"]["orthogonal_residual"] = True
        config["evaluation"]["adapter_path"] = str(
            run_root / name / "adapters" / "shared" / "shared"
        )
        config["evaluation"]["output_dir"] = str(run_root / f"eval_{name}")
        path = config_dir / f"rank1_{name}.yaml"
        path.write_text(
            yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )

    print(f"Generated rank-1 configs in {config_dir}")
    print(f"Shared SFT adapter: {run_root / 'sft_adapter'}")
    print("Routing modes: " + ", ".join(MODES))


if __name__ == "__main__":
    main()
