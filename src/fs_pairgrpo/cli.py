from __future__ import annotations

import argparse
import json


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fs-pairgrpo")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command, help_text in [
        ("prepare", "Build source and Fourier-pair manifests."),
        ("sft", "Run short-evidence QLoRA supervised fine-tuning."),
        ("grpo", "Continue the SFT adapter with paired GRPO."),
        (
            "crlora-grpo",
            "Train CRLoRA variants, including CISR-LoRA-GRPO rank-wise routing.",
        ),
        ("evaluate", "Evaluate an adapter on an unseen target domain."),
        ("evaluate-ollama", "Evaluate a local Ollama vision model on an unseen target domain."),
        ("diagnose", "Audit LoRA module matches, trainable parameters, and GPU memory."),
        ("summarize", "Compare completed target-domain metric JSON files."),
        ("aggregate-seeds", "Aggregate target-domain metrics across random seeds."),
    ]:
        child = subparsers.add_parser(command, help=help_text)
        if command in {"summarize", "aggregate-seeds"}:
            child.add_argument(
                "--metric",
                action="append",
                required=True,
                help="Repeat NAME=PATH_TO_METRICS_JSON for each method.",
            )
            child.add_argument("--output-dir", required=True)
        else:
            child.add_argument("--config", required=True)
        if command == "prepare":
            child.add_argument("--dry-run", action="store_true")
        elif command == "sft":
            child.add_argument("--include-fourier", action="store_true", default=None)
            child.add_argument("--output-dir", default=None)
        elif command == "grpo":
            child.add_argument("--mode", choices=["single", "independent_pair"], default=None)
            child.add_argument("--output-dir", default=None)
        elif command == "crlora-grpo":
            child.add_argument("--output-dir", default=None)
        elif command == "evaluate":
            child.add_argument("--adapter-path", default=None)
            child.add_argument("--base-model", action="store_true")
            child.add_argument("--output-dir", default=None)
            child.add_argument(
                "--image-list",
                default=None,
                help=(
                    "Evaluate only the images listed in this CSV (image_path column) or "
                    "newline-delimited text file. Paths are matched on their trailing "
                    "domain/class/filename components, so the list may come from a machine "
                    "that mounts the same dataset under a different root."
                ),
            )
        elif command == "evaluate-ollama":
            child.add_argument("--model", default=None)
            child.add_argument("--base-url", default=None)
            child.add_argument("--image-list", default=None)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "summarize":
        from .reporting import summarize_metrics

        print(
            json.dumps(
                summarize_metrics(args.metric, output_dir=args.output_dir),
                ensure_ascii=False,
                indent=2,
            )
        )
    elif args.command == "aggregate-seeds":
        from .reporting import aggregate_seed_metrics

        print(
            json.dumps(
                aggregate_seed_metrics(args.metric, output_dir=args.output_dir),
                ensure_ascii=False,
                indent=2,
            )
        )
    elif args.command == "prepare":
        from .manifest import build_manifests

        print(
            json.dumps(
                build_manifests(args.config, dry_run=args.dry_run), ensure_ascii=False, indent=2
            )
        )
    elif args.command == "sft":
        from .train_sft import run_sft

        run_sft(
            args.config,
            include_fourier_override=args.include_fourier,
            output_dir_override=args.output_dir,
        )
    elif args.command == "grpo":
        from .train_grpo import run_grpo

        run_grpo(
            args.config, mode_override=args.mode, output_dir_override=args.output_dir
        )
    elif args.command == "crlora-grpo":
        from .train_crlora import run_crlora_grpo

        run_crlora_grpo(args.config, output_dir_override=args.output_dir)
    elif args.command == "evaluate":
        from .evaluate import run_evaluation

        adapter_override = "" if args.base_model else args.adapter_path
        print(
            json.dumps(
                run_evaluation(
                    args.config,
                    adapter_override=adapter_override,
                    output_dir_override=args.output_dir,
                    image_list_override=args.image_list,
                ),
                ensure_ascii=False,
                indent=2,
            )
        )
    elif args.command == "evaluate-ollama":
        from .ollama_eval import run_ollama_evaluation

        print(
            json.dumps(
                run_ollama_evaluation(
                    args.config,
                    model_override=args.model,
                    base_url_override=args.base_url,
                    image_list_override=args.image_list,
                ),
                ensure_ascii=False,
                indent=2,
            )
        )
    elif args.command == "diagnose":
        from .diagnostics import diagnose_model

        diagnose_model(args.config)


if __name__ == "__main__":
    main()
