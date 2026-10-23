"""
BigCodeBench Benchmark Data Collection Script:
Runs open-weight LLMs (e.g. Qwen2.5-Coder-7B-Instruct) on BigCodeBench coding tasks,
extracts residual-stream activations at specified anchors (post_gen, pre_gen, thought_end), and
persists shard arrays and trajectory summaries for interpretability research.
"""

import argparse
from pathlib import Path
import sys

# Ensure repository root is in Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.activations import ActivationExtractor
from src.benchmark.bigcodebench.bigcodebench import BigCodeBenchmark
from src.runner import Runner


def parse_args():
    parser = argparse.ArgumentParser(description="BigCodeBench Activation Harvesting")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-Coder-7B-Instruct",
        help="Hugging Face model ID (default: Qwen/Qwen2.5-Coder-7B-Instruct)",
    )
    parser.add_argument(
        "--subset",
        type=str,
        choices=["hard", "full"],
        default="full",
        help="BigCodeBench benchmark subset: hard (148 tasks) or full (1140 tasks) (default: full)",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["instruct", "complete"],
        default="instruct",
        help="Task prompt formulation: instruct or complete (default: instruct)",
    )
    parser.add_argument(
        "--num_tasks",
        type=int,
        default=None,
        help="Number of BigCodeBench tasks to run (default: None, runs all tasks in subset)",
    )
    parser.add_argument(
        "--n_rollouts",
        type=int,
        default=5,
        help="Number of rollouts per task (default: 5 rollouts)",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=5,
        help="Maximum turns per episode allowing iterative test-error repair (default: 5)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7 for stochastic rollouts)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=15.0,
        help="Timeout in seconds for code evaluation execution (default: 15.0s)",
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=[4, 10, 16, 18, 20, 22, 24, 27],
        help="Transformer layer indices to extract (default: 4 10 16 18 20 22 24 27)",
    )
    parser.add_argument(
        "--anchors",
        type=str,
        nargs="+",
        default=["post_gen", "pre_gen", "thought_end"],
        help="Anchor tokens to extract (default: post_gen pre_gen thought_end)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="artifacts/bigcodebench/qwen2.5-7b/qwen2.5-7b",
        help="Directory to persist features and summaries",
    )
    parser.add_argument("--model_revision", default=None, help="Model revision to resolve and pin")
    parser.add_argument("--split_manifest", default=None, help="Existing task-to-probe-split manifest JSON")
    parser.add_argument("--source_split", default=None, help="Optional exact source filter; must match --release")
    parser.add_argument("--dataset_config", default=None, help="Hugging Face dataset configuration")
    parser.add_argument("--dataset_revision", default=None, help="Dataset revision to resolve and pin")
    parser.add_argument("--release", default=BigCodeBenchmark.DEFAULT_RELEASE,
                        help="One official release partition (default: v0.1.4); never combines releases")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Maximum generated tokens per decision (default: 512)")
    return parser.parse_args()


def main():
    args = parse_args()
    output_path = Path(args.output_dir)
    Runner.validate_output_directory(output_path)

    # 1. Initialize Benchmark first to know exact task count
    print("=" * 65)
    print("BigCodeBench Activation Harvesting")
    print("=" * 65)
    print("\n[1/3] Initializing BigCodeBench sandbox suite...")
    benchmark = BigCodeBenchmark(
        subset=args.subset,
        split=args.split,
        num_tasks=args.num_tasks,
        timeout=args.timeout,
        dataset_config=args.dataset_config,
        dataset_revision=args.dataset_revision,
        release=args.release,
    )
    total_tasks = len(benchmark.list_tasks(split=args.source_split))
    total_episodes = total_tasks * args.n_rollouts

    print(f"Model:              {args.model}")
    print(f"Subset:             {args.subset}")
    print(f"Release:            {args.release}")
    print(f"Split:              {args.split}")
    print(f"Tasks:              {total_tasks} (Rollouts per task: {args.n_rollouts})")
    print(f"Total Episodes:     {total_episodes}")
    print(f"Max Turns / Ep:     {args.max_steps}")
    print(f"Execution Timeout:  {args.timeout}s")
    print(f"Temperature:        {args.temperature}")
    print(f"Target Layers:      {args.layers}")
    print(f"Anchor Positions:   {args.anchors}")
    print(f"Output Directory:   {output_path.resolve()}")
    print("=" * 65)

    # 2. Initialize Activation Engine
    print("\n[2/3] Loading model engine...")
    extractor = ActivationExtractor(model=args.model, revision=args.model_revision)

    # 3. Execute Runner
    print("\n[3/3] Starting code execution & activation harvesting...")
    runner = Runner(
        extractor=extractor,
        benchmark=benchmark,
        output_dir=output_path,
        layer_ids=args.layers,
        anchors=args.anchors,
        split_manifest=args.split_manifest,
    )

    results = runner.run_benchmark(
        split=args.source_split,
        max_tasks=args.num_tasks,
        max_steps=args.max_steps,
        n_rollouts=args.n_rollouts,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
    )

    completed_episodes = len(results)
    successes = sum(1 for r in results if r["is_success"])
    rate = (successes / completed_episodes) * 100 if completed_episodes > 0 else 0.0

    print("\n" + "=" * 65)
    print(f"BigCodeBench Complete! Success Rate: {rate:.1f}% ({successes}/{completed_episodes})")
    print(f"Saved artifacts to: {output_path.resolve()}")
    print("=" * 65)


if __name__ == "__main__":
    main()
