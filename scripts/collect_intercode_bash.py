"""
InterCode-Bash Benchmark Data Collection Script:
Runs open-weight LLMs (e.g. Qwen2.5-Coder-7B-Instruct) on interactive Linux command-line tasks,
extracts residual-stream activations at specified anchors (post_gen, pre_gen), and
persists shard arrays and trajectory summaries.
"""

import argparse
from pathlib import Path
import sys

# Ensure repository root is in Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.activations import ActivationExtractor
from src.benchmark.intercode.bash import InterCodeBashBenchmark
from src.runner import Runner


def parse_args():
    parser = argparse.ArgumentParser(description="InterCode-Bash Activation Harvesting")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-Coder-7B-Instruct",
        help="Hugging Face model ID (default: Qwen/Qwen2.5-Coder-7B-Instruct)",
    )
    parser.add_argument(
        "--num_tasks",
        type=int,
        default=50,
        help="Number of InterCode-Bash tasks to run (default: 50 tasks)",
    )
    parser.add_argument(
        "--n_rollouts",
        type=int,
        default=5,
        help="Number of rollouts per task (default: 5 rollouts -> 250 episodes)",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=8,
        help="Maximum bash command turns per episode (default: 8)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7 for stochastic rollouts)",
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=[20],
        help="Transformer layer indices to extract (default: 20)",
    )
    parser.add_argument(
        "--anchors",
        type=str,
        nargs="+",
        default=["post_gen", "pre_gen"],
        help="Anchor tokens to extract (default: post_gen pre_gen)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="artifacts/intercode_bash/qwen2.5-7b/qwen2.5-7b",
        help="Directory to persist features and summaries",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_path = Path(args.output_dir)

    print("=" * 65)
    print("InterCode-Bash Activation Harvesting")
    print("=" * 65)
    print(f"Model:              {args.model}")
    print(f"Tasks:              {args.num_tasks} (Rollouts per task: {args.n_rollouts})")
    print(f"Total Episodes:     {args.num_tasks * args.n_rollouts}")
    print(f"Max Turns / Ep:     {args.max_steps}")
    print(f"Temperature:        {args.temperature}")
    print(f"Target Layers:      {args.layers}")
    print(f"Anchor Positions:   {args.anchors}")
    print(f"Output Directory:   {output_path.resolve()}")
    print("=" * 65)

    # 1. Initialize Activation Engine
    print("\n[1/3] Loading model engine...")
    extractor = ActivationExtractor(model=args.model)

    # 2. Initialize InterCode-Bash Benchmark
    print("\n[2/3] Initializing InterCode-Bash sandbox suite...")
    benchmark = InterCodeBashBenchmark()
    print(f"  -> Initialized {len(benchmark.list_tasks())} total bash benchmark tasks.")

    # 3. Execute Runner
    print("\n[3/3] Starting interactive execution & activation harvesting...")
    runner = Runner(
        extractor=extractor,
        benchmark=benchmark,
        output_dir=output_path,
        layer_ids=args.layers,
        anchors=args.anchors,
    )

    results = runner.run_benchmark(
        max_tasks=args.num_tasks,
        max_steps=args.max_steps,
        n_rollouts=args.n_rollouts,
        temperature=args.temperature,
    )

    total_episodes = len(results)
    successes = sum(1 for r in results if r["is_success"])
    rate = (successes / total_episodes) * 100 if total_episodes > 0 else 0.0

    print("\n" + "=" * 65)
    print(f"InterCode-Bash Complete! Success Rate: {rate:.1f}% ({successes}/{total_episodes})")
    print(f"Saved artifacts to: {output_path.resolve()}")
    print("=" * 65)


if __name__ == "__main__":
    main()
