"""
BigCodeBench Benchmark Data Collection Script:
Runs open-weight LLMs (e.g. Qwen2.5-Coder-7B-Instruct) on BigCodeBench coding tasks,
extracts residual-stream activations at specified anchors (post_gen, pre_gen), and
persists shard arrays and trajectory summaries for interpretability research.
"""

import argparse
from pathlib import Path
import sys

# Ensure repository root is in Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.activations import ActivationExtractor
from src.benchmark.bigcodebench import BigCodeBenchmark
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
        default="hard",
        help="BigCodeBench benchmark subset (default: hard)",
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
        default=50,
        help="Number of BigCodeBench tasks to run (default: 50 tasks)",
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
        "--sample_only",
        action="store_true",
        help="Use bundled sample tasks (offline quick verification)",
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
        default="artifacts/bigcodebench/qwen2.5-7b/qwen2.5-7b",
        help="Directory to persist features and summaries",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_path = Path(args.output_dir)

    print("=" * 65)
    print("BigCodeBench Activation Harvesting")
    print("=" * 65)
    print(f"Model:              {args.model}")
    print(f"Subset:             {args.subset}")
    print(f"Split:              {args.split}")
    print(f"Tasks:              {args.num_tasks} (Rollouts per task: {args.n_rollouts})")
    print(f"Total Episodes:     {args.num_tasks * args.n_rollouts}")
    print(f"Max Turns / Ep:     {args.max_steps}")
    print(f"Execution Timeout:  {args.timeout}s")
    print(f"Temperature:        {args.temperature}")
    print(f"Target Layers:      {args.layers}")
    print(f"Anchor Positions:   {args.anchors}")
    print(f"Output Directory:   {output_path.resolve()}")
    print("=" * 65)

    # 1. Initialize Activation Engine
    print("\n[1/3] Loading model engine...")
    extractor = ActivationExtractor(model=args.model)

    # 2. Initialize BigCodeBench Benchmark
    print("\n[2/3] Initializing BigCodeBench sandbox suite...")
    benchmark = BigCodeBenchmark(
        subset=args.subset,
        split=args.split,
        num_tasks=args.num_tasks,
        timeout=args.timeout,
        sample_only=args.sample_only,
        single_turn=False if args.max_steps > 1 else True,
    )
    print(f"  -> Initialized {len(benchmark.list_tasks())} total BigCodeBench tasks.")

    # 3. Execute Runner
    print("\n[3/3] Starting code execution & activation harvesting...")
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
    print(f"BigCodeBench Complete! Success Rate: {rate:.1f}% ({successes}/{total_episodes})")
    print(f"Saved artifacts to: {output_path.resolve()}")
    print("=" * 65)


if __name__ == "__main__":
    main()
