"""
TextCraft Benchmark Data Collection Script.
Runs open-weight LLMs on Minecraft sequential recipe crafting tasks,
extracts residual-stream activations at decision tokens, and persists
trajectories to Parquet and Safetensors.
"""

import argparse
from pathlib import Path
import sys

# Ensure repository root is in Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.activations import ActivationExtractor
from src.benchmark import TextCraftBenchmark
from src.runner import Runner


def parse_args():
    parser = argparse.ArgumentParser(description="TextCraft Agent Data Collection")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-Coder-7B-Instruct",
        help="Hugging Face model ID (e.g. Qwen/Qwen2.5-Coder-7B-Instruct, meta-llama/Llama-3.1-8B-Instruct)",
    )
    parser.add_argument(
        "--num_tasks",
        type=int,
        default=50,
        help="Number of TextCraft tasks to generate and run (default: 50)",
    )
    parser.add_argument(
        "--n_rollouts",
        type=int,
        default=1,
        help="Number of rollouts per task (default: 1; use >= 3 for multi-rollout probing)",
    )
    parser.add_argument(
        "--min_depth",
        type=int,
        default=2,
        help="Minimum recipe depth for goal selection (default: 2)",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=12,
        help="Maximum crafting steps per episode before terminating",
    )
    parser.add_argument(
        "--save_mode",
        type=str,
        choices=["decision", "all"],
        default="decision",
        help="Activation storage mode: 'decision' (default, ~200KB/step) or 'all' (~200MB/step)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/runs/textcraft_qwen7b",
        help="Directory to persist safetensors and parquet summaries",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Generation temperature (0.0 = greedy, >0.0 for sampling across rollouts)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_path = Path(args.output_dir)

    print("=" * 60)
    print("TextCraft Activation Collection")
    print("=" * 60)
    print(f"Model:            {args.model}")
    print(f"Save Mode:        {args.save_mode} ({'~200 KB/step' if args.save_mode == 'decision' else '~200 MB/step'})")
    print(f"Num Tasks:        {args.num_tasks}")
    print(f"Rollouts / Task:  {args.n_rollouts}")
    print(f"Min Recipe Depth: {args.min_depth}")
    print(f"Max Steps/Ep:     {args.max_steps}")
    print(f"Output Directory: {output_path.resolve()}")
    print("=" * 60)

    # 1. Initialize Activation Extractor (Loads Model to GPU)
    print("\n[1/3] Loading model engine...")
    extractor = ActivationExtractor(model=args.model)

    # 2. Initialize TextCraft Benchmark
    print("\n[2/3] Initializing TextCraft crafting environment...")
    benchmark = TextCraftBenchmark(
        num_tasks=args.num_tasks,
        min_depth=args.min_depth,
    )
    total_tasks = len(benchmark.list_tasks())
    print(f"  -> Initialized {total_tasks} total TextCraft benchmark tasks.")

    # 3. Execute Runner
    print("\n[3/3] Starting episode rollouts and activation harvesting...")
    runner = Runner(
        extractor=extractor,
        benchmark=benchmark,
        output_dir=output_path,
        save_mode=args.save_mode,
    )

    results = runner.run_benchmark(
        max_tasks=args.num_tasks,
        max_steps=args.max_steps,
        n_rollouts=args.n_rollouts,
        temperature=args.temperature,
    )

    print("\n" + "=" * 60)
    print(f"Collection Complete! Saved to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
