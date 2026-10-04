"""
MiniWoB++ Benchmark Data Collection Script.
Runs open-weight LLMs on MiniWoB++ web environments, extracts residual-stream activations,
and persists trajectories to Parquet and Safetensors.
"""

import argparse
from pathlib import Path
import sys

# Ensure repository root is in Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.activations import ActivationExtractor
from src.benchmark import MiniWoBBenchmark
from src.runner import Runner


def parse_args():
    parser = argparse.ArgumentParser(description="MiniWoB++ Agent Data Collection")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-Coder-7B-Instruct",
        help="Hugging Face model ID (e.g. Qwen/Qwen2.5-Coder-7B-Instruct, meta-llama/Llama-3.1-8B-Instruct)",
    )
    parser.add_argument(
        "--all_envs",
        action="store_true",
        help="Run across ALL 128 registered MiniWoB++ environments",
    )
    parser.add_argument(
        "--dom_only",
        action="store_true",
        help="Run across curated ~45 text/DOM environments (excluding canvas/pixel drag tasks)",
    )
    parser.add_argument(
        "--envs",
        type=str,
        nargs="+",
        default=[
            "miniwob/click-test-2-v1",
            "miniwob/login-user-v1",
            "miniwob/enter-text-v1",
            "miniwob/choose-date-v1",
            "miniwob/click-dialog-v1",
        ],
        help="MiniWoB environment names, 'all', or 'dom_only'",
    )
    parser.add_argument(
        "--save_mode",
        type=str,
        choices=["decision", "all"],
        default="decision",
        help="Activation storage mode: 'decision' (default, ~200KB/step) or 'all' (~200MB/step)",
    )
    parser.add_argument(
        "--episodes_per_env",
        type=int,
        default=1,
        help="Number of episodes to run per environment",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=None,
        help="Total episode limit across all environments (optional)",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=8,
        help="Maximum steps per episode before terminating",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/runs/miniwob_qwen7b",
        help="Directory to persist safetensors and parquet summaries",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Generation temperature (0.0 = greedy)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Determine environment list
    if args.all_envs or args.envs == ["all"] or args.envs == "all":
        env_selection = "all"
        display_envs = f"ALL 128 MiniWoB++ environments (episodes_per_env={args.episodes_per_env})"
    elif args.dom_only or args.envs == ["dom_only"] or args.envs == "dom_only":
        env_selection = "dom_only"
        display_envs = f"Curated ~45 DOM-only environments (episodes_per_env={args.episodes_per_env})"
    else:
        env_selection = args.envs
        display_envs = str(args.envs)

    print("=" * 60)
    print("MiniWoB++ Activation Collection")
    print("=" * 60)
    print(f"Model:            {args.model}")
    print(f"Environments:     {display_envs}")
    print(f"Save Mode:        {args.save_mode} ({'~200 KB/step' if args.save_mode == 'decision' else '~200 MB/step'})")
    print(f"Episodes/Env:     {args.episodes_per_env}")
    print(f"Max Steps/Ep:     {args.max_steps}")
    print(f"Output Directory: {args.output_dir}")
    print("=" * 60)

    # 1. Initialize Activation Extractor (Loads Model to GPU)
    print("\n[1/3] Loading model engine...")
    extractor = ActivationExtractor(model=args.model)

    # 2. Initialize MiniWoB Benchmark
    print("\n[2/3] Initializing MiniWoB environments...")
    benchmark = MiniWoBBenchmark(
        env_names=env_selection,
        episodes_per_env=args.episodes_per_env,
    )
    total_tasks = len(benchmark.list_tasks())
    print(f"  -> Generated {total_tasks} total benchmark tasks.")

    # 3. Execute Runner
    print("\n[3/3] Starting episode rollouts and activation harvesting...")
    output_path = Path(args.output_dir)
    runner = Runner(
        extractor=extractor,
        benchmark=benchmark,
        output_dir=output_path,
        save_mode=args.save_mode,
    )

    results = runner.run_benchmark(max_tasks=args.num_episodes, max_steps=args.max_steps)

    print("\n" + "=" * 60)
    print(f"Collection Complete! Saved to: {output_path.resolve()}")


if __name__ == "__main__":
    main()
