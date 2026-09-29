"""
MiniWoB++ Benchmark Data Collection Script.
Runs open-weight LLMs on MiniWoB++ web environments, extracts residual-stream activations,
and persists trajectories to Parquet and Safetensors.
"""

import argparse
from pathlib import Path

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
        help="MiniWoB environment names to evaluate",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=20,
        help="Number of episodes / tasks to run",
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

    print("=" * 60)
    print("MiniWoB++ Activation Collection")
    print("=" * 60)
    print(f"Model:       {args.model}")
    print(f"Envs:        {args.envs}")
    print(f"Episodes:    {args.num_episodes}")
    print(f"Max Steps:   {args.max_steps}")
    print(f"Output Dir:  {args.output_dir}")
    print("=" * 60)

    # 1. Initialize Activation Extractor (Loads Model to GPU)
    print("\n[1/3] Loading model engine...")
    extractor = ActivationExtractor(model=args.model)

    # 2. Initialize MiniWoB Benchmark
    print("\n[2/3] Initializing MiniWoB environments...")
    benchmark = MiniWoBBenchmark(env_names=args.envs)

    # 3. Execute Runner
    print("\n[3/3] Starting episode rollouts and activation harvesting...")
    output_path = Path(args.output_dir)
    runner = Runner(extractor=extractor, benchmark=benchmark, output_dir=output_path)

    results = runner.run_benchmark(max_tasks=args.num_episodes, max_steps=args.max_steps)

    print("\n" + "=" * 60)
    print(f"Collection Complete! Saved to: {output_path.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
