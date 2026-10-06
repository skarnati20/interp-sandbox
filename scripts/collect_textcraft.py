"""
TextCraft Benchmark Data Collection Script.
Runs open-weight LLMs on Minecraft sequential recipe crafting tasks,
extracts dual-anchor residual-stream activations, and persists
sharded features ('feat.shard*.npz' and 'meta.shard*.jsonl') for probing.
"""

import argparse
from pathlib import Path
import sys

# Ensure repository root is in Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.activations import ActivationExtractor
from src.benchmark.textcraft import TextCraftBenchmark
from src.runner import Runner


def parse_args():
    parser = argparse.ArgumentParser(description="TextCraft Agent Data Collection & Sharded Feature Extraction")
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
        "--layers",
        type=int,
        nargs="+",
        default=None,
        help="Specific layer indices to extract (default: None = all layers; e.g. --layers 20)",
    )
    parser.add_argument(
        "--anchors",
        type=str,
        nargs="+",
        default=["post_gen", "pre_gen"],
        choices=["post_gen", "pre_gen", "first_gen", "mean_gen", "mean_prompt", "thought_end"],
        help="Anchors to extract (default: post_gen pre_gen)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/runs/textcraft_qwen7b",
        help="Directory to persist feature shards and parquet summaries",
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
    print("TextCraft Activation Collection (Sharded Storage)")
    print("=" * 60)
    print(f"Model:            {args.model}")
    print(f"Num Tasks:        {args.num_tasks}")
    print(f"Rollouts / Task:  {args.n_rollouts}")
    print(f"Min Recipe Depth: {args.min_depth}")
    print(f"Max Steps/Ep:     {args.max_steps}")
    print(f"Target Layers:    {args.layers or 'All Layers'}")
    print(f"Anchors:          {args.anchors}")
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
        layer_ids=args.layers,
        anchors=args.anchors,
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
