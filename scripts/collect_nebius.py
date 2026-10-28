"""
Nebius SWE-agent Trajectory Replay & Activation Extraction Script.
Replays pre-recorded agent trajectories (from nebius/SWE-agent-trajectories or local files)
through an open-weight LLM, extracts residual-stream activations at specified anchors
(post_gen, pre_gen, thought_end), and persists sharded features ('feat.shard*.npz' and
'meta.shard*.jsonl') for interpretability and probing experiments.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Ensure repository root is in Python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.activations import ActivationExtractor
from src.replay.replay_runner import ReplayRunner
from src.traces.nebius import NebiusSWEAgentTraces


def parse_args():
    parser = argparse.ArgumentParser(
        description="Nebius SWE-agent Trajectory Replay & Activation Extraction"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-Coder-7B-Instruct",
        help="Hugging Face model ID (e.g. Qwen/Qwen2.5-Coder-7B-Instruct, meta-llama/Llama-3.1-8B-Instruct)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="nebius/SWE-agent-trajectories",
        help="Hugging Face dataset ID or local path containing parquet files (default: nebius/SWE-agent-trajectories)",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="/tmp/swe_agent_trajectories",
        help="Local directory to cache downloaded parquet files (default: /tmp/swe_agent_trajectories)",
    )
    parser.add_argument(
        "--max_trajectories",
        type=int,
        default=50,
        help="Maximum number of trajectories to replay (default: 50; use 0 or None for all)",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Maximum turns per trajectory before truncating (default: None = replay all turns)",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=16384,
        help="Maximum sequence length before truncating older conversation turns from the left (default: 16384)",
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=None,
        help="Transformer layer indices to extract (default: None = all layers; e.g. --layers 20)",
    )
    parser.add_argument(
        "--anchors",
        type=str,
        nargs="+",
        default=["post_gen", "pre_gen"],
        choices=["post_gen", "pre_gen", "first_gen", "mean_gen", "mean_prompt", "mean_all", "thought_end"],
        help="Anchor tokens to extract (default: post_gen pre_gen)",
    )
    parser.add_argument(
        "--shard_size",
        type=int,
        default=2000,
        help="Maximum number of step samples per shard file (default: 2000)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/runs/nebius_swe_replay",
        help="Directory to persist feature shards and run summary (default: data/runs/nebius_swe_replay)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_path = Path(args.output_dir)
    max_trajectories = None if (args.max_trajectories is None or args.max_trajectories <= 0) else args.max_trajectories

    print("=" * 65)
    print("Nebius SWE-agent Trajectory Replay & Feature Harvesting")
    print("=" * 65)
    print(f"Model:               {args.model}")
    print(f"Dataset / Path:      {args.dataset}")
    print(f"Cache Directory:     {args.cache_dir}")
    print(f"Max Trajectories:    {max_trajectories or 'All'}")
    print(f"Max Steps / Traj:    {args.max_steps or 'All'}")
    print(f"Max Seq Length:      {args.max_seq_len or 'Unlimited'}")
    print(f"Target Layers:       {args.layers or 'All Layers'}")
    print(f"Anchors:             {args.anchors}")
    print(f"Shard Size:          {args.shard_size}")
    print(f"Output Directory:    {output_path.resolve()}")
    print("=" * 65)

    # 1. Initialize Trace Adapter
    print("\n[1/3] Initializing Nebius SWE-agent trace adapter...")
    trace_adapter = NebiusSWEAgentTraces(
        dataset_name_or_path=args.dataset,
        cache_dir=args.cache_dir,
    )

    # 2. Initialize Activation Extractor (Loads Model onto Device)
    print("\n[2/3] Loading model engine into memory...")
    extractor = ActivationExtractor(
        model=args.model,
        max_seq_len=args.max_seq_len,
    )

    # 3. Initialize and Run ReplayRunner
    print("\n[3/3] Starting trace replay and activation harvesting...")
    runner = ReplayRunner(
        extractor=extractor,
        trace_adapter=trace_adapter,
        output_dir=output_path,
        layer_ids=args.layers,
        anchors=args.anchors,
        shard_size=args.shard_size,
        verbose=True,
    )

    results = runner.run_replay(
        max_trajectories=max_trajectories,
        max_steps=args.max_steps,
    )

    total_episodes = len(results)
    successes = sum(1 for r in results if r["is_success"])
    rate = (successes / total_episodes) * 100 if total_episodes > 0 else 0.0

    print("\n" + "=" * 65)
    print(f"Replay Complete! Success Rate: {rate:.1f}% ({successes}/{total_episodes})")
    print(f"Saved feature shards to {output_path / 'features'}")
    print(f"Saved run summary to {output_path / 'run_summary.parquet'}")
    print("=" * 65)


if __name__ == "__main__":
    main()
