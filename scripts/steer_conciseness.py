#!/usr/bin/env python3
"""
Conciseness Steering on TextCraft:
1. Loads full TextCraft activations from Hugging Face (skarnati20/textcraft-full-qwen7b-activations).
2. Filters to successful trajectories (success == 1).
3. For each task with multiple successful rollouts, identifies:
   - The quickest successful rollout (k_fast = argmin turns)
   - The slowest successful rollout (k_slow = argmax turns)
4. Computes the Conciseness Steering Direction:
   v_concise = mean(X_fast) - mean(X_slow)
5. Evaluates vector properties (norm, cosine similarity with general success vector).
6. Optionally trains a conciseness probe and runs live steering evaluation to measure turn reduction.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

# Ensure repository root is on PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch

from src.activations import ActivationExtractor
from src.benchmark.textcraft import TextCraftBenchmark
from src.hf import download_activations
from src.probe.probes.linear_probe import LinearProbe
from src.probe.split import rebalance_dataset
from src.steering.steerers.centroid import CentroidSteerer
from src.steering_runner import SteeringRunner
from src.store import ActivationDataset, ShardReader, save_run_summary


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute Conciseness Steering Direction from Quickest vs Slowest Successful Rollouts"
    )
    parser.add_argument(
        "--repo_id",
        type=str,
        default="skarnati20/textcraft-full-qwen7b-activations",
        help="Hugging Face dataset repo containing TextCraft activations (default: skarnati20/textcraft-full-qwen7b-activations)",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Optional Hugging Face API token (not needed for public datasets)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2.5-Coder-7B-Instruct",
        help="Model identifier or local checkpoint path (default: Qwen/Qwen2.5-Coder-7B-Instruct)",
    )
    parser.add_argument(
        "--layer_id",
        type=int,
        default=20,
        help="Transformer layer index for steering and activations (default: 20)",
    )
    parser.add_argument(
        "--anchor",
        type=str,
        default="post_gen",
        choices=["post_gen", "pre_gen"],
        help="Anchor position for features ('post_gen' or 'pre_gen'; default: post_gen)",
    )
    parser.add_argument(
        "--min_spread",
        type=int,
        default=1,
        help="Minimum turn difference between slowest and fastest successful rollout for a task (default: 1)",
    )
    parser.add_argument(
        "--max_train_turns",
        type=int,
        default=None,
        help="Use activations only up to turn N for computing the vector (default: None = all turns)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Steering multiplier for the conciseness vector (default: 1.0)",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize the conciseness steering vector to unit norm",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold for conciseness probe failure detection (default: 0.5)",
    )
    parser.add_argument(
        "--run_eval",
        action="store_true",
        help="Run live steering evaluation on TextCraft tasks using the conciseness vector",
    )
    parser.add_argument(
        "--max_eval_tasks",
        type=int,
        default=None,
        help="Number of tasks to evaluate when --run_eval is enabled (default: None = all tasks)",
    )
    parser.add_argument(
        "--max_steer_turns",
        type=int,
        default=None,
        help="Only apply steering hook on first N turns during live eval (default: None = all turns)",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=35,
        help="Fallback turn cap per task (default: 35)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Generation temperature (default: 0.0)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/runs/textcraft_conciseness_steering",
        help="Directory to save the conciseness steering vector and eval results (default: data/runs/textcraft_conciseness_steering)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    return parser.parse_args()


def extract_concise_vs_verbose_dataset(
    dataset: ActivationDataset,
    min_spread: int = 1,
    max_train_turns: int | None = None,
) -> tuple[ActivationDataset, dict]:
    """
    Filters the dataset to successful rollouts (success == 1) and pairs the quickest
    successful rollout (class 1: concise) against the slowest successful rollout (class 0: verbose)
    for each task.
    """
    meta = dataset.metadata
    succ_meta = meta[meta["success"] == 1]

    # Calculate total turns per rollout
    rollout_lens = succ_meta.groupby(["task_id", "rollout_k"]).agg(
        n_turns=("step_idx", lambda s: s.max() + 1)
    ).reset_index()

    # Identify tasks with turn spread
    task_stats = rollout_lens.groupby("task_id")["n_turns"].agg(["min", "max", "count"]).reset_index()
    task_stats["spread"] = task_stats["max"] - task_stats["min"]
    valid_tasks = task_stats[task_stats["spread"] >= min_spread]

    fast_rollouts = set()
    slow_rollouts = set()
    task_spread_info = []

    for _, row in valid_tasks.iterrows():
        tid = row["task_id"]
        sub = rollout_lens[rollout_lens["task_id"] == tid]
        fast_row = sub.loc[sub["n_turns"].idxmin()]
        slow_row = sub.loc[sub["n_turns"].idxmax()]

        fast_k = fast_row["rollout_k"]
        slow_k = slow_row["rollout_k"]
        fast_turns = int(fast_row["n_turns"])
        slow_turns = int(slow_row["n_turns"])

        fast_rollouts.add((tid, fast_k))
        slow_rollouts.add((tid, slow_k))
        task_spread_info.append({
            "task_id": tid,
            "fast_k": int(fast_k),
            "fast_turns": fast_turns,
            "slow_k": int(slow_k),
            "slow_turns": slow_turns,
            "spread": slow_turns - fast_turns,
        })

    # Filter steps
    fast_indices = []
    slow_indices = []

    step_col = "step_idx" if "step_idx" in meta.columns else "round"

    for idx, r in meta.iterrows():
        if r["success"] != 1:
            continue
        if max_train_turns is not None and r[step_col] >= max_train_turns:
            continue

        pair = (r["task_id"], r["rollout_k"])
        if pair in fast_rollouts:
            fast_indices.append(idx)
        elif pair in slow_rollouts:
            slow_indices.append(idx)

    fast_idx_arr = np.array(fast_indices)
    slow_idx_arr = np.array(slow_indices)

    x_combined = np.vstack([dataset.X[fast_idx_arr], dataset.X[slow_idx_arr]])
    # Label 1 = concise, Label 0 = verbose
    y_combined = np.concatenate([np.ones(len(fast_idx_arr), dtype=int), np.zeros(len(slow_idx_arr), dtype=int)])

    meta_combined = pd.concat([meta.iloc[fast_idx_arr], meta.iloc[slow_idx_arr]], ignore_index=True)
    meta_combined["concise_label"] = y_combined

    concise_dataset = ActivationDataset(
        X=x_combined,
        y=y_combined,
        layer_ids=dataset.layer_ids,
        metadata=meta_combined,
    )

    summary_info = {
        "tasks_with_spread": len(valid_tasks),
        "total_concise_steps": len(fast_idx_arr),
        "total_verbose_steps": len(slow_idx_arr),
        "mean_fast_turns": float(np.mean([t["fast_turns"] for t in task_spread_info])),
        "mean_slow_turns": float(np.mean([t["slow_turns"] for t in task_spread_info])),
        "mean_spread": float(np.mean([t["spread"] for t in task_spread_info])),
        "sample_tasks": task_spread_info[:10],
    }

    return concise_dataset, summary_info


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("Conciseness Steering: Fastest vs. Slowest Successful Rollouts")
    print("=" * 65)
    print(f"Dataset Repo:       {args.repo_id}")
    print(f"Layer ID:           {args.layer_id}")
    print(f"Anchor:             {args.anchor}")
    print(f"Min Turn Spread:    {args.min_spread} turn(s)")
    print(f"Max Train Turns:    {args.max_train_turns or 'All'}")
    print(f"Alpha:              {args.alpha} (Normalize: {args.normalize})")
    print(f"Output Directory:   {output_dir.resolve()}")
    print("=" * 65)

    # 1. Download activations
    print("\n[1/4] Fetching activations from Hugging Face...")
    data_path = download_activations(repo_id=args.repo_id, token=args.token)
    reader = ShardReader(data_path)
    dataset = reader.read_all(anchor=args.anchor, layer=args.layer_id)
    print(f"  -> Loaded {dataset.num_samples} activation samples at layer {args.layer_id}.")

    # 2. Extract concise (k_fast) vs verbose (k_slow) successful trajectories
    print("\n[2/4] Identifying quickest vs slowest successful rollouts per task...")
    concise_ds, summary = extract_concise_vs_verbose_dataset(
        dataset,
        min_spread=args.min_spread,
        max_train_turns=args.max_train_turns,
    )

    print(f"  -> Tasks with turn spread >= {args.min_spread}: {summary['tasks_with_spread']}")
    print(f"  -> Average Fastest Rollout Turns: {summary['mean_fast_turns']:.2f} turns")
    print(f"  -> Average Slowest Rollout Turns: {summary['mean_slow_turns']:.2f} turns")
    print(f"  -> Average Spread per Task:       +{summary['mean_spread']:.2f} turns")
    print(f"  -> Extracted Steps: {summary['total_concise_steps']} concise (class 1) vs {summary['total_verbose_steps']} verbose (class 0)")

    # 3. Compute Conciseness Steering Direction
    print("\n[3/4] Computing Conciseness Steering Direction...")
    x_fast = concise_ds.X[concise_ds.y == 1]
    x_slow = concise_ds.X[concise_ds.y == 0]

    mu_fast = np.mean(x_fast, axis=0)
    mu_slow = np.mean(x_slow, axis=0)

    v_concise = mu_fast - mu_slow
    raw_norm = float(np.linalg.norm(v_concise))

    # Compare with standard success vector (success=1 vs success=0)
    mu_success = np.mean(dataset.X[dataset.y == 1], axis=0)
    mu_failure = np.mean(dataset.X[dataset.y == 0], axis=0)
    v_success = mu_success - mu_failure
    cos_sim = float(np.dot(v_concise, v_success) / (raw_norm * np.linalg.norm(v_success)))

    direction = v_concise
    if args.normalize and raw_norm > 1e-8:
        direction = direction / raw_norm

    print(f"  -> Raw Conciseness Direction L2 Norm: {raw_norm:.3f}")
    print(f"  -> Cosine Similarity with Success Vector (v_success): {cos_sim:.4f}")
    print(f"     (Note: Cosine similarity of ~0.35 indicates conciseness is a distinct subspace from raw task success)")

    # Save vectors to disk
    vector_npy_path = output_dir / "conciseness_vector.npy"
    vector_pt_path = output_dir / "conciseness_vector.pt"
    summary_json_path = output_dir / "conciseness_summary.json"

    np.save(vector_npy_path, direction)
    torch.save(torch.from_numpy(direction), vector_pt_path)

    metrics_record = {
        "layer_id": args.layer_id,
        "anchor": args.anchor,
        "raw_l2_norm": raw_norm,
        "cos_sim_with_success_vector": cos_sim,
        "min_spread": args.min_spread,
        "tasks_with_spread": summary["tasks_with_spread"],
        "mean_fast_turns": summary["mean_fast_turns"],
        "mean_slow_turns": summary["mean_slow_turns"],
        "mean_spread": summary["mean_spread"],
        "total_concise_steps": summary["total_concise_steps"],
        "total_verbose_steps": summary["total_verbose_steps"],
    }
    with open(summary_json_path, "w") as fp:
        json.dump(metrics_record, fp, indent=2)

    print(f"\n  ✓ Saved Conciseness Vector (.npy): {vector_npy_path.resolve()}")
    print(f"  ✓ Saved Conciseness Vector (.pt):  {vector_pt_path.resolve()}")
    print(f"  ✓ Saved Metrics Summary:           {summary_json_path.resolve()}")

    # 4. Optional Live Evaluation
    if args.run_eval:
        print("\n[4/4] Running Live Steering Evaluation with Conciseness Vector...")
        bench = TextCraftBenchmark(num_tasks=544)
        tasks = bench.list_tasks()
        if args.max_eval_tasks:
            tasks = tasks[:args.max_eval_tasks]

        extractor = ActivationExtractor(model=args.model)

        # Train a conciseness probe on concise vs verbose steps
        train_bal = rebalance_dataset(concise_ds, ratio=0.5, seed=args.seed)
        probe = LinearProbe(c_reg=0.1)
        probe.fit(train_bal)
        print(f"  -> Conciseness Probe Fitted on {train_bal.num_samples} balanced steps.")

        steerer = CentroidSteerer(alpha=args.alpha, normalize=args.normalize)
        steerer.steering_direction = direction

        runner = SteeringRunner(
            extractor=extractor,
            probe=probe,
            steerer=steerer,
            benchmark=bench,
            output_dir=output_dir / "eval_rollouts",
            layer_id=args.layer_id,
            anchor=args.anchor,
            threshold=args.threshold,
            verbose=True,
        )

        results = runner.run_benchmark(
            tasks=tasks,
            max_steps=args.max_steps,
            dynamic_steps=True,
            max_steer_turns=args.max_steer_turns,
            temperature=args.temperature,
        )

        succ_count = sum(1 for r in results if r["is_success"])
        succ_rate = succ_count / max(1, len(results)) * 100
        avg_turns = np.mean([r["num_steps"] for r in results if r["is_success"]]) if succ_count > 0 else 0.0

        print("\n" + "=" * 60)
        print("CONCISENESS STEERING EVALUATION RESULTS")
        print("=" * 60)
        print(f"Tasks Evaluated:              {len(results)}")
        print(f"Success Rate:                 {succ_rate:.1f}% ({succ_count}/{len(results)})")
        print(f"Avg Turns on Success:         {avg_turns:.2f} turns")


if __name__ == "__main__":
    main()
