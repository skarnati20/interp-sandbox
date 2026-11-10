#!/usr/bin/env python3
"""
5-Fold Cross-Validation Steering on TextCraft:
1. Downloads full TextCraft activations from Hugging Face (skarnati20/textcraft-full-qwen7b-activations).
2. Performs stratified group 5-fold cross-validation by task_id to ensure zero data leakage.
3. In each fold:
   - Trains probe and CentroidSteerer on 80% training activations at target anchor (pre_gen or post_gen).
   - Runs multi-turn rollouts on the remaining 20% test tasks, evaluating & steering turns at the chosen anchor token.
4. Collects and saves consolidated steering telemetry and success rates across all folds.
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
from sklearn.model_selection import StratifiedGroupKFold

from src.activations import ActivationExtractor
from src.benchmark.textcraft import TextCraftBenchmark
from src.hf import download_activations
from src.probe.probes.linear_probe import LinearProbe
from src.probe.split import rebalance_dataset
from src.steering.steerers.centroid import CentroidSteerer
from src.steering_runner import SteeringRunner
from src.store import ShardReader, save_run_summary


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run 5-Fold Cross-Validation Steering on TextCraft Activations"
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
        help="Transformer layer index for steering and activation extraction (default: 20)",
    )
    parser.add_argument(
        "--anchor",
        type=str,
        default="post_gen",
        choices=["post_gen", "pre_gen"],
        help="Anchor token position to evaluate & steer ('post_gen' or 'pre_gen'; default: post_gen)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Failure detection probability threshold for steering intervention (default: 0.5)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Centroid steering direction scaling multiplier (default: 1.0)",
    )
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize the centroid steering vector to unit norm",
    )
    parser.add_argument(
        "--c_reg",
        type=float,
        default=0.1,
        help="L2 regularization constant C for LinearProbe (default: 0.1)",
    )
    parser.add_argument(
        "--max_train_turns",
        "--train_max_turns",
        type=int,
        default=None,
        dest="max_train_turns",
        help="Train probe and steerer only on activations from the first N turns (default: None = all turns)",
    )
    parser.add_argument(
        "--max_steer_turns",
        "--steer_max_turns",
        "--eval_max_turns",
        type=int,
        default=None,
        dest="max_steer_turns",
        help="Only apply steering hooks during the first N turns of each task; subsequent turns run unsteered (default: None = steer all turns)",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=35,
        help="Fallback turn cap per episode if dynamic_steps is disabled (default: 35)",
    )
    parser.add_argument(
        "--no_dynamic_steps",
        action="store_false",
        dest="dynamic_steps",
        default=True,
        help="Disable adaptive recipe-DAG step budgeting and use fixed max_steps",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature during generation (default: 0.0 for greedy)",
    )
    parser.add_argument(
        "--max_eval_tasks_per_fold",
        type=int,
        default=None,
        help="Cap evaluation tasks per fold (useful for quick validation; default: None = all 20%% tasks)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/runs/textcraft_steered_cv",
        help="Output directory to save shards and summary (default: data/runs/textcraft_steered_cv)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for fold splitting and dataset rebalancing (default: 42)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce verbose output during generation",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 65)
    print("5-Fold Cross-Validation Steering on TextCraft")
    print("=" * 65)
    print(f"Dataset Repo:       {args.repo_id}")
    print(f"Model:              {args.model}")
    print(f"Layer ID:           {args.layer_id}")
    print(f"Anchor:             {args.anchor}")
    print(f"Threshold:          {args.threshold}")
    print(f"Alpha:              {args.alpha} (Normalize: {args.normalize})")
    print(f"Max Train Turns:    {args.max_train_turns or 'All'}")
    print(f"Max Steer Turns:    {args.max_steer_turns or 'All turns in episode'}")
    print(f"Dynamic Steps:      {args.dynamic_steps} (Fallback max: {args.max_steps})")
    print(f"Output Directory:   {output_dir.resolve()}")
    print("=" * 65)

    # 1. Download / Load TextCraft Activations from Hugging Face
    print("\n[1/4] Fetching TextCraft activations from Hugging Face...")
    data_path = download_activations(
        repo_id=args.repo_id,
        token=args.token,
    )
    reader = ShardReader(data_path)
    dataset = reader.read_all(anchor=args.anchor, layer=args.layer_id)
    print(f"  -> Loaded {dataset.num_samples} activation samples.")
    print(f"  -> Unique task IDs: {dataset.metadata['task_id'].nunique()}")
    print(f"  -> Success label distribution: 0={np.sum(dataset.y == 0)}, 1={np.sum(dataset.y == 1)}")

    # 2. Initialize TextCraft benchmark
    print("\n[2/4] Loading TextCraft benchmark tasks...")
    bench = TextCraftBenchmark(num_tasks=544)
    all_tasks = bench.list_tasks()
    task_map = {t.task_id: t for t in all_tasks}
    print(f"  -> Initialized {len(all_tasks)} TextCraft task specifications.")

    # 3. Initialize Model and Activation Extractor
    print("\n[3/4] Initializing model & ActivationExtractor...")
    extractor = ActivationExtractor(model=args.model)

    # 4. 5-Fold Stratified Group Cross-Validation
    print("\n[4/4] Executing 5-Fold Cross-Validation Multi-Turn Steering...")
    groups = dataset.metadata["task_id"].values
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=args.seed)
    splits = list(sgkf.split(dataset.X, dataset.y, groups))

    all_steering_results = []
    fold_metrics = []

    for fold_idx, (train_idx, val_idx) in enumerate(splits):
        fold_num = fold_idx + 1
        print(f"\n{'=' * 30} Fold {fold_num} / 5 {'=' * 30}")

        train_set = dataset.slice(train_idx)
        val_set = dataset.slice(val_idx)

        # Optionally filter training activations to first N turns
        train_data = train_set
        if args.max_train_turns is not None:
            step_col = "step_idx" if "step_idx" in train_set.metadata.columns else "round"
            turn_mask = (train_set.metadata[step_col] < args.max_train_turns).values
            train_data = train_set.slice(np.where(turn_mask)[0])
            print(f"  Filtered train set to first {args.max_train_turns} turns: {train_data.num_samples} samples")

        # 50/50 balance training set for robust probe and centroid vector
        train_bal = rebalance_dataset(train_data, ratio=0.5, seed=args.seed + fold_idx)
        print(f"  Train samples (rebalanced): {train_bal.num_samples} (pos={np.sum(train_bal.y == 1)}, neg={np.sum(train_bal.y == 0)})")

        # Resolve validation tasks for this fold (20% test tasks)
        val_task_ids = list(val_set.metadata["task_id"].unique())
        val_tasks = [task_map[tid] for tid in val_task_ids if tid in task_map]
        print(f"  Validation tasks: {len(val_tasks)} unique tasks")

        # Initialize fold-specific components
        probe = LinearProbe(c_reg=args.c_reg)
        steerer = CentroidSteerer(alpha=args.alpha, normalize=args.normalize)
        fold_out_dir = output_dir / f"fold_{fold_num}"

        runner = SteeringRunner(
            extractor=extractor,
            probe=probe,
            steerer=steerer,
            benchmark=bench,
            output_dir=fold_out_dir,
            layer_id=args.layer_id,
            anchor=args.anchor,
            threshold=args.threshold,
            verbose=not args.quiet,
        )

        # Train on 80% fold activations
        print(f"  Fitting LinearProbe and CentroidSteerer on Fold {fold_num} training data...")
        runner.train_intervention(train_bal)

        # Evaluate probe classification performance on validation fold activations as sanity check
        probe_eval = probe.evaluate(val_set)
        print(f"  Fold {fold_num} Probe Sanity Check: ROC-AUC = {probe_eval.get('roc_auc', 0.0):.4f}, Acc = {probe_eval.get('accuracy', 0.0):.4f}")

        # Run multi-turn steering on the 20% validation tasks until complete or dynamic budget
        print(f"  Running multi-turn steered rollouts on Fold {fold_num} test tasks...")
        fold_results = runner.run_benchmark(
            tasks=val_tasks,
            max_tasks=args.max_eval_tasks_per_fold,
            max_steps=args.max_steps,
            dynamic_steps=args.dynamic_steps,
            max_steer_turns=args.max_steer_turns,
            temperature=args.temperature,
        )

        # Attach fold metadata to each result
        for r in fold_results:
            r["fold"] = fold_num
        all_steering_results.extend(fold_results)

        # Fold summary stats
        fold_turns = sum(r["num_steps"] for r in fold_results)
        fold_steered = sum(r["steered_steps"] for r in fold_results)
        pct_steered = (fold_steered / max(1, fold_turns)) * 100
        fold_successes = sum(1 for r in fold_results if r.get("is_success", False))
        fold_success_rate = (fold_successes / max(1, len(fold_results))) * 100

        fold_stat = {
            "fold": fold_num,
            "val_tasks": len(fold_results),
            "successes": fold_successes,
            "success_rate": fold_success_rate,
            "probe_roc_auc": probe_eval.get("roc_auc", 0.0),
            "probe_accuracy": probe_eval.get("accuracy", 0.0),
            "total_turns": fold_turns,
            "steered_turns": fold_steered,
            "pct_turns_steered": pct_steered,
        }
        fold_metrics.append(fold_stat)
        print(
            f"  Fold {fold_num} Complete: Success Rate: {fold_success_rate:.1f}% ({fold_successes}/{len(fold_results)}) | "
            f"Turns: {fold_turns} ({fold_steered} steered, {pct_steered:.1f}%)"
        )

    # 5. Save consolidated steering results across all 5 folds
    print("\n" + "=" * 65)
    print("Collecting and Saving Consolidated 5-Fold Steering Results")
    print("=" * 65)

    summary_parquet_path = output_dir / "all_folds_steering_results.parquet"
    save_run_summary(summary_parquet_path, all_steering_results)

    metrics_json_path = output_dir / "kfold_steering_summary.json"
    with open(metrics_json_path, "w") as fp:
        json.dump(fold_metrics, fp, indent=2)

    total_all_turns = sum(r["num_steps"] for r in all_steering_results)
    total_all_steered = sum(r["steered_steps"] for r in all_steering_results)
    overall_pct_steered = (total_all_steered / max(1, total_all_turns)) * 100
    total_successes = sum(1 for r in all_steering_results if r.get("is_success", False))
    overall_success_rate = (total_successes / max(1, len(all_steering_results))) * 100

    print(f"Total Tasks Evaluated:  {len(all_steering_results)}")
    print(f"Overall Success Rate:   {overall_success_rate:.1f}% ({total_successes}/{len(all_steering_results)})")
    print(f"Total Turns Run:        {total_all_turns}")
    print(f"Total Turns Steered:    {total_all_steered} ({overall_pct_steered:.1f}%)")
    print(f"Saved Full Telemetry:   {summary_parquet_path.resolve()}")
    print(f"Saved Fold Metrics:     {metrics_json_path.resolve()}")


if __name__ == "__main__":
    main()
