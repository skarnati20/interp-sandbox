from __future__ import annotations

from pathlib import Path
from typing import Optional
import uuid

from src.activations import ActivationExtractor, ExtractionResult
from src.benchmark import BaseBenchmark, StepObservation, TaskInstance
from src.store import ShardWriter, save_run_summary


class Runner:
    """
    Executes benchmark rollouts across LLM agent tasks, records trajectory turns,
    and persists dual-anchor activation tensors with grouped task metadata directly
    into consolidated shards ('feat.shard*.npz' and 'meta.shard*.jsonl').
    """

    def __init__(
        self,
        extractor: ActivationExtractor,
        benchmark: BaseBenchmark,
        output_dir: str | Path = "data/runs",
        layer_ids: Optional[list[int]] = None,
        anchors: Optional[list[str]] = None,
        shard_size: int = 2000,
    ):
        self.extractor = extractor
        self.benchmark = benchmark
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.layer_ids = layer_ids
        self.anchors = anchors or ["post_gen", "pre_gen"]

        # Sharded storage writer for features
        self.features_dir = self.output_dir / "features"
        self.shard_writer = ShardWriter(
            output_dir=self.features_dir,
            shard_size=shard_size,
            layer_ids=layer_ids,
        )

    def get_tasks(self, split: str = "train", limit: Optional[int] = None) -> list[TaskInstance]:
        tasks = self.benchmark.list_tasks(split=split)
        return tasks[:limit] if limit else tasks

    def run_task(
        self,
        task: TaskInstance,
        max_steps: int = 10,
        rollout_k: int = 0,
        task_idx: Optional[int] = None,
        temperature: float = 0.0,
        episode_id: Optional[str] = None,
    ) -> dict:
        t_idx = task_idx if task_idx is not None else task.info.get("task_idx", 0)
        episode_id = episode_id or f"{task.task_id}_k{rollout_k}_{uuid.uuid4().hex[:6]}"

        try:
            # 1. Reset benchmark environment
            obs = self.benchmark.reset(task)

            # 2. Initialize chat messages
            messages = [
                {"role": "system", "content": task.system_prompt},
                {
                    "role": "user",
                    "content": f"Task: {task.instruction}\n\nObservation:\n{obs.observation_text}",
                },
            ]

            episode_steps: list[tuple[ExtractionResult, dict]] = []
            total_reward = 0.0

            for step_idx in range(max_steps):
                # Run forward pass & extract hidden states
                result: ExtractionResult = self.extractor.step_forward(
                    messages, temperature=temperature
                )
                action_text = result.completion_text.strip()

                obs: StepObservation = self.benchmark.step(action_text)
                total_reward += obs.step_reward

                step_meta = {
                    "episode_id": episode_id,
                    "task_id": task.task_id,
                    "task_idx": t_idx,
                    "rollout_k": rollout_k,
                    "round": step_idx,
                    "step_idx": step_idx,
                    "action": action_text,
                    "reward": obs.step_reward,
                    "is_done": obs.is_done,
                    "prompt_tokens": result.prompt_tokens,
                    "completion_tokens": result.completion_tokens,
                }
                episode_steps.append((result, step_meta))

                if obs.is_done:
                    break

                # Append conversation turns
                messages.append({"role": "assistant", "content": action_text})
                messages.append(
                    {"role": "user", "content": f"Observation:\n{obs.observation_text}"}
                )

            # 3. Evaluate ground-truth task success
            is_success = self.benchmark.evaluate(task)
            success_int = 1 if is_success else 0

            # 4. Stream step activations to shard writer with ground-truth success attached
            for result, meta in episode_steps:
                meta["success"] = success_int
                meta["n_rounds"] = len(episode_steps)
                meta["hit_max_rounds"] = len(episode_steps) >= max_steps and not is_success

                for anchor in self.anchors:
                    anchor_meta = dict(meta)
                    anchor_meta["anchor"] = anchor
                    feat_vec = result.get_anchor_state(anchor=anchor, layer_ids=self.layer_ids)
                    self.shard_writer.add_step(feat_vec, anchor_meta)

            return {
                "episode_id": episode_id,
                "task_id": task.task_id,
                "task_idx": t_idx,
                "rollout_k": rollout_k,
                "is_success": is_success,
                "total_reward": total_reward,
                "num_steps": len(episode_steps),
                "hit_max_rounds": len(episode_steps) >= max_steps and not is_success,
            }

        finally:
            self.benchmark.close()

    def run_benchmark(
        self,
        split: str = "train",
        max_tasks: Optional[int] = None,
        max_steps: int = 10,
        n_rollouts: int = 1,
        temperature: float = 0.0,
    ) -> list[dict]:
        tasks = self.get_tasks(split=split, limit=max_tasks)
        results = []
        total_episodes = len(tasks) * n_rollouts

        print(
            f"Starting benchmark run: {len(tasks)} tasks x {n_rollouts} rollouts = {total_episodes} episodes..."
        )
        ep_count = 0
        for i, task in enumerate(tasks):
            task_idx = task.info.get("task_idx", i)
            for k in range(n_rollouts):
                ep_count += 1
                effective_temp = temperature if (temperature > 0.0 or n_rollouts == 1) else 0.7
                print(
                    f"[{ep_count}/{total_episodes}] Running task {i+1}/{len(tasks)} "
                    f"({task.task_id}) [Rollout k={k}, temp={effective_temp}]..."
                )
                result = self.run_task(
                    task=task,
                    max_steps=max_steps,
                    rollout_k=k,
                    task_idx=task_idx,
                    temperature=effective_temp,
                )
                results.append(result)
                status = "SUCCESS" if result["is_success"] else "FAILED"
                print(
                    f"  Result: {status} (Steps: {result['num_steps']}, Reward: {result['total_reward']})"
                )

        # Finalize and flush all feature shards to disk
        self.shard_writer.close()

        # Save run summary Parquet
        summary_path = self.output_dir / "run_summary.parquet"
        save_run_summary(summary_path, results)
        success_count = sum(1 for r in results if r["is_success"])
        print(
            f"\nBenchmark Complete! Success Rate: {(success_count / len(results)) * 100:.1f}% "
            f"({success_count}/{len(results)})"
        )
        print(f"Saved feature shards to {self.features_dir}")
        print(f"Saved run summary to {summary_path}")

        return results
