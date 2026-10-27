from __future__ import annotations

from pathlib import Path
from typing import Optional
import uuid

from src.activations import ActivationExtractor, ExtractionResult
from src.benchmark.base import BaseBenchmark, StepObservation, TaskInstance
from src.probe.base import Probe, load_probe
from src.steering.strategy import BaseSteeringStrategy, SteeringDecision
from src.store import ShardWriter, save_run_summary


class SteeringRunner:
    """
    Executes benchmark tasks under probe-guided steering strategies.
    At each step:
      1. Uses a SteeringStrategy to evaluate activations with a trained Probe.
      2. Intervenes if warranted (e.g. vector steering or rejection sampling).
      3. Steps the benchmark environment with the selected action.
      4. Persists activations, probe predictions, and intervention telemetry into shards.
    """

    def __init__(
        self,
        extractor: ActivationExtractor,
        benchmark: BaseBenchmark,
        probe: Probe | str | Path,
        strategy: BaseSteeringStrategy,
        probe_layer: int,
        probe_anchor: str = "post_gen",
        output_dir: str | Path = "data/runs/steered",
        layer_ids: Optional[list[int]] = None,
        anchors: Optional[list[str]] = None,
        shard_size: int = 2000,
        verbose: bool = True,
    ):
        self.extractor = extractor
        self.benchmark = benchmark
        self.probe = load_probe(probe) if isinstance(probe, (str, Path)) else probe
        self.strategy = strategy
        self.probe_layer = probe_layer
        self.probe_anchor = probe_anchor
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.layer_ids = layer_ids
        self.anchors = anchors or ["post_gen", "pre_gen"]
        self.shard_size = shard_size
        self.verbose = verbose

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
            steered_steps = 0

            for step_idx in range(max_steps):
                # 3. Apply steering strategy to select action and extract activations
                decision: SteeringDecision = self.strategy.step(
                    messages=messages,
                    extractor=self.extractor,
                    probe=self.probe,
                    probe_layer=self.probe_layer,
                    probe_anchor=self.probe_anchor,
                    temperature=temperature,
                )
                action_text = decision.action_text

                # 4. Step benchmark with chosen action
                obs: StepObservation = self.benchmark.step(action_text)
                total_reward += obs.step_reward

                if decision.intervened:
                    steered_steps += 1

                if self.verbose:
                    first_line = action_text.split("\n")[0] if "\n" in action_text else action_text
                    steer_tag = " [STEERED]" if decision.intervened else ""
                    print(
                        f"    [Step {step_idx + 1}] Probe: {decision.probe_score:.3f}{steer_tag} "
                        f"-> Action: '{first_line[:65]}' -> Obs: '{obs.observation_text[:65]}'"
                    )

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
                    "prompt_tokens": decision.result.prompt_tokens,
                    "completion_tokens": decision.result.completion_tokens,
                    "probe_score": decision.probe_score,
                    "steered": decision.intervened,
                    **decision.strategy_info,
                }
                episode_steps.append((decision.result, step_meta))

                if obs.is_done:
                    break

                # 5. Append conversation turns
                messages.append({"role": "assistant", "content": action_text})
                messages.append(
                    {"role": "user", "content": f"Observation:\n{obs.observation_text}"}
                )

            # 6. Evaluate ground-truth task success
            is_success = self.benchmark.evaluate(task)
            success_int = 1 if is_success else 0

            # 7. Stream step activations to shard writer with ground-truth success attached
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
                "steered_steps": steered_steps,
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

        if self.verbose:
            print(
                f"Starting steered rollout: {len(tasks)} tasks x {n_rollouts} rollouts = {total_episodes} episodes..."
            )

        ep_count = 0
        for i, task in enumerate(tasks):
            task_idx = task.info.get("task_idx", i)
            for k in range(n_rollouts):
                ep_count += 1
                effective_temp = temperature if (temperature > 0.0 or n_rollouts == 1) else 0.7
                if self.verbose:
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
                if self.verbose:
                    print(
                        f"  Result: {status} (Steps: {result['num_steps']}, "
                        f"Steered Steps: {result['steered_steps']}, Reward: {result['total_reward']})"
                    )

        # Finalize and flush all feature shards to disk
        self.shard_writer.close()

        # Save run summary Parquet
        summary_path = self.output_dir / "run_summary.parquet"
        save_run_summary(summary_path, results)
        success_count = sum(1 for r in results if r["is_success"])
        total_steered = sum(r["steered_steps"] for r in results)
        total_steps = sum(r["num_steps"] for r in results)
        steered_pct = (total_steered / max(1, total_steps)) * 100

        if self.verbose and results:
            print(
                f"\nSteered Benchmark Complete! Success Rate: {(success_count / len(results)) * 100:.1f}% "
                f"({success_count}/{len(results)}) | Steered Interventions: {steered_pct:.1f}% ({total_steered}/{total_steps} steps)"
            )
            print(f"Saved feature shards to {self.features_dir}")
            print(f"Saved run summary to {summary_path}")

        return results
