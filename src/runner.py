from __future__ import annotations

from pathlib import Path
from typing import Optional
import json
import math
import uuid

from src.activations import ActivationExtractor, ExtractionResult, validate_layer_ids
from src.benchmark.base import BaseBenchmark, StepObservation, TaskInstance
from src.store import ShardWriter, save_run_summary
from src.splits import content_hash


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
        verbose: bool = True,
    ):
        self.extractor = extractor
        self.benchmark = benchmark
        self.output_dir = Path(output_dir)
        self._require_empty_output()
        self.layer_ids = validate_layer_ids(layer_ids, extractor.num_layers)
        self.anchors = anchors or ["post_gen", "pre_gen"]
        self.verbose = verbose
        catalog = benchmark.task_catalog()
        self.model_metadata = getattr(extractor, "metadata", {"model_id": "unknown", "model_revision": None})
        self._base_run_metadata = {
            **self.model_metadata, "layer_ids": self.layer_ids, "anchors": self.anchors,
            "sources": list({content_hash(record.provenance): record.provenance for record in catalog}.values()),
        }
        self._base_run_metadata["benchmark_settings"] = benchmark.collection_settings()
        self.shard_size = shard_size
        self.shard_writer = None
        self._run_started = False

    def _require_empty_output(self) -> None:
        self.validate_output_directory(self.output_dir)

    @staticmethod
    def validate_output_directory(output_dir: str | Path) -> None:
        """Collectors can reject occupied outputs before loading a model or source."""
        directory = Path(output_dir)
        if directory.is_symlink() or (directory.exists() and (
            not directory.is_dir() or any(directory.iterdir())
        )):
            raise FileExistsError(
                f"Collection output must be a new or empty directory: {directory}. "
                "Resume is not supported; use a fresh output directory."
            )

    def _start_collection(self, tasks, *, mode, source_split, max_tasks, max_steps,
                          n_rollouts, temperature, max_new_tokens, rollout_k=None):
        if self._run_started:
            raise RuntimeError("This Runner has already started a collection; create a new Runner and output directory")
        for name, value in (("max_steps", max_steps), ("n_rollouts", n_rollouts),
                            ("max_new_tokens", max_new_tokens)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not math.isfinite(temperature) or temperature < 0:
            raise ValueError("temperature must be finite and nonnegative")
        self._require_empty_output()
        records = [self.benchmark.task_record(task) for task in tasks]
        effective_temp = temperature if temperature > 0 or n_rollouts == 1 else 0.7
        self.run_metadata = {
            **self._base_run_metadata,
            "mode": mode, "status": "running", "completed_episodes": 0,
            "n_rollouts": n_rollouts, "max_steps": max_steps,
            "generation": {
                "model_defaults": getattr(self.extractor, "generation_defaults", {}),
                "requested_temperature": temperature, "effective_temperature": effective_temp,
                "do_sample": effective_temp > 0, "max_new_tokens": max_new_tokens,
                "return_dict_in_generate": True, "output_hidden_states": True,
            },
            "task_selection": {
                "source_split": source_split, "max_tasks": max_tasks,
                "selected_task_count": len(tasks),
                "tasks": [{"task_uid": record.task_uid, "task_id": task.task_id,
                           "source_task_id": record.source_task_id or record.task_id,
                           "source_split": record.source_split, "task_idx": task.info.get("task_idx", i)}
                          for i, (task, record) in enumerate(zip(tasks, records))],
            },
        }
        if rollout_k is not None:
            self.run_metadata["rollout_k"] = rollout_k
        self.output_dir.mkdir(parents=True, exist_ok=True)
        # An exclusive metadata file claims this output even if two runners
        # were constructed while it was empty. No matching-metadata overwrite.
        with (self.output_dir / "run_metadata.json").open("x", encoding="utf-8") as fp:
            json.dump(self.run_metadata, fp, indent=2, sort_keys=True)
            fp.write("\n")
        self._run_started = True
        self.features_dir = self.output_dir / "features"
        self.shard_writer = ShardWriter(self.features_dir, shard_size=self.shard_size, layer_ids=self.layer_ids)
        return effective_temp

    def _finish_collection(self, results: list[dict], complete: bool) -> None:
        self.shard_writer.close()
        save_run_summary(self.output_dir / "run_summary.parquet", results)
        self.run_metadata.update(status="complete" if complete else "failed", completed_episodes=len(results))
        (self.output_dir / "run_metadata.json").write_text(
            json.dumps(self.run_metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    def get_tasks(self, split: Optional[str] = None, limit: Optional[int] = None) -> list[TaskInstance]:
        if limit is not None and (isinstance(limit, bool) or not isinstance(limit, int) or limit < 0):
            raise ValueError("Task limit must be a nonnegative integer")
        tasks = self.benchmark.list_tasks(split=split)
        return tasks[:limit] if limit is not None else tasks

    def run_task(
        self,
        task: TaskInstance,
        max_steps: int = 10,
        rollout_k: int = 0,
        task_idx: Optional[int] = None,
        temperature: float = 0.0,
        episode_id: Optional[str] = None,
        max_new_tokens: int = 512,
    ) -> dict:
        """Collect one episode into a fresh output, with the same protection as a batch."""
        self._start_collection([task], mode="single_task", source_split=task.source_split,
                               max_tasks=1, max_steps=max_steps, n_rollouts=1,
                               temperature=temperature, max_new_tokens=max_new_tokens, rollout_k=rollout_k)
        results = []
        complete = False
        try:
            result = self._run_task(task, max_steps=max_steps, rollout_k=rollout_k, task_idx=task_idx,
                                    temperature=temperature, episode_id=episode_id, max_new_tokens=max_new_tokens)
            results.append(result)
            complete = True
            return result
        finally:
            self._finish_collection(results, complete)

    def _run_task(
        self,
        task: TaskInstance,
        max_steps: int = 10,
        rollout_k: int = 0,
        task_idx: Optional[int] = None,
        temperature: float = 0.0,
        episode_id: Optional[str] = None,
        max_new_tokens: int = 512,
    ) -> dict:
        t_idx = task_idx if task_idx is not None else task.info.get("task_idx", 0)
        episode_id = episode_id or f"{task.task_id}_k{rollout_k}_{uuid.uuid4().hex[:6]}"
        record = self.benchmark.task_record(task)
        task_metadata = {
            "task_uid": record.task_uid, "namespace": record.namespace, "group_id": record.group_id,
            "source_task_id": record.source_task_id or record.task_id,
            "source_split": record.source_split, "source_provenance": record.provenance,
            "probe_split": None, "manifest_id": None,
        }
        task_metadata["grouping"] = "task"
        task_metadata.update(self.model_metadata)
        task_metadata["layer_ids"] = self.layer_ids

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
                    messages, temperature=temperature, max_new_tokens=max_new_tokens
                )
                action_text = result.completion_text.strip()

                obs: StepObservation = self.benchmark.step(action_text)
                total_reward += obs.step_reward

                if self.verbose:
                    first_line = action_text.split('\n')[0] if '\n' in action_text else action_text
                    print(f"    [Step {step_idx + 1}] Action: '{first_line[:70]}' -> Obs: '{obs.observation_text[:70]}'")

                step_meta = {
                    **task_metadata,
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
                **task_metadata,
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
        split: Optional[str] = None,
        max_tasks: Optional[int] = None,
        max_steps: int = 10,
        n_rollouts: int = 1,
        temperature: float = 0.0,
        max_new_tokens: int = 512,
    ) -> list[dict]:
        tasks = self.get_tasks(split=split, limit=max_tasks)
        effective_temp = self._start_collection(
            tasks, mode="benchmark", source_split=split, max_tasks=max_tasks, max_steps=max_steps,
            n_rollouts=n_rollouts, temperature=temperature, max_new_tokens=max_new_tokens,
        )
        results = []
        total_episodes = len(tasks) * n_rollouts

        print(
            f"Starting benchmark run: {len(tasks)} tasks x {n_rollouts} rollouts = {total_episodes} episodes..."
        )
        ep_count = 0
        complete = False
        try:
            for i, task in enumerate(tasks):
                task_idx = task.info.get("task_idx", i)
                for k in range(n_rollouts):
                    ep_count += 1
                    print(
                        f"[{ep_count}/{total_episodes}] Running task {i+1}/{len(tasks)} "
                        f"({task.task_id}) [Rollout k={k}, temp={effective_temp}]..."
                    )
                    result = self._run_task(
                        task=task, max_steps=max_steps, rollout_k=k, task_idx=task_idx,
                        temperature=effective_temp, max_new_tokens=max_new_tokens,
                    )
                    results.append(result)
                    status = "SUCCESS" if result["is_success"] else "FAILED"
                    print(f"  Result: {status} (Steps: {result['num_steps']}, Reward: {result['total_reward']})")
            complete = True
        finally:
            # Keep completed episodes readable even when a later episode fails.
            self._finish_collection(results, complete)

        summary_path = self.output_dir / "run_summary.parquet"
        success_count = sum(1 for r in results if r["is_success"])
        print(
            f"\nBenchmark Complete! Success Rate: {(success_count / len(results)) * 100 if results else 0:.1f}% "
            f"({success_count}/{len(results)})"
        )
        print(f"Saved feature shards to {self.features_dir}")
        print(f"Saved run summary to {summary_path}")
        return results
