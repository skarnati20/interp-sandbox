from pathlib import Path
from typing import Optional
import uuid

from src.activations import ActivationExtractor, ExtractionResult
from src.benchmark import BaseBenchmark, StepObservation, TaskInstance
from src.store import save_activation_result, save_run_summary


class Runner:
    """
    Executes benchmark rollouts across LLM agent tasks, records trajectory turns,
    and persists activation tensors with grouped task metadata for probing.
    """

    def __init__(
        self,
        extractor: ActivationExtractor,
        benchmark: BaseBenchmark,
        output_dir: str | Path = "data/runs",
        save_mode: str = "decision",
    ):
        self.extractor = extractor
        self.benchmark = benchmark
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.save_mode = save_mode

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
        episode_dir = self.output_dir / episode_id

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

            step_history = []
            total_reward = 0.0

            for step_idx in range(max_steps):
                # Run forward pass & extract hidden states
                result: ExtractionResult = self.extractor.step_forward(
                    messages, temperature=temperature
                )
                action_text = result.completion_text.strip()

                obs: StepObservation = self.benchmark.step(action_text)
                total_reward += obs.step_reward

                # Persist step activations + metadata with grouped probing keys
                step_dir = episode_dir / f"step_{step_idx}"
                save_activation_result(
                    dir_path=step_dir,
                    result=result,
                    mode=self.save_mode,
                    extra_metadata={
                        "episode_id": episode_id,
                        "task_id": task.task_id,
                        "task_idx": t_idx,
                        "rollout_k": rollout_k,
                        "round": step_idx,
                        "step_idx": step_idx,
                        "action": action_text,
                        "reward": obs.step_reward,
                        "is_done": obs.is_done,
                    },
                )

                step_history.append({
                    "step_idx": step_idx,
                    "action": action_text,
                    "reward": obs.step_reward,
                })

                if obs.is_done:
                    break

                # Append conversation turns
                messages.append({"role": "assistant", "content": action_text})
                messages.append(
                    {"role": "user", "content": f"Observation:\n{obs.observation_text}"}
                )

            # 3. Evaluate ground-truth task success
            is_success = self.benchmark.evaluate(task)

            return {
                "episode_id": episode_id,
                "task_id": task.task_id,
                "task_idx": t_idx,
                "rollout_k": rollout_k,
                "is_success": is_success,
                "total_reward": total_reward,
                "num_steps": len(step_history),
                "hit_max_rounds": len(step_history) >= max_steps and not is_success,
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
                # Use stochastic temperature for multi-rollout if temperature is default 0.0
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

        # Save run summary using store abstraction
        summary_path = self.output_dir / "run_summary.parquet"
        save_run_summary(summary_path, results)
        success_count = sum(1 for r in results if r["is_success"])
        print(
            f"\nBenchmark Complete! Success Rate: {(success_count / len(results)) * 100:.1f}% "
            f"({success_count}/{len(results)})"
        )
        print(f"Saved run summary to {summary_path}")

        return results
