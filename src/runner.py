from pathlib import Path
from typing import Optional
import uuid

from src.activations import ActivationExtractor, ExtractionResult
from src.benchmark import BaseBenchmark, StepObservation, TaskInstance
from src.store import save_activation_result, save_run_summary


class Runner:
    def __init__(
        self,
        extractor: ActivationExtractor,
        benchmark: BaseBenchmark,
        output_dir: str | Path = "data/runs",
    ):
        self.extractor = extractor
        self.benchmark = benchmark
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def get_tasks(self, split: str = "train", limit: Optional[int] = None) -> list[TaskInstance]:
        tasks = self.benchmark.list_tasks(split=split)
        return tasks[:limit] if limit else tasks

    def run_task(
        self,
        task: TaskInstance,
        max_steps: int = 10,
        episode_id: Optional[str] = None,
    ) -> dict:
        episode_id = episode_id or f"{task.task_id}_{uuid.uuid4().hex[:6]}"
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
                result: ExtractionResult = self.extractor.step_forward(messages)
                action_text = result.completion_text.strip()

                obs: StepObservation = self.benchmark.step(action_text)
                total_reward += obs.step_reward

                # Persist step activations + metadata
                step_dir = episode_dir / f"step_{step_idx}"
                save_activation_result(
                    dir_path=step_dir,
                    result=result,
                    extra_metadata={
                        "episode_id": episode_id,
                        "task_id": task.task_id,
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
                "is_success": is_success,
                "total_reward": total_reward,
                "num_steps": len(step_history),
            }

        finally:
            self.benchmark.close()

    def run_benchmark(
        self,
        split: str = "train",
        max_tasks: Optional[int] = None,
        max_steps: int = 10,
    ) -> list[dict]:
        tasks = self.get_tasks(split=split, limit=max_tasks)
        results = []

        print(f"Starting benchmark run: {len(tasks)} tasks...")
        for i, task in enumerate(tasks):
            print(f"[{i+1}/{len(tasks)}] Running task: {task.task_id}...")
            result = self.run_task(task, max_steps=max_steps)
            results.append(result)
            status = "SUCCESS" if result["is_success"] else "FAILED"
            print(f"  Result: {status} (Steps: {result['num_steps']}, Reward: {result['total_reward']})")

        # Save run summary using store abstraction
        if results:
            summary_path = self.output_dir / "run_summary.parquet"
            save_run_summary(summary_path, results)
            success_count = sum(1 for r in results if r["is_success"])
            success_rate = success_count / len(results)
            print(f"\nBenchmark Complete! Success Rate: {success_rate * 100:.1f}%")
            print(f"Saved run summary to {summary_path}")

        return results
