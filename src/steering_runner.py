from __future__ import annotations

from pathlib import Path
from typing import Optional
import numpy as np
import torch

from src.activations import ActivationExtractor
from src.benchmark.base import BaseBenchmark, StepObservation, TaskInstance
from src.probe.base import Probe
from src.steering.base import Steerer
from src.store import ActivationDataset, ShardWriter, save_run_summary


class SteeringRunner:
    """
    Executes multi-turn agent rollouts under probe-guided token-level steering.
    At each turn:
      1. Generates action tokens autoregressively.
      2. For each token, evaluates residual-stream activation with a Probe.
      3. If probe score < threshold, intervenes via Steerer.intervene() to replace activation.
      4. Steps the benchmark environment until task completes or dynamic budget cap is reached.
    """

    def __init__(
        self,
        extractor: ActivationExtractor,
        probe: Probe,
        steerer: Steerer,
        benchmark: BaseBenchmark,
        output_dir: str | Path = "data/runs/steered",
        layer_id: int = 20,
        threshold: float = 0.5,
        shard_size: int = 2000,
        verbose: bool = True,
    ):
        self.extractor = extractor
        self.probe = probe
        self.steerer = steerer
        self.benchmark = benchmark

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.layer_id = layer_id
        self.threshold = threshold
        self.shard_size = shard_size
        self.verbose = verbose

        self.features_dir = self.output_dir / "features"
        self.shard_writer = ShardWriter(
            output_dir=self.features_dir,
            shard_size=shard_size,
            layer_ids=[layer_id],
        )

    def train_intervention(self, train_set: ActivationDataset) -> None:
        """Trains both the prediction probe and the steering module on training activations."""
        self.probe.fit(train_set)
        self.steerer.fit(train_set)

    def get_tasks(
        self,
        tasks: Optional[list[TaskInstance]] = None,
        limit: Optional[int] = None,
    ) -> list[TaskInstance]:
        """Returns target task instances, defaulting to benchmark.list_tasks()."""
        task_list = tasks if tasks is not None else self.benchmark.list_tasks()
        return task_list[:limit] if limit else task_list

    def run_task(
        self,
        task: TaskInstance,
        max_steps: int = 35,
        dynamic_steps: bool = True,
        max_steer_turns: Optional[int] = None,
        temperature: float = 0.0,
    ) -> dict:
        """
        Runs a multi-turn task rollout until completion (is_done) or hitting dynamic step budget.
        On every turn, actions are generated token-at-a-time with probe checking & steerer intervention.
        If max_steer_turns is specified, steering hooks are only applied on the first N turns.
        """
        effective_max_steps = task.info.get("max_steps", max_steps) if dynamic_steps else max_steps

        try:
            obs = self.benchmark.reset(task)
            messages = [
                {"role": "system", "content": task.system_prompt},
                {
                    "role": "user",
                    "content": f"Task: {task.instruction}\n\nObservation:\n{obs.observation_text}",
                },
            ]

            all_generated_tokens = []
            all_probe_scores = []
            all_steered_tokens = []
            total_reward = 0.0
            turn_count = 0
            last_act = None

            for step_idx in range(effective_max_steps):
                turn_count += 1
                can_steer = (max_steer_turns is None) or (step_idx < max_steer_turns)

                prompt_text = self.extractor.tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                inputs = self.extractor.tokenizer(prompt_text, return_tensors="pt")
                input_ids = inputs.input_ids.to(self.extractor.device)

                turn_tokens = []
                turn_steered = []

                for _ in range(64):
                    if can_steer:
                        step_info = {"score": 0.0, "steered": False}

                        def hook_fn(module, input, output):
                            nonlocal last_act
                            is_tuple = isinstance(output, tuple)
                            hidden = output[0] if is_tuple else output

                            act_np = hidden[:, -1, :].detach().cpu().to(torch.float32).numpy()
                            score = float(self.probe.predict(act_np, return_proba=True)[0])
                            step_info["score"] = score
                            last_act = hidden[:, -1, :].detach().clone()

                            if score < self.threshold:
                                steered_np = self.steerer.intervene(act_np)
                                steered_tensor = torch.from_numpy(steered_np).to(device=hidden.device, dtype=hidden.dtype)
                                hidden = hidden.clone()
                                hidden[:, -1, :] = steered_tensor
                                step_info["steered"] = True
                            else:
                                step_info["steered"] = False

                            return (hidden,) + output[1:] if is_tuple else hidden

                        handle = self.extractor.model.model.layers[self.layer_id].register_forward_hook(hook_fn)
                        try:
                            with torch.inference_mode():
                                outputs = self.extractor.model(input_ids=input_ids, use_cache=False)
                        finally:
                            handle.remove()

                        all_probe_scores.append(step_info["score"])
                        turn_steered.append(step_info["steered"])
                    else:
                        with torch.inference_mode():
                            outputs = self.extractor.model(input_ids=input_ids, use_cache=False)
                        turn_steered.append(False)

                    logits = outputs.logits[0, -1, :]

                    if temperature > 0.0:
                        probs = torch.softmax(logits / temperature, dim=-1)
                        next_token = torch.multinomial(probs, num_samples=1)
                    else:
                        next_token = torch.argmax(logits, dim=-1, keepdim=True)

                    next_token_id = next_token.item()
                    turn_tokens.append(next_token_id)

                    eos_id = getattr(self.extractor.tokenizer, "eos_token_id", None)
                    if eos_id is not None and next_token_id == eos_id:
                        break

                    input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)

                all_generated_tokens.extend(turn_tokens)
                all_steered_tokens.extend(turn_steered)

                action_text = self.extractor.tokenizer.decode(
                    turn_tokens, skip_special_tokens=True
                ).strip()

                obs: StepObservation = self.benchmark.step(action_text)
                total_reward += obs.step_reward

                steered_in_turn = sum(turn_steered)
                if self.verbose:
                    first_line = action_text.split("\n")[0] if "\n" in action_text else action_text
                    steer_tag = f" [{steered_in_turn}/{len(turn_tokens)} steered]" if steered_in_turn > 0 else (" (unsteered turn)" if not can_steer else "")
                    print(
                        f"    [Step {step_idx + 1}/{effective_max_steps}]{steer_tag} "
                        f"Action: '{first_line[:60]}' -> Obs: '{obs.observation_text[:60]}'"
                    )

                if obs.is_done:
                    break

                messages.append({"role": "assistant", "content": action_text})
                messages.append(
                    {"role": "user", "content": f"Observation:\n{obs.observation_text}"}
                )

            is_success = self.benchmark.evaluate(task)
            num_steered = sum(all_steered_tokens)
            total_tokens = len(all_generated_tokens)

            if self.verbose:
                status = "SUCCESS" if is_success else "FAILED"
                print(
                    f"  -> Result: {status} in {turn_count}/{effective_max_steps} turns | "
                    f"Reward: {total_reward:.1f} | Tokens: {total_tokens} ({num_steered} steered)"
                )

            # Stream activations to shard writer
            if self.shard_writer is not None and last_act is not None:
                feat_vec = last_act.unsqueeze(0).to(torch.float16).cpu()
                step_meta = {
                    "task_id": task.task_id,
                    "is_success": int(is_success),
                    "num_steps": turn_count,
                    "max_steps": effective_max_steps,
                    "total_reward": total_reward,
                    "total_tokens": total_tokens,
                    "steered_tokens": num_steered,
                    "avg_probe_score": float(np.mean(all_probe_scores)) if all_probe_scores else 0.0,
                }
                self.shard_writer.add_step(feat_vec, step_meta)

            return {
                "task_id": task.task_id,
                "is_success": bool(is_success),
                "num_steps": turn_count,
                "max_steps": effective_max_steps,
                "hit_max_rounds": turn_count >= effective_max_steps and not is_success,
                "total_reward": total_reward,
                "num_tokens": total_tokens,
                "steered_steps": num_steered,
                "avg_probe_score": float(np.mean(all_probe_scores)) if all_probe_scores else 0.0,
            }

        finally:
            self.benchmark.close()

    def run_benchmark(
        self,
        tasks: Optional[list[TaskInstance]] = None,
        max_tasks: Optional[int] = None,
        max_steps: int = 35,
        dynamic_steps: bool = True,
        max_steer_turns: Optional[int] = None,
        temperature: float = 0.0,
    ) -> list[dict]:
        """Runs multi-turn evaluation across specified tasks (e.g. 20% validation split) or all benchmark tasks."""
        target_tasks = self.get_tasks(tasks=tasks, limit=max_tasks)
        results = []

        if self.verbose:
            steer_mode = f"first {max_steer_turns} turns" if max_steer_turns is not None else "all turns"
            print(f"Starting steered multi-turn evaluation on {len(target_tasks)} tasks (Steering: {steer_mode})...")

        for i, task in enumerate(target_tasks):
            if self.verbose:
                print(f"[{i + 1}/{len(target_tasks)}] Evaluating Task: {task.task_id}")
            res = self.run_task(
                task=task,
                max_steps=max_steps,
                dynamic_steps=dynamic_steps,
                max_steer_turns=max_steer_turns,
                temperature=temperature,
            )
            results.append(res)

        self.shard_writer.close()
        summary_path = self.output_dir / "run_summary.parquet"
        save_run_summary(summary_path, results)

        success_count = sum(1 for r in results if r.get("is_success", False))
        success_rate = (success_count / max(1, len(results))) * 100
        total_tokens = sum(r["num_tokens"] for r in results)
        total_steered = sum(r["steered_steps"] for r in results)
        steered_pct = (total_steered / max(1, total_tokens)) * 100

        if self.verbose and results:
            print(
                f"\nEvaluation Complete! Success Rate: {success_rate:.1f}% ({success_count}/{len(results)}) | "
                f"Steered Tokens: {steered_pct:.1f}% ({total_steered}/{total_tokens})"
            )
            print(f"Saved feature shards to {self.features_dir}")
            print(f"Saved run summary to {summary_path}")

        return results

    # Alias for backwards compatibility
    run_eval_tasks = run_benchmark
