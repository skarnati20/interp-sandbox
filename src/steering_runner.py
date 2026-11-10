from __future__ import annotations

from pathlib import Path
from typing import Optional
import numpy as np
import torch

from src.activations import ActivationExtractor, ExtractionResult
from src.benchmark.base import BaseBenchmark, StepObservation, TaskInstance
from src.probe.base import Probe
from src.steering.base import Steerer
from src.store import ActivationDataset, ShardWriter, save_run_summary


class SteeringRunner:
    """
    Executes multi-turn agent rollouts under probe-guided turn-level steering.
    Supports two intervention methods:
      - 'vector': Injects steering vector derived from Steerer.intervene() into target layer.
      - 'resample': Rejection sampling / Best-of-N: regenerates candidate actions up to budget
                    until probe score >= threshold.
    """

    def __init__(
        self,
        extractor: ActivationExtractor,
        probe: Probe,
        steerer: Steerer,
        benchmark: BaseBenchmark,
        output_dir: str | Path = "data/runs/steered",
        layer_id: int = 20,
        anchor: str = "post_gen",
        threshold: float = 0.5,
        method: str = "vector",
        max_resample_attempts: int = 5,
        resample_temperature: float = 0.7,
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
        self.anchor = anchor.strip().lower()
        self.threshold = threshold
        self.method = method.strip().lower()
        self.max_resample_attempts = max(1, max_resample_attempts)
        self.resample_temperature = resample_temperature
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

    def _steer_forward(
        self,
        messages: list[dict],
        delta_vector: np.ndarray,
        temperature: float = 0.0,
    ) -> ExtractionResult:
        """Runs extractor generation while injecting the intervention delta vector into the target layer."""
        target_layer = self.extractor.model.model.layers[self.layer_id]
        vec_tensor = torch.from_numpy(delta_vector).to(device=self.extractor.device)

        def hook_fn(module, input, output):
            is_tuple = isinstance(output, tuple)
            hidden = output[0] if is_tuple else output
            v = vec_tensor.to(dtype=hidden.dtype)
            hidden = hidden + v
            return (hidden,) + output[1:] if is_tuple else hidden

        handle = target_layer.register_forward_hook(hook_fn)
        try:
            return self.extractor.step_forward(messages, temperature=temperature)
        finally:
            handle.remove()

    def run_turn(
        self,
        messages: list[dict],
        can_steer: bool = True,
        temperature: float = 0.0,
    ) -> tuple[ExtractionResult, float, bool]:
        """
        Executes one turn of action generation:
          - If method == 'resample': Regenerates actions up to max_resample_attempts until probe score >= threshold.
          - If method == 'vector' and anchor == 'pre_gen': Evaluates prompt decision token; steers forward pass if score < threshold.
          - If method == 'vector' and anchor == 'post_gen': Evaluates candidate completion; regenerates with steering vector if score < threshold.
        Returns: (result, probe_score, intervened)
        """
        # --- Resampling Intervention (Best-of-N Rejection Sampling) ---
        if self.method == "resample":
            candidates: list[tuple[ExtractionResult, float, bool]] = []
            for attempt in range(self.max_resample_attempts):
                curr_temp = temperature if attempt == 0 else max(self.resample_temperature, 0.1)
                result = self.extractor.step_forward(messages, temperature=curr_temp)

                feat = result.get_anchor_state(anchor=self.anchor, layer_ids=[self.layer_id])
                act_np = feat.detach().cpu().to(torch.float32).numpy().reshape(1, -1)
                score = float(self.probe.predict(act_np, return_proba=True)[0])
                candidates.append((result, score, attempt > 0))

                # Accept immediately if score passes threshold, or if steering is disabled on this turn
                if score >= self.threshold or not can_steer:
                    return result, score, (attempt > 0)

            # Budget exhausted: pick candidate with highest predicted success
            best_result, best_score, best_intervened = max(candidates, key=lambda c: c[1])
            return best_result, best_score, (len(candidates) > 1)

        # --- Vector Steering Intervention ---
        if self.anchor == "pre_gen":
            # Quick forward pass over prompt to extract pre_gen activation
            prompt_text = self.extractor.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = self.extractor.tokenizer(prompt_text, return_tensors="pt").to(self.extractor.device)
            with torch.inference_mode():
                out = self.extractor.model(**inputs, output_hidden_states=True)

            num_layers = len(out.hidden_states) - 1
            layer_idx = min(max(0, self.layer_id + 1), num_layers)
            pre_gen_tensor = out.hidden_states[layer_idx][0, -1, :]
            act_np = pre_gen_tensor.detach().cpu().to(torch.float32).numpy().reshape(1, -1)

            score = float(self.probe.predict(act_np, return_proba=True)[0])
            intervened = False

            if can_steer and score < self.threshold:
                steered_act = self.steerer.intervene(act_np)
                delta = steered_act - act_np
                result = self._steer_forward(messages, delta_vector=delta, temperature=temperature)
                intervened = True
            else:
                result = self.extractor.step_forward(messages, temperature=temperature)

            return result, score, intervened

        else:
            # Default: 'post_gen' anchor vector steering
            result = self.extractor.step_forward(messages, temperature=temperature)

            feat = result.get_anchor_state(anchor="post_gen", layer_ids=[self.layer_id])
            act_np = feat.detach().cpu().to(torch.float32).numpy().reshape(1, -1)

            score = float(self.probe.predict(act_np, return_proba=True)[0])
            intervened = False

            if can_steer and score < self.threshold:
                steered_act = self.steerer.intervene(act_np)
                delta = steered_act - act_np
                result = self._steer_forward(messages, delta_vector=delta, temperature=temperature)
                intervened = True

            return result, score, intervened

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
        On every turn, the action is evaluated and intervened at the chosen anchor token position.
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

            episode_steps: list[tuple[ExtractionResult, dict]] = []
            turn_scores = []
            turn_steered = []
            actions = []
            total_reward = 0.0
            turn_count = 0

            for step_idx in range(effective_max_steps):
                turn_count += 1
                can_steer = (max_steer_turns is None) or (step_idx < max_steer_turns)

                result, score, intervened = self.run_turn(
                    messages=messages,
                    can_steer=can_steer,
                    temperature=temperature,
                )

                action_text = result.completion_text.strip()
                turn_scores.append(score)
                turn_steered.append(intervened)
                actions.append(action_text)

                # Step benchmark environment
                obs: StepObservation = self.benchmark.step(action_text)
                total_reward += obs.step_reward

                if self.verbose:
                    first_line = action_text.split("\n")[0] if "\n" in action_text else action_text
                    mode_tag = "RESAMPLED" if self.method == "resample" else "STEERED"
                    steer_tag = f" [{mode_tag}]" if intervened else (" (unsteered turn)" if not can_steer else "")
                    print(
                        f"    [Turn {step_idx + 1}/{effective_max_steps}] P(success)={score:.3f}{steer_tag} "
                        f"Action: '{first_line[:60]}' -> Obs: '{obs.observation_text[:60]}'"
                    )

                step_meta = {
                    "task_id": task.task_id,
                    "round": step_idx,
                    "step_idx": step_idx,
                    "action": action_text,
                    "reward": obs.step_reward,
                    "is_done": obs.is_done,
                    "prompt_tokens": result.prompt_tokens,
                    "completion_tokens": result.completion_tokens,
                    "probe_score": score,
                    "steered": intervened,
                }
                episode_steps.append((result, step_meta))

                if obs.is_done:
                    break

                messages.append({"role": "assistant", "content": action_text})
                messages.append(
                    {"role": "user", "content": f"Observation:\n{obs.observation_text}"}
                )

            # Evaluate ground-truth task completion
            is_success = self.benchmark.evaluate(task)
            num_steered = sum(turn_steered)

            if self.verbose:
                status = "SUCCESS" if is_success else "FAILED"
                print(
                    f"  -> Result: {status} in {turn_count}/{effective_max_steps} turns | "
                    f"Reward: {total_reward:.1f} | Intervened Turns: {num_steered}/{turn_count}"
                )

            # Stream turn activations to shard writer with ground-truth success attached
            success_int = 1 if is_success else 0
            if self.shard_writer is not None:
                for res, meta in episode_steps:
                    meta["success"] = success_int
                    meta["n_rounds"] = len(episode_steps)
                    meta["hit_max_rounds"] = len(episode_steps) >= effective_max_steps and not is_success
                    meta["anchor"] = self.anchor
                    feat_vec = res.get_anchor_state(anchor=self.anchor, layer_ids=[self.layer_id])
                    self.shard_writer.add_step(feat_vec, meta)

            return {
                "task_id": task.task_id,
                "is_success": bool(is_success),
                "num_steps": turn_count,
                "max_steps": effective_max_steps,
                "hit_max_rounds": turn_count >= effective_max_steps and not is_success,
                "total_reward": total_reward,
                "steered_steps": num_steered,
                "avg_probe_score": float(np.mean(turn_scores)) if turn_scores else 0.0,
                "actions": actions,
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
            print(f"Starting multi-turn evaluation on {len(target_tasks)} tasks (Method: {self.method}, Anchor: {self.anchor}, Steering: {steer_mode})...")

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
        total_turns = sum(r["num_steps"] for r in results)
        total_steered = sum(r["steered_steps"] for r in results)
        steered_pct = (total_steered / max(1, total_turns)) * 100

        if self.verbose and results:
            print(
                f"\nEvaluation Complete! Success Rate: {success_rate:.1f}% ({success_count}/{len(results)}) | "
                f"Intervened Turns: {steered_pct:.1f}% ({total_steered}/{total_turns})"
            )
            print(f"Saved feature shards to {self.features_dir}")
            print(f"Saved run summary to {summary_path}")

        return results

    # Alias for backwards compatibility
    run_eval_tasks = run_benchmark
