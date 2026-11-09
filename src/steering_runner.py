from __future__ import annotations

from pathlib import Path
from typing import Optional
import numpy as np
import torch

from src.activations import ActivationExtractor
from src.benchmark.base import TaskInstance
from src.probe.base import Probe
from src.steering.base import Steerer
from src.store import ActivationDataset, ShardWriter, save_run_summary


class SteeringRunner:
    """
    Runs generation token-by-token:
    On each token, checks residual-stream activations with a Probe, and intervenes
    with Steerer.intervene() to replace the activation when below threshold.
    """

    def __init__(
        self,
        extractor: ActivationExtractor,
        probe: Probe,
        steerer: Steerer,
        eval_set: list[TaskInstance],
        output_dir: str | Path = "data/runs/steered",
        layer_id: int = 20,
        threshold: float = 0.5,
        shard_size: int = 2000,
        verbose: bool = True,
    ):
        self.extractor = extractor
        self.probe = probe
        self.steerer = steerer
        self.eval_set = eval_set

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

    def _get_model_layers(self):
        """Resolves the transformer decoder layers module list from the extractor's model."""
        model = self.extractor.model
        if hasattr(model, "model") and hasattr(model.model, "layers"):
            return model.model.layers
        elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
            return model.transformer.h
        elif hasattr(model, "layers"):
            return model.layers
        raise AttributeError("Could not locate transformer decoder layers on extractor model.")

    def train_intervention(self, train_set: ActivationDataset) -> None:
        """Trains both the prediction probe and the steering module on training activations."""
        self.probe.fit(train_set)
        self.steerer.fit(train_set)

    def get_tasks(self, limit: Optional[int] = None) -> list[TaskInstance]:
        """Returns evaluation task instances from eval_set."""
        tasks = self.eval_set if self.eval_set is not None else []
        return tasks[:limit] if limit else tasks

    def run_task(
        self,
        task: TaskInstance,
        max_steps: int = 128,
        temperature: float = 0.0,
    ) -> dict:
        """
        Runs generation a token at a time:
        On each token, evaluates the layer activation with the probe.
        If below threshold, replaces the activation with steerer.intervene(activation).
        """
        messages = [
            {"role": "system", "content": task.system_prompt},
            {"role": "user", "content": task.instruction},
        ]

        prompt_text = self.extractor.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.extractor.tokenizer(prompt_text, return_tensors="pt")
        input_ids = inputs.input_ids.to(self.extractor.device)

        layers = self._get_model_layers()
        target_layer = layers[min(max(0, self.layer_id), len(layers) - 1)]

        generated_token_ids = []
        token_probe_scores = []
        steered_tokens = []

        for _ in range(max_steps):
            step_info = {"score": 0.0, "steered": False}

            def hook_fn(module, input, output):
                is_tuple = isinstance(output, tuple)
                hidden = output[0] if is_tuple else output

                act_np = hidden[:, -1, :].detach().cpu().to(torch.float32).numpy()
                score = float(self.probe.predict(act_np, return_proba=True)[0])
                step_info["score"] = score

                if score < self.threshold:
                    steered_np = self.steerer.intervene(act_np)
                    steered_tensor = torch.from_numpy(steered_np).to(device=hidden.device, dtype=hidden.dtype)
                    hidden = hidden.clone()
                    hidden[:, -1, :] = steered_tensor
                    step_info["steered"] = True
                else:
                    step_info["steered"] = False

                return (hidden,) + output[1:] if is_tuple else hidden

            handle = target_layer.register_forward_hook(hook_fn)
            try:
                with torch.inference_mode():
                    outputs = self.extractor.model(input_ids=input_ids, use_cache=False)
            finally:
                handle.remove()

            token_probe_scores.append(step_info["score"])
            steered_tokens.append(step_info["steered"])

            logits = outputs.logits[0, -1, :]

            if temperature > 0.0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True)

            next_token_id = next_token.item()
            generated_token_ids.append(next_token_id)

            eos_id = getattr(self.extractor.tokenizer, "eos_token_id", None)
            if eos_id is not None and next_token_id == eos_id:
                break

            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=1)

        completion_text = self.extractor.tokenizer.decode(
            generated_token_ids, skip_special_tokens=True
        )

        num_steered = sum(steered_tokens)
        if self.verbose:
            print(
                f"[{task.task_id}] Generated {len(generated_token_ids)} tokens "
                f"({num_steered} steered) -> '{completion_text[:70]}'"
            )

        return {
            "task_id": task.task_id,
            "completion_text": completion_text,
            "generated_tokens": generated_token_ids,
            "probe_scores": token_probe_scores,
            "steered_tokens": steered_tokens,
            "num_tokens": len(generated_token_ids),
            "steered_steps": num_steered,
        }

    def run_eval_tasks(
        self,
        max_tasks: Optional[int] = None,
        max_steps: int = 128,
        temperature: float = 0.0,
    ) -> list[dict]:
        """Runs token-at-a-time evaluation across tasks in eval_set."""
        tasks = self.get_tasks(limit=max_tasks)
        results = []

        if self.verbose:
            print(f"Starting steered evaluation on {len(tasks)} tasks...")

        for task in tasks:
            res = self.run_task(task, max_steps=max_steps, temperature=temperature)
            results.append(res)

        self.shard_writer.close()
        summary_path = self.output_dir / "run_summary.parquet"
        save_run_summary(summary_path, results)
        return results
