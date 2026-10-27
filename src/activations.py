from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import re
from typing import Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class ExtractionResult:
    """
    Holds residual-stream hidden states and sequence metadata for a forward generation step.
    Provides extensible anchor resolution across pre-generation, post-generation,
    first-token, mean-pooled, and thought-boundary representations.
    """
    completion_text: str
    hidden_states: torch.Tensor  # Shape: [num_layers, total_seq_len, hidden_dim] (float16)
    prompt_tokens: int
    completion_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @classmethod
    def supported_anchors(cls) -> list[str]:
        """List of standard named anchor strategies."""
        return [
            "pre_gen",       # Last token of prompt before generation begins
            "post_gen",      # Last token of generated completion
            "first_gen",     # First token of generated completion
            "mean_gen",      # Mean-pooled state across all generated tokens
            "mean_prompt",   # Mean-pooled state across all prompt tokens
            "mean_all",      # Mean-pooled state across prompt + completion
            "thought_end",   # Boundary token where thinking ends and action begins
        ]

    def resolve_anchor(self, anchor: str) -> torch.Tensor:
        """
        Resolves a named anchor or integer offset to a [num_layers, hidden_dim] state tensor.
        """
        norm_anchor = anchor.strip().lower()

        match norm_anchor:
            case "pre_gen":
                return self.hidden_states[:, max(0, self.prompt_tokens - 1), :]

            case "post_gen":
                return self.hidden_states[:, -1, :]

            case "first_gen":
                idx = self.prompt_tokens if self.completion_tokens > 0 else max(0, self.prompt_tokens - 1)
                return self.hidden_states[:, min(idx, self.hidden_states.shape[1] - 1), :]

            case "mean_gen":
                if self.completion_tokens > 0:
                    return self.hidden_states[:, self.prompt_tokens :, :].mean(dim=1)
                return self.hidden_states[:, max(0, self.prompt_tokens - 1), :]

            case "mean_prompt":
                return self.hidden_states[:, : self.prompt_tokens, :].mean(dim=1)

            case "mean_all":
                return self.hidden_states.mean(dim=1)

            case "thought_end":
                return self.resolve_thought_boundary(norm_anchor)

            case _:
                try:
                    return self.hidden_states[:, int(norm_anchor), :]
                except ValueError:
                    raise ValueError(
                        f"Unsupported anchor '{anchor}'. Supported named anchors: {self.supported_anchors()} "
                        f"or relative integer index strings (e.g. '-1', '0')."
                    )

    def resolve_thought_boundary(self, anchor_type: str) -> torch.Tensor:
        """
        Locates the transition token between reasoning/thoughts and the executable command.
        """
        text = self.completion_text
        match = re.search(r"(?:think:?.*?\n|thought:?.*?\n|Action:\s*)", text, re.DOTALL | re.IGNORECASE)

        if match and self.completion_tokens > 1:
            # Estimate token boundary from character offset proportion
            char_boundary = match.end()
            ratio = min(1.0, max(0.0, char_boundary / max(1, len(text))))
            gen_idx = int(round(ratio * (self.completion_tokens - 1)))
            token_idx = min(self.prompt_tokens + gen_idx, self.hidden_states.shape[1] - 1)
            return self.hidden_states[:, token_idx, :]

        return self.resolve_anchor("post_gen")

    @property
    def pre_gen_state(self) -> torch.Tensor:
        """Hidden state at the pre-generation decision token."""
        return self.resolve_anchor("pre_gen")

    @property
    def post_gen_state(self) -> torch.Tensor:
        """Hidden state at the post-generation commitment token."""
        return self.resolve_anchor("post_gen")

    def get_anchor_state(
        self,
        anchor: str = "post_gen",
        layer_ids: Optional[list[int]] = None,
    ) -> torch.Tensor:
        """
        Extracts residual-stream activations at a specific anchor position and layer slice.
        Args:
            anchor: Any supported anchor name ('pre_gen', 'post_gen', 'first_gen',
                    'mean_gen', 'mean_prompt', 'mean_all', 'thought_end')
                    or relative integer offset.
            layer_ids: List of layer indices to select (None for all layers).
        Returns:
            Tensor of shape [len(layer_ids), hidden_dim]
        """
        state = self.resolve_anchor(anchor)
        if layer_ids is not None:
            indices = [min(max(0, l), state.shape[0] - 1) for l in layer_ids]
            return state[indices, :]
        return state


class ActivationExtractor:
    def __init__(
        self,
        model: str,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        torch_dtype: torch.dtype = torch.bfloat16,
    ):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForCausalLM.from_pretrained(
            model,
            torch_dtype=torch_dtype,
            device_map="auto" if device == "cuda" else None,
        )
        if device != "cuda":
            self.model.to(self.device)
        self.model.eval()

    def get_layers(self):
        """Returns the module list containing transformer decoder layers."""
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return self.model.model.layers
        elif hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            return self.model.transformer.h
        elif hasattr(self.model, "layers"):
            return self.model.layers
        raise AttributeError("Could not find transformer decoder layers on model.")

    @contextmanager
    def steer(
        self,
        layer_idx: int,
        vector: torch.Tensor | np.ndarray,
        coeff: float = 1.0,
    ):
        """
        Context manager that attaches a residual-stream steering hook to layer `layer_idx`,
        adding `coeff * vector` to hidden states during forward passes and generation.
        """
        layers = self.get_layers()
        target_layer = layers[min(max(0, layer_idx), len(layers) - 1)]

        if isinstance(vector, np.ndarray):
            vec_tensor = torch.from_numpy(vector).to(self.device)
        else:
            vec_tensor = vector.to(self.device)

        def hook_fn(module, input, output):
            v = vec_tensor.to(dtype=output[0].dtype if isinstance(output, tuple) else output.dtype)
            if isinstance(output, tuple):
                steered = output[0] + (coeff * v)
                return (steered,) + output[1:]
            else:
                return output + (coeff * v)

        handle = target_layer.register_forward_hook(hook_fn)
        try:
            yield
        finally:
            handle.remove()

    @torch.inference_mode()
    def step_forward(
        self,
        messages: list[dict],
        max_new_tokens: int = 512,
        temperature: float = 0.0,
    ) -> ExtractionResult:
        prompt_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.device)
        prompt_len = inputs.input_ids.shape[1]

        generate_kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0.0,
            "return_dict_in_generate": True,
            "output_hidden_states": True,
        }
        if temperature > 0.0:
            generate_kwargs["temperature"] = temperature

        gen_out = self.model.generate(**inputs, **generate_kwargs)

        completion_ids = gen_out.sequences[0, prompt_len:]
        completion_text = self.tokenizer.decode(completion_ids, skip_special_tokens=True)

        num_layers = len(gen_out.hidden_states[0]) - 1

        layer_tensors = []
        for l in range(1, num_layers + 1):
            prompt_layer = gen_out.hidden_states[0][l][0]  # [prompt_len, hidden_dim]
            gen_steps = [step[l][0] for step in gen_out.hidden_states[1:]]  # list of [1, hidden_dim]
            if gen_steps:
                full_seq = torch.cat([prompt_layer] + gen_steps, dim=0)
            else:
                full_seq = prompt_layer
            layer_tensors.append(full_seq.to(torch.float16).cpu())

        # Shape: [num_layers, total_seq_len, hidden_dim]
        full_hidden_states = torch.stack(layer_tensors)
        num_completion_hidden_states = full_hidden_states.shape[1] - prompt_len

        return ExtractionResult(
            completion_text=completion_text,
            hidden_states=full_hidden_states,
            prompt_tokens=prompt_len,
            completion_tokens=num_completion_hidden_states,
        )

    @torch.inference_mode()
    def step_forward_completion(
        self,
        messages: list[dict],
        completion_text: str,
    ) -> ExtractionResult:
        """
        Executes a single forward pass over pre-existing messages and completion (teacher-forcing),
        extracting residual stream activations without autoregressive generation.
        """
        prompt_text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prompt_inputs = self.tokenizer(prompt_text, return_tensors="pt")
        prompt_len = prompt_inputs.input_ids.shape[1]

        completion_inputs = self.tokenizer(
            completion_text, return_tensors="pt", add_special_tokens=False
        )
        completion_len = completion_inputs.input_ids.shape[1]

        full_input_ids = torch.cat(
            [prompt_inputs.input_ids, completion_inputs.input_ids], dim=1
        ).to(self.device)
        attention_mask = torch.ones_like(full_input_ids).to(self.device)

        out = self.model(
            input_ids=full_input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        num_layers = len(out.hidden_states) - 1
        layer_tensors = [
            out.hidden_states[l][0].to(torch.float16).cpu()
            for l in range(1, num_layers + 1)
        ]
        full_hidden_states = torch.stack(layer_tensors)

        return ExtractionResult(
            completion_text=completion_text,
            hidden_states=full_hidden_states,
            prompt_tokens=prompt_len,
            completion_tokens=completion_len,
        )
