from dataclasses import dataclass
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class ExtractionResult:
    completion_text: str
    hidden_states: torch.Tensor
    prompt_tokens: int
    completion_tokens: int

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def last_prompt_state(self) -> torch.Tensor:
        """Hidden state at the decision token (last prompt token) across all layers.
        Shape: [num_layers, hidden_dim]
        """
        return self.hidden_states[:, self.prompt_tokens - 1, :]

    @property
    def prompt_states(self) -> torch.Tensor:
        """All prompt token hidden states across all layers.
        Shape: [num_layers, prompt_len, hidden_dim]
        """
        return self.hidden_states[:, : self.prompt_tokens, :]

    @property
    def completion_states(self) -> torch.Tensor:
        """All generated token hidden states across all layers.
        Shape: [num_layers, completion_tokens, hidden_dim]
        """
        return self.hidden_states[:, self.prompt_tokens :, :]


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
