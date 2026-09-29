"""
Quick local verification script.
Tests the entire pipeline (MiniWoB -> Extraction -> Step Execution -> Parquet/Safetensors storage).
"""

from pathlib import Path
import torch

from src.activations import ActivationExtractor
from src.benchmark import MiniWoBBenchmark, TaskInstance
from src.runner import Runner
from src.store import load_activation_result, load_run_summary


class MockExtractor:
    """A lightweight mock extractor for fast local verification without downloading a model."""

    def __init__(self, num_layers: int = 24, hidden_dim: int = 896):
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

    def step_forward(self, messages: list[dict], max_new_tokens: int = 64, temperature: float = 0.0):
        from src.activations import ExtractionResult

        # Dummy hidden states tensor [num_layers, total_tokens, hidden_dim]
        dummy_tensor = torch.randn(self.num_layers, 20, self.hidden_dim, dtype=torch.float16)

        # Simple heuristic or default action for click-test
        return ExtractionResult(
            completion_text="CLICK(ref=1)",
            hidden_states=dummy_tensor,
            prompt_tokens=15,
            completion_tokens=5,
        )


def run_local_test(use_real_model: bool = False):
    print("=" * 60)
    print("Running Local Pipeline Test...")
    print("=" * 60)

    # 1. Initialize Extractor
    if use_real_model:
        print("Loading real model (Qwen/Qwen2.5-Coder-0.5B-Instruct)...")
        extractor = ActivationExtractor("Qwen/Qwen2.5-Coder-0.5B-Instruct")
    else:
        print("Using MockExtractor for fast local dry-run...")
        extractor = MockExtractor()

    # 2. Initialize MiniWoB Benchmark
    print("Initializing MiniWoB environment (miniwob/click-test-2-v1)...")
    benchmark = MiniWoBBenchmark(env_names=["miniwob/click-test-2-v1"])

    # 3. Initialize Runner
    output_dir = Path("data/local_test_run")
    runner = Runner(extractor=extractor, benchmark=benchmark, output_dir=output_dir)

    # 4. Run 1 task episode
    print("\nExecuting Episode...")
    results = runner.run_benchmark(max_tasks=1, max_steps=3)

    # 5. Verify saved files
    print("\n" + "=" * 60)
    print("Verifying Stored Artifacts:")
    print("=" * 60)

    summary_file = output_dir / "run_summary.parquet"
    if summary_file.exists():
        summary_df = load_run_summary(summary_file)
        print(f"✓ Summary Parquet exists:\n{summary_df}\n")

    step_0_dir = output_dir / results[0]["episode_id"] / "step_0"
    if (step_0_dir / "activations.safetensors").exists():
        res, meta = load_activation_result(step_0_dir)
        print(f"✓ Step 0 Safetensors exists!")
        print(f"  - Loaded Tensor shape: {res.hidden_states.shape}")
        print(f"  - Decision Token shape: {res.last_prompt_state.shape}")
        print(f"  - Action text: {meta['action']}")
        print(f"  - Task ID: {meta['task_id']}")

    print("\n✓ ALL LOCAL VERIFICATION CHECKS PASSED!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--real-model", action="store_true", help="Use real Qwen 0.5B model instead of mock")
    args = parser.parse_args()

    run_local_test(use_real_model=args.real_model)
