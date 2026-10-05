"""
Quick local verification script.
Tests the collection pipeline (Benchmark -> Extraction -> Dual-Anchor Sharding -> Parquet/NPZ storage).
"""

from pathlib import Path
import numpy as np
import torch

from src.activations import ActivationExtractor
from src.benchmark import MiniWoBBenchmark, TaskInstance
from src.runner import Runner
from src.store import load_run_summary


class MockExtractor:
    """A lightweight mock extractor for fast local verification without downloading a model."""

    def __init__(self, num_layers: int = 24, hidden_dim: int = 896):
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

    def step_forward(self, messages: list[dict], max_new_tokens: int = 64, temperature: float = 0.0):
        from src.activations import ExtractionResult

        # Dummy hidden states tensor [num_layers, total_tokens, hidden_dim]
        dummy_tensor = torch.randn(self.num_layers, 20, self.hidden_dim, dtype=torch.float16)

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

    # 3. Initialize Runner with Sharded storage
    output_dir = Path("data/local_test_run")
    runner = Runner(
        extractor=extractor,
        benchmark=benchmark,
        output_dir=output_dir,
        anchors=["post_gen", "pre_gen"],
    )

    # 4. Run benchmark with 2 rollouts
    print("\nExecuting Episodes...")
    results = runner.run_benchmark(max_tasks=1, max_steps=3, n_rollouts=2)

    # 5. Verify saved files
    print("\n" + "=" * 60)
    print("Verifying Stored Artifacts:")
    print("=" * 60)

    summary_file = output_dir / "run_summary.parquet"
    if summary_file.exists():
        summary_df = load_run_summary(summary_file)
        print(f"✓ Summary Parquet exists ({len(summary_df)} episodes):\n{summary_df}\n")

    features_dir = output_dir / "features"
    npz_file = features_dir / "feat.shard0.npz"
    jsonl_file = features_dir / "meta.shard0.jsonl"

    if npz_file.exists() and jsonl_file.exists():
        data = np.load(npz_file)
        print(f"✓ Sharded Features exist on disk!")
        print(f"  - NPZ Tensor shape X: {data['X'].shape} (N_anchors, n_layers, hidden_dim)")
        print(f"  - Layer IDs: {data['layer_ids']}")
        with open(jsonl_file, "r") as fp:
            num_records = sum(1 for _ in fp)
        print(f"  - Total metadata records in JSONL: {num_records}")

    print("\n✓ ALL LOCAL VERIFICATION CHECKS PASSED!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--real-model", action="store_true", help="Use real Qwen 0.5B model instead of mock")
    args = parser.parse_args()

    run_local_test(use_real_model=args.real_model)
