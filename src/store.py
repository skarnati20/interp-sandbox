from pathlib import Path
from typing import Any, Optional
import pandas as pd
from safetensors.torch import load_file, save_file
import torch

from src.activations import ExtractionResult


def save_activation_result(
    dir_path: str | Path,
    result: ExtractionResult,
    extra_metadata: Optional[dict[str, Any]] = None,
) -> tuple[Path, Path]:
    save_dir = Path(dir_path)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save tensor via safetensors
    activations_path = save_dir / "activations.safetensors"
    save_file({"hidden_states": result.hidden_states.contiguous()}, activations_path)

    # 2. Prepare and save metadata
    meta = {
        "completion_text": result.completion_text,
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "num_layers": result.hidden_states.shape[0],
        "seq_len": result.hidden_states.shape[1],
        "hidden_dim": result.hidden_states.shape[2],
    }
    if extra_metadata:
        meta.update(extra_metadata)

    metadata_path = save_dir / "metadata.parquet"
    df = pd.DataFrame([meta])
    df.to_parquet(metadata_path, index=False)

    return activations_path, metadata_path


def load_activation_result(dir_path: str | Path) -> tuple[ExtractionResult, dict[str, Any]]:
    load_dir = Path(dir_path)
    activations_path = load_dir / "activations.safetensors"
    metadata_path = load_dir / "metadata.parquet"

    if not activations_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(f"Missing activation or metadata file in {load_dir}")

    # 1. Load tensor
    tensors = load_file(activations_path)
    hidden_states = tensors["hidden_states"]

    # 2. Load metadata
    df = pd.read_parquet(metadata_path)
    meta_dict = df.to_dict(orient="records")[0]

    result = ExtractionResult(
        completion_text=str(meta_dict["completion_text"]),
        hidden_states=hidden_states,
        prompt_tokens=int(meta_dict["prompt_tokens"]),
        completion_tokens=int(meta_dict["completion_tokens"]),
    )

    return result, meta_dict


def save_run_summary(file_path: str | Path, records: list[dict[str, Any]]) -> Path:
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records)
    df.to_parquet(path, index=False)
    return path


def load_run_summary(file_path: str | Path) -> pd.DataFrame:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Summary file not found: {path}")
    return pd.read_parquet(path)
