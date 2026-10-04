from pathlib import Path
from typing import Any, Optional
import numpy as np
import pandas as pd
from safetensors.torch import load_file, save_file
import torch

from src.activations import ExtractionResult


def save_activation_result(
    dir_path: str | Path,
    result: ExtractionResult,
    extra_metadata: Optional[dict[str, Any]] = None,
    mode: str = "decision",
) -> tuple[Path, Path]:
    """
    Persists activations and step metadata to disk.

    Args:
        dir_path: Directory where artifacts are stored.
        result: ExtractionResult containing tensors and completion metadata.
        extra_metadata: Additional dict metadata to include in metadata.parquet.
        mode: "decision" (default, saves only [num_layers, hidden_dim] ~200KB/step)
              or "all" (saves full 3D tensor [num_layers, seq_len, hidden_dim] ~200MB/step).
    """
    save_dir = Path(dir_path)
    save_dir.mkdir(parents=True, exist_ok=True)

    activations_path = save_dir / "activations.safetensors"

    if mode == "decision":
        # Save only the decision token across all layers [num_layers, hidden_dim]
        tensor_dict = {"decision_state": result.last_prompt_state.contiguous()}
    elif mode == "all":
        # Save the full 3D tensor [num_layers, seq_len, hidden_dim]
        tensor_dict = {"hidden_states": result.hidden_states.contiguous()}
    else:
        raise ValueError(f"Unknown save mode: {mode}. Choose 'decision' or 'all'.")

    save_file(tensor_dict, activations_path)

    # Prepare and save metadata
    meta = {
        "completion_text": result.completion_text,
        "prompt_tokens": result.prompt_tokens,
        "completion_tokens": result.completion_tokens,
        "save_mode": mode,
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
    """
    Loads saved activations and step metadata from disk.
    Supports both 'decision' and 'all' save modes transparently.
    """
    load_dir = Path(dir_path)
    activations_path = load_dir / "activations.safetensors"
    metadata_path = load_dir / "metadata.parquet"

    if not activations_path.exists() or not metadata_path.exists():
        raise FileNotFoundError(f"Missing activation or metadata file in {load_dir}")

    tensors = load_file(activations_path)
    df = pd.read_parquet(metadata_path)
    meta_dict = df.to_dict(orient="records")[0]

    if "decision_state" in tensors:
        # Reshape [num_layers, hidden_dim] -> [num_layers, 1, hidden_dim]
        hidden_states = tensors["decision_state"].unsqueeze(1)
        prompt_tokens = 1
        completion_tokens = 0
    else:
        hidden_states = tensors["hidden_states"]
        prompt_tokens = int(meta_dict.get("prompt_tokens", 1))
        completion_tokens = int(meta_dict.get("completion_tokens", 0))

    result = ExtractionResult(
        completion_text=str(meta_dict["completion_text"]),
        hidden_states=hidden_states,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
    )

    return result, meta_dict


def save_run_summary(file_path: str | Path, records: list[dict[str, Any]]) -> Path:
    """
    Saves a list of episode result dicts to a single Parquet file.
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(records)
    df.to_parquet(path, index=False)
    return path


def load_run_summary(file_path: str | Path) -> pd.DataFrame:
    """
    Loads a run summary Parquet file into a pandas DataFrame.
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Summary file not found: {path}")
    return pd.read_parquet(path)


def load_probing_dataset(
    run_dir: str | Path,
    step_idx: int = 0,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Loads all decision-token activations and labels for a specific step across all episodes.

    Returns:
      - X: np.ndarray of shape [num_episodes, num_layers, hidden_dim]
      - y: np.ndarray of shape [num_episodes] (1 = success, 0 = failure)
      - summary_df: pd.DataFrame of matched episode records
    """
    run_path = Path(run_dir)
    summary_df = load_run_summary(run_path / "run_summary.parquet")

    X_list = []
    y_list = []
    valid_rows = []

    for _, row in summary_df.iterrows():
        ep_id = row["episode_id"]
        step_dir = run_path / ep_id / f"step_{step_idx}"

        try:
            result, meta = load_activation_result(step_dir)
            # Use last_prompt_state property from ExtractionResult [num_layers, hidden_dim]
            decision_tensor = result.last_prompt_state.numpy().astype(np.float32)
            X_list.append(decision_tensor)
            y_list.append(1 if row["is_success"] else 0)
            valid_rows.append(row)
        except FileNotFoundError:
            # Episode ended before this step_idx
            continue

    if not X_list:
        raise ValueError(f"No step_{step_idx} data found in {run_path}")

    X = np.stack(X_list, axis=0)  # [N, num_layers, hidden_dim]
    y = np.array(y_list, dtype=np.int64)  # [N]
    matched_df = pd.DataFrame(valid_rows).reset_index(drop=True)

    return X, y, matched_df
