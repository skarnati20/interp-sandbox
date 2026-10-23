from __future__ import annotations

from pathlib import Path
from typing import Optional
from huggingface_hub import snapshot_download


def download_activations(
    repo_id: str,
    repo_type: str = "dataset",
    allow_patterns: Optional[list[str]] = None,
    local_dir: Optional[str | Path] = None,
    token: Optional[str | bool] = None,
    revision: Optional[str] = None,
    cache_dir: Optional[str | Path] = None,
    local_files_only: bool = False,
) -> Path:
    """
    Downloads activation shards and metadata from a Hugging Face repository.

    Args:
        repo_id: Hugging Face repository ID (e.g., 'username/dataset-name').
        repo_type: Type of the repository (default: 'dataset').
        allow_patterns: File patterns to download (defaults to .npz and .jsonl shards).
        local_dir: Local destination folder. If None, uses default HF cache.
        token: Optional HF authentication token.

    Returns:
        Path to the local directory where the files are stored.
    """
    if allow_patterns is None:
        allow_patterns = ["*feat.shard*.npz", "*meta.shard*.jsonl"]

    download_dir = snapshot_download(
        repo_id=repo_id,
        repo_type=repo_type,
        allow_patterns=allow_patterns,
        local_dir=str(local_dir) if local_dir is not None else None,
        token=token,
        revision=revision,
        cache_dir=cache_dir,
        local_files_only=local_files_only,
    )
    return Path(download_dir)
