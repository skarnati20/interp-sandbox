"""Load official source partitions at a resolved Hugging Face dataset commit."""

from typing import Optional


def load_source_dataset(repo_id: str, config: Optional[str] = None, revision: Optional[str] = None,
                        token: Optional[str] = None, source_split: Optional[str] = None):
    from datasets import DatasetDict, load_dataset
    from huggingface_hub import HfApi

    resolved_revision = HfApi().dataset_info(repo_id, revision=revision, token=token).sha
    if not resolved_revision:
        raise ValueError(f"Could not resolve dataset revision for {repo_id}")
    loading = {"name": config, "revision": resolved_revision, "token": token}
    if source_split is None:
        partitions = load_dataset(repo_id, **loading)
    else:
        # Release partitions may contain alternate versions of the same tasks.
        partitions = DatasetDict({source_split: load_dataset(repo_id, split=source_split, **loading)})
    configs = {partition.info.config_name for partition in partitions.values()}
    if len(configs) != 1:
        raise ValueError("Expected one dataset configuration across official partitions")
    return partitions, {
        "kind": "huggingface", "dataset_id": repo_id,
        "config": configs.pop(), "revision": resolved_revision,
    }
