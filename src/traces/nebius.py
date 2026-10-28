from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, List, Optional
from huggingface_hub import snapshot_download
import pyarrow.parquet as pq

from .base import BaseTraceAdapter, Message, TurnStep, Trajectory


class NebiusSWEAgentTraces(BaseTraceAdapter):
    """
    Adapter for Nebius SWE-agent trajectories.
    Supports standard sequential extraction as well as smart balanced sampling:
      - Filters tasks to 'divergent tasks' (tasks that have both success and failure outcomes).
      - Samples a balanced 50/50 ratio of successes and failures across diverse repository tasks.
    """

    def __init__(
        self,
        dataset_name_or_path: str = "nebius/SWE-agent-trajectories",
        cache_dir: Optional[str | Path] = "/tmp/swe_agent_trajectories",
        allow_patterns: Optional[List[str]] = None,
        balanced: bool = True,
        divergent_only: bool = True,
        max_trajectories: Optional[int] = None,
        seed: int = 42,
    ):
        super().__init__(dataset_name_or_path=dataset_name_or_path)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.allow_patterns = allow_patterns or ["data/*.parquet", "README.md"]
        self.balanced = balanced
        self.divergent_only = divergent_only
        self.max_trajectories = max_trajectories
        self.seed = seed

    def load_smart_dataset(self, parquet_files: list[Path]) -> Iterator[dict]:
        import numpy as np
        import pandas as pd

        # 1. Read index columns ["instance_id", "target"] across parquet files
        file_dfs = []
        for f_idx, file_path in enumerate(parquet_files):
            pf = pq.ParquetFile(file_path)
            sub_df = pf.read(columns=["instance_id", "target"]).to_pandas()
            sub_df["file_idx"] = f_idx
            sub_df["local_row_idx"] = np.arange(len(sub_df))
            file_dfs.append(sub_df)
            # If we have enough candidate rows to fulfill max_trajectories, stop reading index
            if self.max_trajectories and sum(len(df) for df in file_dfs) >= max(3000, self.max_trajectories * 3):
                break

        full_index = pd.concat(file_dfs, ignore_index=True)

        # 2. Filter to divergent tasks if requested
        if self.divergent_only:
            task_stats = full_index.groupby("instance_id")["target"].agg(["count", "sum"])
            divergent_tasks = set(task_stats[(task_stats["sum"] > 0) & (task_stats["sum"] < task_stats["count"])].index)
            filtered_index = full_index[full_index["instance_id"].isin(divergent_tasks)]
        else:
            filtered_index = full_index

        pos_df = filtered_index[filtered_index["target"] == True]
        neg_df = filtered_index[filtered_index["target"] == False]

        target_count = self.max_trajectories if self.max_trajectories else len(filtered_index)
        half = target_count // 2 if self.balanced else target_count

        n_pos = min(half, len(pos_df))
        n_neg = min(half, len(neg_df)) if self.balanced else min(target_count - n_pos, len(neg_df))

        sampled_pos = pos_df.sample(n=n_pos, random_state=self.seed)
        sampled_neg = neg_df.sample(n=n_neg, random_state=self.seed)
        selected_index = pd.concat([sampled_pos, sampled_neg]).sort_values(["file_idx", "local_row_idx"])

        # 3. Stream selected full records from the parquet files
        for f_idx, group in selected_index.groupby("file_idx"):
            target_rows = set(group["local_row_idx"])
            pf = pq.ParquetFile(parquet_files[f_idx])
            curr_row = 0
            for batch in pf.iter_batches():
                batch_len = len(batch)
                batch_rows = set(range(curr_row, curr_row + batch_len))
                if batch_rows & target_rows:
                    batch_pylist = batch.to_pylist()
                    for offset, rec in enumerate(batch_pylist):
                        if (curr_row + offset) in target_rows:
                            yield rec
                curr_row += batch_len

    def load_raw_dataset(self) -> Iterator[dict]:
        local_path = Path(self.dataset_name_or_path)
        if local_path.exists():
            download_dir = local_path
        elif self.cache_dir and self.cache_dir.exists() and any(self.cache_dir.glob("data/*.parquet")):
            download_dir = self.cache_dir
        else:
            download_dir = snapshot_download(
                repo_id=self.dataset_name_or_path,
                repo_type="dataset",
                local_dir=str(self.cache_dir) if self.cache_dir else None,
                allow_patterns=self.allow_patterns,
            )

        path = Path(download_dir)
        parquet_files = sorted(path.glob("data/*.parquet"))
        if not parquet_files:
            parquet_files = sorted(path.glob("*.parquet"))
        if not parquet_files:
            parquet_files = sorted(path.glob("**/*.parquet"))

        # Default: Smart balanced sampling
        if self.balanced or self.divergent_only:
            yield from self.load_smart_dataset(parquet_files)
            return

        # Sequential raw loading
        for file_path in parquet_files:
            pf = pq.ParquetFile(file_path)
            for batch in pf.iter_batches():
                for record in batch.to_pylist():
                    yield record

    def parse_record(self, raw_record: dict) -> Trajectory:
        task_id = raw_record.get("instance_id", "")
        label = raw_record.get("target")

        trajectory_data = raw_record.get("trajectory", [])
        if isinstance(trajectory_data, str):
            trajectory_data = json.loads(trajectory_data)

        history: List[Message] = []
        turns: List[TurnStep] = []

        for item in trajectory_data:
            raw_role = (item.get("role") or "").lower().strip()
            if raw_role in ("ai", "assistant"):
                role = "assistant"
            elif raw_role == "system":
                role = "system"
            elif raw_role in ("user", "human"):
                role = "user"
            else:
                role = raw_role or "user"

            content = item.get("text") or item.get("content") or item.get("system_prompt") or ""
            msg = Message(role=role, content=content)
            history.append(msg)

            if role == "assistant":
                turn = TurnStep(
                    turn_idx=len(turns),
                    messages=list(history),
                )
                turns.append(turn)

        metadata = {
            "model_name": raw_record.get("model_name"),
            "exit_status": raw_record.get("exit_status"),
            "generated_patch": raw_record.get("generated_patch"),
            "eval_logs": raw_record.get("eval_logs"),
        }
        metadata = {k: v for k, v in metadata.items() if v is not None}

        return Trajectory(
            task_id=task_id,
            source_dataset=self.dataset_name_or_path,
            label=label,
            turns=turns,
            metadata=metadata,
        )
