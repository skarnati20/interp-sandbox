from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator, List, Optional
from huggingface_hub import snapshot_download
import pyarrow.parquet as pq

from .base import BaseTraceAdapter, Message, TurnStep, Trajectory


class NebiusSWEAgentTraces(BaseTraceAdapter):
    def __init__(
        self,
        dataset_name_or_path: str = "nebius/SWE-agent-trajectories",
        cache_dir: Optional[str | Path] = "/tmp/swe_agent_trajectories",
        allow_patterns: Optional[List[str]] = None,
    ):
        super().__init__(dataset_name_or_path=dataset_name_or_path)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.allow_patterns = allow_patterns or ["data/*.parquet", "README.md"]

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
