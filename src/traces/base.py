from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional, List, Iterator


@dataclass
class Message:
    role: str
    content: str
    tool_call_id: Optional[str] = None

    def to_dict(self) -> dict[str, str]:
        d = {"role": self.role, "content": self.content}
        if self.tool_call_id is not None:
            d["tool_call_id"] = self.tool_call_id
        return d


@dataclass
class TurnStep:
    turn_idx: int
    messages: List[Message]

    def to_chat_messages(self) -> List[dict[str, str]]:
        return [m.to_dict() for m in self.messages]

    @property
    def action_text(self) -> Optional[str]:
        for msg in reversed(self.messages):
            if msg.role == "assistant":
                return msg.content
        return None


@dataclass
class Trajectory:
    task_id: str
    source_dataset: str
    label: Any
    turns: List[TurnStep]
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def total_turns(self) -> int:
        return len(self.turns)

    @property
    def is_success(self) -> int:
        if isinstance(self.label, bool):
            return int(self.label)
        if isinstance(self.label, (int, float)):
            return int(self.label)
        return 1 if str(self.label).strip().lower() in ("resolved", "pass", "success", "true", "1") else 0


class BaseTraceAdapter(ABC):
    def __init__(self, dataset_name_or_path: str):
        self.dataset_name_or_path = dataset_name_or_path

    @abstractmethod
    def load_raw_dataset(self) -> Iterator[dict] | List[dict]:
        pass

    @abstractmethod
    def parse_record(self, raw_record: dict) -> Trajectory:
        pass

    def iter_trajectories(self, max_trajectories: Optional[int] = None) -> Iterator[Trajectory]:
        count = 0
        for raw in self.load_raw_dataset():
            try:
                traj = self.parse_record(raw)
                yield traj
                count += 1
                if max_trajectories and count >= max_trajectories:
                    break
            except Exception as e:
                print(f"Skipping corrupt record in {self.dataset_name_or_path}: {e}")
