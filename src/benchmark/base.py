from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

from src.splits import TaskRecord


@dataclass
class StepObservation:
    observation_text: str
    step_reward: float = 0.0
    is_done: bool = False
    info: dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskInstance:
    task_id: str
    instruction: str
    system_prompt: str
    info: dict[str, Any] = field(default_factory=dict)
    source_split: Optional[str] = None


class BaseBenchmark(ABC):
    def collection_settings(self) -> dict:
        """Adapter-specific execution settings beyond source provenance."""
        return {}

    def get_group_key(self, task: TaskInstance, grouping: str = "task") -> str:
        if grouping != "task":
            raise ValueError(f"Unsupported grouping mode: {grouping}")
        return task.info.get("raw_task_id", task.task_id)

    def task_record(self, task: TaskInstance, grouping: str = "task") -> TaskRecord:
        return TaskRecord(
            namespace=self.source_namespace,
            task_id=task.task_id,
            source_split=task.source_split,
            provenance=self.source_provenance,
            group_key=self.get_group_key(task, grouping),
            source_task_id=task.info.get("raw_task_id", task.task_id),
        )

    def task_catalog(self, grouping: str = "task") -> list[TaskRecord]:
        return [self.task_record(task, grouping) for task in self.list_tasks()]

    @staticmethod
    def select_source_tasks(tasks: list[TaskInstance], split: Optional[str]) -> list[TaskInstance]:
        if split is None:
            return tasks
        if split not in {task.source_split for task in tasks}:
            raise ValueError(f"Official source partition does not exist: {split}")
        return [task for task in tasks if task.source_split == split]

    @abstractmethod
    def list_tasks(self, split: Optional[str] = None) -> list[TaskInstance]:
        pass

    @abstractmethod
    def reset(self, task: TaskInstance, seed: Optional[int] = None) -> StepObservation:
        pass

    @abstractmethod
    def step(self, action_str: str) -> StepObservation:
        pass

    @abstractmethod
    def evaluate(self, task: TaskInstance) -> bool:
        pass

    @abstractmethod
    def close(self) -> None:
        pass
