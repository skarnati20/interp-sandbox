from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional


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


class BaseBenchmark(ABC):
    @abstractmethod
    def list_tasks(self, split: str = "train") -> list[TaskInstance]:
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
