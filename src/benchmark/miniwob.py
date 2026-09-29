import re
from typing import Any, Optional
import gymnasium


import miniwob
from miniwob.action import ActionTypes

gymnasium.register_envs(miniwob)

from .base import BaseBenchmark, StepObservation, TaskInstance


class MiniWoBDomFormatter:
    """Formats raw MiniWoB DOM element tuples into clean text for LLM prompts."""

    @staticmethod
    def format_dom(dom_elements: tuple[dict[str, Any], ...]) -> str:
        if not dom_elements:
            return "(No visible DOM elements)"

        lines = []
        for elem in dom_elements:
            if not elem.get("visible", True):
                continue

            ref = elem.get("ref", "")
            tag = elem.get("tag", "div")
            text = elem.get("text", "").strip()
            value = elem.get("value", "")
            classes = elem.get("classes", "")
            elem_id = elem.get("id", "")

            attr_parts = []
            if elem_id:
                attr_parts.append(f'id="{elem_id}"')
            if classes:
                attr_parts.append(f'class="{classes}"')
            if value:
                attr_parts.append(f'value="{value}"')

            attrs = " " + " ".join(attr_parts) if attr_parts else ""
            lines.append(f"[{ref}] <{tag}{attrs}>{text}</{tag}>")

        return "\n".join(lines) if lines else "(No visible DOM elements)"


class MiniWoBActionParser:
    """Parses LLM textual action strings into MiniWoB Action objects."""

    @staticmethod
    def parse_action(action_str: str, env: Any) -> dict[str, Any]:
        action_str = action_str.strip()

        # 1. Match CLICK(ref=1) or CLICK(1)
        click_match = re.search(r"CLICK\(.*?ref\s*=\s*(\d+).*?\)|CLICK\((\d+)\)", action_str, re.IGNORECASE)
        if click_match:
            ref_id = int(click_match.group(1) or click_match.group(2))
            try:
                return env.unwrapped.create_action(ActionTypes.CLICK_ELEMENT, ref=ref_id)
            except Exception:
                pass

        # 2. Match TYPE(ref=1, text="hello") or TYPE(1, "hello")
        type_match = re.search(
            r'TYPE\(.*?ref\s*=\s*(\d+).*?text\s*=\s*["\'](.*?)["\'].*?\)|TYPE\((\d+),\s*["\'](.*?)["\']\)',
            action_str,
            re.IGNORECASE,
        )
        if type_match:
            ref_id = int(type_match.group(1) or type_match.group(3))
            text_val = type_match.group(2) or type_match.group(4)
            try:
                return env.unwrapped.create_action(ActionTypes.TYPE_TEXT, ref=ref_id, text=text_val)
            except Exception:
                pass

        # 3. Match PRESS_ENTER()
        if "PRESS_ENTER" in action_str.upper():
            try:
                return env.unwrapped.create_action(ActionTypes.PRESS_ENTER)
            except Exception:
                pass

        # Fallback: NONE / No-op
        try:
            return env.unwrapped.create_action(ActionTypes.NONE)
        except Exception:
            return {"action_type": 0}


class MiniWoBBenchmark(BaseBenchmark):
    """Benchmark adapter for Farama MiniWoB++ web environments."""

    DEFAULT_ENV_NAMES = [
        "miniwob/click-test-2-v1",
        "miniwob/login-user-v1",
        "miniwob/enter-text-v1",
        "miniwob/choose-date-v1",
        "miniwob/click-dialog-v1",
    ]

    DEFAULT_SYSTEM_PROMPT = (
        "You are an intelligent web navigation agent. Interact with the web page by issuing one of the following commands:\n"
        "  - CLICK(ref=ID) : Clicks the element with the given ref ID.\n"
        "  - TYPE(ref=ID, text=\"STRING\") : Types string into the element with given ref ID.\n"
        "  - PRESS_ENTER() : Simulates pressing Enter.\n"
        "Respond ONLY with the exact command."
    )

    def __init__(
        self,
        env_names: Optional[list[str]] = None,
        wait_ms: int = 150,
        render_mode: Optional[str] = None,
    ):
        self.env_names = env_names or self.DEFAULT_ENV_NAMES
        self.wait_ms = wait_ms
        self.render_mode = render_mode
        self.current_env = None
        self.current_task_id = None
        self.cumulative_reward = 0.0
        self.formatter = MiniWoBDomFormatter()
        self.parser = MiniWoBActionParser()

    def list_tasks(self, split: str = "train") -> list[TaskInstance]:
        tasks = []
        for name in self.env_names:
            tasks.append(
                TaskInstance(
                    task_id=name,
                    instruction="Accomplish the goal indicated in the web page utterance.",
                    system_prompt=self.DEFAULT_SYSTEM_PROMPT,
                    info={"env_name": name},
                )
            )
        return tasks

    def reset(self, task: TaskInstance, seed: Optional[int] = None) -> StepObservation:
        self.close()

        self.current_task_id = task.task_id
        self.current_env = gymnasium.make(
            task.task_id,
            render_mode=self.render_mode,
            wait_ms=self.wait_ms,
        )
        obs, info = self.current_env.reset(seed=seed)
        self.cumulative_reward = 0.0

        utterance = obs.get("utterance", "")
        dom_text = self.formatter.format_dom(obs.get("dom_elements", ()))

        obs_text = f"Goal: {utterance}\n\nVisible Elements:\n{dom_text}"
        return StepObservation(
            observation_text=obs_text,
            step_reward=0.0,
            is_done=False,
            info=info,
        )

    def step(self, action_str: str) -> StepObservation:
        if self.current_env is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")

        action = self.parser.parse_action(action_str, self.current_env)
        obs, reward, terminated, truncated, info = self.current_env.step(action)

        self.cumulative_reward += float(reward)
        is_done = bool(terminated or truncated)

        dom_text = self.formatter.format_dom(obs.get("dom_elements", ()))
        utterance = obs.get("utterance", "")
        obs_text = f"Goal: {utterance}\n\nUpdated Elements:\n{dom_text}"

        return StepObservation(
            observation_text=obs_text,
            step_reward=float(reward),
            is_done=is_done,
            info=info,
        )

    def evaluate(self, task: TaskInstance) -> bool:
        return self.cumulative_reward > 0.0

    def close(self) -> None:
        if self.current_env is not None:
            try:
                self.current_env.close()
            except Exception:
                pass
            self.current_env = None
