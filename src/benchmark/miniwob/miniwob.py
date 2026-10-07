from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Optional

import gymnasium

try:
    from miniwob.action import ActionTypes
    ACTION_TYPE_INDEX = {at: i for i, at in enumerate(ActionTypes)}
except ImportError:
    try:
        from miniwob.constants import ACTION_TYPE_INDEX, ActionTypes
    except ImportError:
        ACTION_TYPE_INDEX = {}
        ActionTypes = None

from ..base import BaseBenchmark, StepObservation, TaskInstance


@dataclass
class MiniWoBDomFormatter:
    """Formats raw MiniWoB DOM element tuples into a structured text hierarchy."""

    max_elements: int = 60

    def format_dom(self, dom_elements: tuple[dict, ...]) -> str:
        if not dom_elements:
            return "(No visible DOM elements)"

        lines = []
        for i, el in enumerate(dom_elements[: self.max_elements]):
            tag = el.get("tag", "div").lower()
            ref = el.get("ref", i)
            text = el.get("text", "").strip()
            value = el.get("value", "").strip()

            attrs = []
            if text:
                attrs.append(f'text="{text}"')
            if value:
                attrs.append(f'value="{value}"')
            if el.get("id"):
                attrs.append(f'id="{el["id"]}"')
            if el.get("classes"):
                attrs.append(f'class="{el["classes"]}"')

            attr_str = " " + " ".join(attrs) if attrs else ""
            lines.append(f"[{ref}] <{tag}{attr_str}>")

        return "\n".join(lines)


class MiniWoBActionParser:
    """Parses textual agent actions into MiniWoB environment action dictionaries."""

    @staticmethod
    def parse(action_str: str) -> dict[str, Any]:
        action_str = action_str.strip()

        if ActionTypes is None:
            return {}

        # 1. CLICK(ref=4) or CLICK(4)
        click_match = re.search(r"CLICK\((?:ref=)?(\d+)\)", action_str, re.IGNORECASE)
        if click_match:
            ref = int(click_match.group(1))
            return {
                "action_type": ACTION_TYPE_INDEX.get(ActionTypes.CLICK_ELEMENT, 8),
                "ref": ref,
            }

        # 2. TYPE(ref=2, text="hello") or TYPE(2, "hello")
        type_match = re.search(
            r"TYPE\((?:ref=)?(\d+),\s*(?:text=)?[\"'](.*?)[\"']\)",
            action_str,
            re.IGNORECASE,
        )
        if type_match:
            ref = int(type_match.group(1))
            text = type_match.group(2)
            return {
                "action_type": ACTION_TYPE_INDEX.get(ActionTypes.TYPE_TEXT, 10),
                "ref": ref,
                "text": text,
            }

        # 3. PRESS(key="Enter") or PRESS("Enter")
        press_match = re.search(r"PRESS\((?:key=)?[\"'](.*?)[\"']\)", action_str, re.IGNORECASE)
        if press_match:
            key = press_match.group(1)
            return {"action_type": ACTION_TYPE_INDEX.get(ActionTypes.PRESS_KEY, 9), "key": key}

        # 4. SCROLL(dx=0, dy=100) or SCROLL_DOWN / SCROLL_UP
        scroll_match = re.search(
            r"SCROLL\((?:dx=)?(-?\d+),\s*(?:dy=)?(-?\d+)\)", action_str, re.IGNORECASE
        )
        if scroll_match:
            dx = int(scroll_match.group(1))
            dy = int(scroll_match.group(2))
            return {
                "action_type": ACTION_TYPE_INDEX.get(ActionTypes.MOVE_COORDS, 1),
                "coords": (dx, dy),
            }

        if re.search(r"SCROLL_DOWN", action_str, re.IGNORECASE):
            return {
                "action_type": ACTION_TYPE_INDEX.get(ActionTypes.SCROLL_DOWN_COORDS, 7),
                "coords": (0, 100),
            }
        if re.search(r"SCROLL_UP", action_str, re.IGNORECASE):
            return {
                "action_type": ACTION_TYPE_INDEX.get(ActionTypes.SCROLL_UP_COORDS, 6),
                "coords": (0, -100),
            }

        # 5. Fallback heuristic: single integer -> CLICK(ref=X)
        if action_str.isdigit():
            return {
                "action_type": ACTION_TYPE_INDEX.get(ActionTypes.CLICK_ELEMENT, 8),
                "ref": int(action_str),
            }

        # Default fallback to NO_OP
        return {"action_type": ACTION_TYPE_INDEX.get(ActionTypes.NONE, 0)}


class MiniWoBBenchmark(BaseBenchmark):
    """
    Standard MiniWoB++ benchmark adapter.
    Aligns with 'Doomed from the Start' (2026) 24 core MiniWoB environments.
    """

    # Official MiniWoB core subsets tested across literature
    CORE_ENVS = [
        "miniwob/click-test-2-v1",
        "miniwob/click-button-v1",
        "miniwob/click-button-sequence-v1",
        "miniwob/click-checkboxes-v1",
        "miniwob/click-collapsible-v1",
        "miniwob/click-color-v1",
        "miniwob/click-dialog-v1",
        "miniwob/click-link-v1",
        "miniwob/click-option-v1",
        "miniwob/click-pie-v1",
        "miniwob/click-scroll-list-v1",
        "miniwob/click-shades-v1",
        "miniwob/click-shape-v1",
        "miniwob/click-tab-v1",
        "miniwob/click-widget-v1",
        "miniwob/enter-date-v1",
        "miniwob/enter-password-v1",
        "miniwob/enter-text-v1",
        "miniwob/enter-text-dynamic-v1",
        "miniwob/enter-time-v1",
        "miniwob/focus-text-v1",
        "miniwob/identify-shape-v1",
        "miniwob/login-user-v1",
        "miniwob/social-media-v1",
    ]

    DEFAULT_SYSTEM_PROMPT = (
        "You are an expert autonomous web agent completing tasks in a web browser.\n"
        "Given the task instruction and visible DOM elements, output ONLY the next action.\n"
        "Supported actions:\n"
        "  - CLICK(ref=<int>): Click an interactive element by reference ID\n"
        "  - TYPE(ref=<int>, text=\"<str>\"): Type text into an input element\n"
        "  - PRESS(key=\"<str>\"): Press a keyboard key (e.g. \"Enter\")\n"
        "  - SCROLL(dx=<int>, dy=<int>): Scroll viewport\n"
        "Respond strictly with the single action command to execute next."
    )

    def __init__(
        self,
        env_names: Optional[list[str]] = None,
        all_envs: bool = False,
        dom_only: bool = True,
        num_tasks: Optional[int] = None,
        seeds: Optional[list[int]] = None,
        headless: bool = True,
    ):
        if env_names is not None:
            self.env_names = env_names
        elif all_envs:
            all_registered = [env_id for env_id in gymnasium.envs.registry.keys() if "miniwob/" in env_id]
            self.env_names = sorted(all_registered) if all_registered else self.CORE_ENVS
        else:
            self.env_names = self.CORE_ENVS

        self.num_tasks = num_tasks
        self.seeds = seeds or [42]
        self.headless = headless
        self.formatter = MiniWoBDomFormatter()
        self.parser = MiniWoBActionParser()

        self._active_env: Optional[gymnasium.Env] = None
        self._active_task: Optional[TaskInstance] = None
        self._last_raw_obs: Optional[dict] = None

    def list_tasks(self, split: str = "train") -> list[TaskInstance]:
        tasks = []
        task_idx = 0
        for seed in self.seeds:
            for env_name in self.env_names:
                if self.num_tasks is not None and len(tasks) >= self.num_tasks:
                    break
                task_id = f"{env_name}_seed_{seed}"
                tasks.append(
                    TaskInstance(
                        task_id=task_id,
                        instruction=f"Complete the task in {env_name}",
                        system_prompt=self.DEFAULT_SYSTEM_PROMPT,
                        info={"env_name": env_name, "seed": seed, "task_idx": task_idx},
                    )
                )
                task_idx += 1
            if self.num_tasks is not None and len(tasks) >= self.num_tasks:
                break
        return tasks

    def reset(self, task: TaskInstance, seed: Optional[int] = None) -> StepObservation:
        self.close()

        env_name = task.info.get("env_name", self.env_names[0])
        env_seed = seed or task.info.get("seed", 42)

        # Initialize environment with render_mode if needed
        self._active_env = gymnasium.make(env_name)
        obs, info = self._active_env.reset(seed=env_seed)
        self._last_raw_obs = obs
        self._active_task = task

        utterance = obs.get("utterance", task.instruction)
        dom_elements = obs.get("dom_elements", ())
        formatted_dom = self.formatter.format_dom(dom_elements)

        prompt_obs = f"Goal: {utterance}\n\nVisible DOM Elements:\n{formatted_dom}"

        return StepObservation(
            observation_text=prompt_obs,
            step_reward=0.0,
            is_done=False,
            info={"utterance": utterance, "raw_obs": obs},
        )

    def step(self, action_str: str) -> StepObservation:
        if self._active_env is None:
            raise RuntimeError("Environment not initialized. Call reset() before step().")

        parsed_action = self.parser.parse(action_str)
        obs, reward, terminated, truncated, info = self._active_env.step(parsed_action)
        self._last_raw_obs = obs

        is_done = terminated or truncated
        utterance = obs.get("utterance", "")
        dom_elements = obs.get("dom_elements", ())
        formatted_dom = self.formatter.format_dom(dom_elements)

        prompt_obs = f"Goal: {utterance}\n\nVisible DOM Elements:\n{formatted_dom}"

        return StepObservation(
            observation_text=prompt_obs,
            step_reward=float(reward),
            is_done=is_done,
            info={"reward": reward, "terminated": terminated, "truncated": truncated, "info": info},
        )

    def evaluate(self, task: TaskInstance) -> bool:
        if self._last_raw_obs is None:
            return False
        last_reward = self._last_raw_obs.get("reward", 0.0) if isinstance(self._last_raw_obs, dict) else 0.0
        return last_reward > 0.0

    def close(self) -> None:
        if self._active_env is not None:
            try:
                self._active_env.close()
            except Exception:
                pass
            self._active_env = None
        self._active_task = None
        self._last_raw_obs = None
