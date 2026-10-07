from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Optional

import gymnasium
import numpy as np

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

    ALLOWED_KEYS = (
        "<Enter>", "<PageUp>", "<PageDown>", "<Backspace>", "<Delete>", "<Tab>", "<Space>",
        "<ArrowUp>", "<ArrowRight>", "<ArrowDown>", "<ArrowLeft>", "[", "]", "-", "=", ";",
        '"', "\\", ",", ".", "/", "`", "1", "2", "3", "4", "5", "6", "7", "8", "9", "0",
        "<Numpad0>", "<Numpad1>", "<Numpad2>", "<Numpad3>", "<Numpad4>", "<Numpad5>",
        "<Numpad6>", "<Numpad7>", "<Numpad8>", "<Numpad9>", "<NumpadAdd>", "<NumpadMultiply>",
        "<NumpadSubtract>", "<NumpadDivide>", "<NumpadDecimal>", "<NumpadEnter>",
        "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p",
        "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "C-a", "C-c", "C-x", "C-v",
        "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P",
        "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"
    )

    KEY_TO_INDEX: dict[str, int] = {}
    for idx, key_str in enumerate(ALLOWED_KEYS):
        KEY_TO_INDEX[key_str] = idx
        KEY_TO_INDEX[key_str.strip("<>").lower()] = idx
        KEY_TO_INDEX[key_str.lower()] = idx

    @classmethod
    def parse(cls, action_str: str) -> dict[str, Any]:
        action_str = action_str.strip()

        # Schema defaults required by MiniWoB
        res = {
            "action_type": 0,  # NONE
            "ref": 0,
            "text": "",
            "key": 0,
            "coords": np.array([50.0, 50.0], dtype=np.float32),
            "field": 0,
        }

        if ActionTypes is None:
            return res

        # 1. CLICK(ref=4) or CLICK(4)
        click_match = re.search(r"CLICK\((?:ref=)?(\d+)\)", action_str, re.IGNORECASE)
        if click_match:
            ref = int(click_match.group(1))
            res["action_type"] = ACTION_TYPE_INDEX.get(ActionTypes.CLICK_ELEMENT, 8)
            res["ref"] = ref
            return res

        # 2. TYPE(ref=2, text="hello") or TYPE(2, "hello")
        type_match = re.search(
            r"TYPE\((?:ref=)?(\d+),\s*(?:text=)?[\"'](.*?)[\"']\)",
            action_str,
            re.IGNORECASE,
        )
        if type_match:
            ref = int(type_match.group(1))
            text = type_match.group(2)
            res["action_type"] = ACTION_TYPE_INDEX.get(ActionTypes.TYPE_TEXT, 10)
            res["ref"] = ref
            res["text"] = text
            return res

        # 3. PRESS(key="Enter") or PRESS("Enter")
        press_match = re.search(
            r"PRESS\((?:key=)?[\"'](.*?)[\"']\)", action_str, re.IGNORECASE
        )
        if press_match:
            key_name = press_match.group(1)
            key_idx = cls.KEY_TO_INDEX.get(key_name.strip("<>").lower(), 0)
            res["action_type"] = ACTION_TYPE_INDEX.get(ActionTypes.PRESS_KEY, 9)
            res["key"] = key_idx
            return res

        # 4. SCROLL(dx=0, dy=100) or SCROLL_DOWN / SCROLL_UP
        scroll_match = re.search(
            r"SCROLL\((?:dx=)?(-?\d+),\s*(?:dy=)?(-?\d+)\)", action_str, re.IGNORECASE
        )
        if scroll_match:
            dx = float(scroll_match.group(1))
            dy = float(scroll_match.group(2))
            res["action_type"] = ACTION_TYPE_INDEX.get(ActionTypes.MOVE_COORDS, 1)
            res["coords"] = np.array([max(10.0, dx), max(10.0, dy)], dtype=np.float32)
            return res

        if re.search(r"SCROLL_DOWN", action_str, re.IGNORECASE):
            res["action_type"] = ACTION_TYPE_INDEX.get(ActionTypes.SCROLL_DOWN_COORDS, 7)
            res["coords"] = np.array([50.0, 100.0], dtype=np.float32)
            return res

        if re.search(r"SCROLL_UP", action_str, re.IGNORECASE):
            res["action_type"] = ACTION_TYPE_INDEX.get(ActionTypes.SCROLL_UP_COORDS, 6)
            res["coords"] = np.array([50.0, 10.0], dtype=np.float32)
            return res

        # 5. Fallback integer: single integer -> CLICK(ref=X)
        if action_str.isdigit():
            res["action_type"] = ACTION_TYPE_INDEX.get(ActionTypes.CLICK_ELEMENT, 8)
            res["ref"] = int(action_str)
            return res

        return res


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
        elif all_envs and not dom_only:
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
        self._last_reward: float = 0.0

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
        self._last_reward = 0.0

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

        try:
            obs, reward, terminated, truncated, info = self._active_env.step(parsed_action)
            self._last_raw_obs = obs
            self._last_reward = float(reward)

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
        except Exception as e:
            # Handle out-of-bounds or invalid clicks gracefully without crashing the benchmark
            err_msg = str(e).split("\n")[0]
            prev_utterance = self._last_raw_obs.get("utterance", "") if isinstance(self._last_raw_obs, dict) else ""
            prev_dom = self._last_raw_obs.get("dom_elements", ()) if isinstance(self._last_raw_obs, dict) else ()
            formatted_dom = self.formatter.format_dom(prev_dom)
            prompt_obs = f"Action warning ({err_msg}).\n\nGoal: {prev_utterance}\n\nVisible DOM Elements:\n{formatted_dom}"

            return StepObservation(
                observation_text=prompt_obs,
                step_reward=0.0,
                is_done=False,
                info={"error": str(e)},
            )

    def evaluate(self, task: TaskInstance) -> bool:
        return self._last_reward > 0.0

    def close(self) -> None:
        if self._active_env is not None:
            try:
                self._active_env.close()
            except Exception:
                pass
            self._active_env = None
        self._active_task = None
        self._last_raw_obs = None
        self._last_reward = 0.0
