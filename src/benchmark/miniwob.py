import re
from typing import Any, Optional, Union
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
            if elem.get("placeholder"):
                attr_parts.append(f'placeholder="{elem["placeholder"]}"')
            if elem.get("selected"):
                attr_parts.append("selected")

            attrs = " " + " ".join(attr_parts) if attr_parts else ""
            lines.append(f"[{ref}] <{tag}{attrs}>{text}</{tag}>")

        return "\n".join(lines) if lines else "(No visible DOM elements)"


class MiniWoBActionParser:
    """Parses LLM textual action strings into MiniWoB Action objects."""

    @staticmethod
    def resolve_ref(raw_ref: Any, dom_elements: tuple[dict[str, Any], ...] = ()) -> Optional[int]:
        if isinstance(raw_ref, int) or str(raw_ref).isdigit():
            return int(raw_ref)
        ref_str = str(raw_ref).strip("'\"")
        for elem in dom_elements:
            if ref_str in (elem.get("id"), elem.get("classes"), elem.get("text", "").strip()):
                return elem.get("ref")
        return None

    @classmethod
    def parse_action(
        cls,
        action_str: str,
        env: Any,
        dom_elements: tuple[dict[str, Any], ...] = (),
    ) -> dict[str, Any]:
        action_str = action_str.strip()

        # 1. Match CLICK(ref=1), CLICK(ref='subbtn'), or CLICK(1)
        click_match = re.search(
            r"CLICK\(.*?ref\s*=\s*['\"]?([a-zA-Z0-9_\-]+)['\"]?.*?\)|\bCLICK\((\d+)\)",
            action_str,
            re.IGNORECASE,
        )
        if click_match:
            raw_ref = click_match.group(1) or click_match.group(2)
            ref_id = cls.resolve_ref(raw_ref, dom_elements)
            if ref_id is not None:
                try:
                    return env.unwrapped.create_action(ActionTypes.CLICK_ELEMENT, ref=ref_id)
                except Exception:
                    pass

        # 2. Match TYPE(ref=..., text="...") or TYPE(..., "...")
        type_match = re.search(
            r'TYPE\(.*?ref\s*=\s*[\'"]?([a-zA-Z0-9_\-]+)[\'"]?.*?text\s*=\s*["\'](.*?)["\'].*?\)|TYPE\((\d+),\s*["\'](.*?)["\']\)',
            action_str,
            re.IGNORECASE,
        )
        if type_match:
            raw_ref = type_match.group(1) or type_match.group(3)
            text_val = type_match.group(2) or type_match.group(4)
            ref_id = cls.resolve_ref(raw_ref, dom_elements)
            if ref_id is not None:
                try:
                    return env.unwrapped.create_action(ActionTypes.TYPE_TEXT, ref=ref_id, text=text_val)
                except Exception:
                    pass

        # 3. Match SELECT(ref=..., text="...") or SELECT_OPTION
        select_match = re.search(
            r'SELECT(?:_OPTION)?\(.*?ref\s*=\s*[\'"]?([a-zA-Z0-9_\-]+)[\'"]?.*?text\s*=\s*["\'](.*?)["\'].*?\)|SELECT(?:_OPTION)?\((\d+),\s*["\'](.*?)["\']\)',
            action_str,
            re.IGNORECASE,
        )
        if select_match:
            raw_ref = select_match.group(1) or select_match.group(3)
            option_text = (select_match.group(2) or select_match.group(4) or "").strip().lower()

            # Find matching option child in DOM
            opt_ref = None
            for elem in dom_elements:
                if elem.get("tag") == "option" and (
                    elem.get("text", "").strip().lower() == option_text
                    or elem.get("value", "").strip().lower() == option_text
                ):
                    opt_ref = elem.get("ref")
                    break

            if opt_ref is not None:
                try:
                    return env.unwrapped.create_action(ActionTypes.CLICK_ELEMENT, ref=opt_ref)
                except Exception:
                    pass

            ref_id = cls.resolve_ref(raw_ref, dom_elements)
            if ref_id is not None:
                try:
                    return env.unwrapped.create_action(ActionTypes.FOCUS_ELEMENT_AND_TYPE_TEXT, ref=ref_id, text=option_text)
                except Exception:
                    try:
                        return env.unwrapped.create_action(ActionTypes.CLICK_ELEMENT, ref=ref_id)
                    except Exception:
                        pass

        # 4. Match PRESS_ENTER()
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

    DOM_ENV_NAMES = [
        # Button & Checkbox Tasks
        "miniwob/click-button-v1",
        "miniwob/click-button-sequence-v1",
        "miniwob/click-checkboxes-v1",
        "miniwob/click-checkboxes-soft-v1",
        "miniwob/click-checkboxes-transfer-v1",
        "miniwob/click-checkboxes-large-v1",
        "miniwob/number-checkboxes-v1",
        "miniwob/click-dialog-v1",
        "miniwob/click-dialog-2-v1",
        "miniwob/click-link-v1",
        "miniwob/click-test-v1",
        "miniwob/click-test-2-v1",
        "miniwob/click-test-transfer-v1",
        "miniwob/click-widget-v1",
        "miniwob/click-collapsible-v1",
        "miniwob/click-collapsible-nodelay-v1",
        "miniwob/click-tab-v1",
        "miniwob/click-tab-2-v1",
        "miniwob/click-tab-2-easy-v1",
        "miniwob/click-option-v1",
        "miniwob/click-menu-v1",
        "miniwob/click-shades-v1",
        # Text & Input Tasks
        "miniwob/enter-text-v1",
        "miniwob/enter-text-2-v1",
        "miniwob/enter-text-dynamic-v1",
        "miniwob/enter-password-v1",
        "miniwob/enter-date-v1",
        "miniwob/enter-time-v1",
        "miniwob/focus-text-v1",
        "miniwob/focus-text-2-v1",
        "miniwob/copy-paste-v1",
        "miniwob/copy-paste-2-v1",
        "miniwob/search-engine-v1",
        "miniwob/text-transform-v1",
        # Structured Data, Tables & Email Tasks
        "miniwob/login-user-v1",
        "miniwob/login-user-popup-v1",
        "miniwob/read-table-v1",
        "miniwob/read-table-2-v1",
        "miniwob/email-inbox-v1",
        "miniwob/email-inbox-delete-v1",
        "miniwob/email-inbox-important-v1",
        "miniwob/social-media-v1",
        "miniwob/social-media-all-v1",
        "miniwob/social-media-some-v1",
        "miniwob/simple-arithmetic-v1",
    ]

    DEFAULT_SYSTEM_PROMPT = (
        "You are an autonomous web navigation agent. Complete the objective in the fewest steps possible.\n\n"
        "Commands:\n"
        "  - CLICK(ref=ID) : Clicks buttons, checkboxes, radio buttons, tabs, links, or options.\n"
        "  - TYPE(ref=ID, text=\"STRING\") : Types string into an input box or textarea.\n"
        "  - SELECT(ref=ID, text=\"OPTION\") : Selects a dropdown menu option.\n"
        "  - PRESS_ENTER() : Submits the active form.\n\n"
        "Rules:\n"
        "  1. Inspect the Goal and the Visible Elements list.\n"
        "  2. For multi-step forms (e.g. login, booking, search), complete each required input or dropdown, then click Submit or Search.\n"
        "  3. If an input already contains the required text in `value=\"...\"`, do not re-type it; proceed to the next field or button.\n"
        "  4. Output ONLY the exact command."
    )

    @classmethod
    def get_all_registered_envs(cls) -> list[str]:
        """Returns all 120+ MiniWoB++ environment IDs registered in Gymnasium."""
        return sorted([
            env_id for env_id in gymnasium.envs.registry.keys()
            if env_id.startswith("miniwob/")
        ])

    def __init__(
        self,
        env_names: Optional[Union[list[str], str]] = None,
        wait_ms: int = 150,
        render_mode: Optional[str] = None,
        episodes_per_env: int = 1,
    ):
        if env_names == "all" or env_names == ["all"]:
            self.env_names = self.get_all_registered_envs()
        elif env_names == "dom_only" or env_names == ["dom_only"]:
            self.env_names = self.DOM_ENV_NAMES
        else:
            self.env_names = env_names or self.DEFAULT_ENV_NAMES

        self.wait_ms = wait_ms
        self.render_mode = render_mode
        self.episodes_per_env = max(1, episodes_per_env)
        self.current_env = None
        self.current_task_id = None
        self.dom_elements: tuple[dict[str, Any], ...] = ()
        self.cumulative_reward = 0.0
        self.formatter = MiniWoBDomFormatter()
        self.parser = MiniWoBActionParser()

    def list_tasks(self, split: str = "train") -> list[TaskInstance]:
        tasks = []
        for name in self.env_names:
            for ep_idx in range(self.episodes_per_env):
                task_id = name if self.episodes_per_env == 1 else f"{name}_ep{ep_idx}"
                tasks.append(
                    TaskInstance(
                        task_id=task_id,
                        instruction="Accomplish the goal indicated in the web page utterance.",
                        system_prompt=self.DEFAULT_SYSTEM_PROMPT,
                        info={"env_name": name, "episode_idx": ep_idx},
                    )
                )
        return tasks

    def reset(self, task: TaskInstance, seed: Optional[int] = None) -> StepObservation:
        self.close()

        env_name = task.info.get("env_name", task.task_id)
        self.current_task_id = task.task_id
        self.current_env = gymnasium.make(
            env_name,
            render_mode=self.render_mode,
            wait_ms=self.wait_ms,
        )
        obs, info = self.current_env.reset(seed=seed)
        self.cumulative_reward = 0.0
        self.dom_elements = obs.get("dom_elements", ())

        utterance = obs.get("utterance", "")
        dom_text = self.formatter.format_dom(self.dom_elements)

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

        action = self.parser.parse_action(action_str, self.current_env, dom_elements=self.dom_elements)
        obs, reward, terminated, truncated, info = self.current_env.step(action)

        self.cumulative_reward += float(reward)
        is_done = bool(terminated or truncated)

        self.dom_elements = obs.get("dom_elements", ())
        dom_text = self.formatter.format_dom(self.dom_elements)
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
