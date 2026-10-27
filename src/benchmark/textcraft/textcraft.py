"""
TextCraft Benchmark: Minecraft sequential recipe crafting environment.
Faithful implementation of ADaPT (Prasad et al., 2023).
Supports dynamic recipe DAG dependency traversal, procedural distractors,
and standard interactive actions ('get', 'craft', 'inventory', 'think:').
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
import random
import re
from typing import Any, Optional

from ..base import BaseBenchmark, StepObservation, TaskInstance
from .crafting_tree import (
    ActionFailed,
    CraftingTree,
    ItemTag,
    ItemTagWithCount,
    Recipe,
    item_id_to_str,
    str_to_item_id,
)


@dataclass
class ParsedAction:
    action_type: str  # "get", "craft", "inventory", "think", or "unknown"
    target: Optional[str] = None
    target_count: int = 1
    ingredients: Optional[list[tuple[str, int]]] = None
    raw_text: str = ""


class TextCraftActionParser:
    """
    Robust action parser for LLM outputs in TextCraft.
    Handles thinking tags (<think>...</think>), ReAct prefixes ('Action:', 'Command:'),
    markdown codeblocks, conversational words ('more'), and natural language.
    """

    @staticmethod
    def parse(action_str: str) -> ParsedAction:
        clean = action_str.strip()

        # 1. Extract thought if <think>...</think> present
        think_match = re.search(r"<think>(.*?)</think>", clean, re.DOTALL | re.IGNORECASE)
        thought_trace = think_match.group(1).strip() if think_match else ""

        # Remove <think>...</think> blocks from execution parsing
        clean_text = re.sub(r"<think>.*?</think>", "", clean, flags=re.DOTALL | re.IGNORECASE).strip()

        # Remove markdown code blocks ```...```
        clean_text = re.sub(r"```(?:[a-zA-Z0-9_\-]+)?\n?", "", clean_text)
        clean_text = clean_text.replace("```", "").strip()

        # Search candidate lines
        candidates = []
        for line in clean_text.split("\n"):
            line = line.strip()
            if not line:
                continue
            # Strip common LLM prefixes: "Action:", "Command:", "> ", "- ", "1. "
            line_clean = re.sub(
                r"^(?:action|command|step \d+|turn \d+)\s*:\s*", "", line, flags=re.IGNORECASE
            )
            line_clean = re.sub(r"^[>\-\*\d\.\)]+\s+", "", line_clean).strip()
            if line_clean:
                candidates.append(line_clean)

        target_line = candidates[0] if candidates else clean_text

        # 2. Check for think/thought actions: "think: ...", "thought: ..."
        think_line_match = re.match(r"^(?:think|thought)\s*:\s*(.*)", target_line, re.IGNORECASE)
        if think_line_match:
            return ParsedAction(action_type="think", raw_text=think_line_match.group(1).strip())

        # If thought tag was present and no candidate action line remains, treat as thought
        if not target_line and thought_trace:
            return ParsedAction(action_type="think", raw_text=thought_trace)

        # 3. Check for inventory: "inventory", "inv", "check inventory"
        if re.match(r"^(?:check\s+)?inventory$|^inv$", target_line, re.IGNORECASE):
            return ParsedAction(action_type="inventory", raw_text=target_line)

        # 4. Check for 'get' commands: "get <count> <item>" or "get <item>"
        get_match = re.match(
            r"^get\s+(?:(\d+)\s+)?(?:more\s+)?([a-zA-Z0-9_\-\s]+?)(?:\s*\.|\s*$)",
            target_line,
            re.IGNORECASE,
        )
        if get_match:
            count_str, item = get_match.groups()
            cnt = int(count_str) if count_str else 1
            item = item.strip().lower()
            return ParsedAction(
                action_type="get",
                target=item,
                target_count=max(1, cnt),
                raw_text=target_line,
            )

        # 5. Check for 'craft' commands: "craft <count> <target> using <ingredients>"
        # e.g.: "craft 1 dark oak planks using 1 dark oak log"
        craft_match = re.match(
            r"^craft\s+(?:(\d+)\s+)?([a-zA-Z0-9_\-\s]+?)\s+using\s+(.+)$",
            target_line,
            re.IGNORECASE,
        )
        if craft_match:
            count_str, target_item, ings_str = craft_match.groups()
            target_cnt = int(count_str) if count_str else 1
            target_item = target_item.strip().lower()

            # Parse ingredients: comma or 'and' separated
            ing_tokens = re.split(r",\s*|\s+and\s+", ings_str.strip())
            ingredients = []
            for token in ing_tokens:
                token = token.strip()
                if not token:
                    continue
                item_match = re.match(r"^(?:(\d+)\s+)?([a-zA-Z0-9_\-\s]+?)$", token)
                if item_match:
                    c_str, ing_name = item_match.groups()
                    c = int(c_str) if c_str else 1
                    ingredients.append((ing_name.strip().lower(), max(1, c)))

            return ParsedAction(
                action_type="craft",
                target=target_item,
                target_count=max(1, target_cnt),
                ingredients=ingredients,
                raw_text=target_line,
            )

        # 6. Fallback: simple craft without 'using': "craft <count> <item>"
        simple_craft = re.match(
            r"^craft\s+(?:(\d+)\s+)?([a-zA-Z0-9_\-\s]+?)(?:\s*\.|\s*$)",
            target_line,
            re.IGNORECASE,
        )
        if simple_craft:
            c_str, item = simple_craft.groups()
            cnt = int(c_str) if c_str else 1
            return ParsedAction(
                action_type="craft",
                target=item.strip().lower(),
                target_count=max(1, cnt),
                ingredients=[],
                raw_text=target_line,
            )

        return ParsedAction(action_type="unknown", raw_text=action_str)


class TextCraftBenchmark(BaseBenchmark):
    """
    TextCraft environment implementing BaseBenchmark.
    Supports procedural task generation across recipe dependency depths (1-4).
    Validates state transitions:
      - 'get <item>': valid only for non-craftable raw materials.
      - 'craft <count> <target> using <ingredients>': validates recipe and checks inventory.
      - 'inventory': displays current carried items.
    """

    DEFAULT_SYSTEM_PROMPT = (
        "You are an agent playing Minecraft text crafting (TextCraft).\n"
        "At each turn, you can:\n"
        "1. Obtain raw materials: 'get <count> <item>' (NOTE: you can only 'get' base raw materials like wood logs, sand, ores. Any item with a recipe must be crafted from ingredients, not obtained with 'get').\n"
        "2. Craft items: 'craft <count> <target> using <count> <item>, ...'\n"
        "3. Check inventory: 'inventory'\n"
        "Always respond with a single executable command."
    )

    def __init__(
        self,
        recipes_path: Optional[str | Path] = None,
        num_tasks: int = 50,
        min_depth: int = 1,
        max_depth: Optional[int] = None,
        max_distractors: int = 10,
        seed: int = 42,
        easy_first: bool = False,
        balanced: bool = True,
    ):
        self.recipes_path = self.resolve_recipes_path(recipes_path)
        self.num_tasks = num_tasks
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.max_distractors = max_distractors
        self.seed = seed
        self.easy_first = easy_first
        self.balanced = balanced

        self.crafting_tree = CraftingTree(self.recipes_path)
        self.parser = TextCraftActionParser()

        self.inventory: Counter = Counter()
        self.goal_item_id: str = ""
        self.goal_name: str = ""
        self.goal_count: int = 1
        self.active_task_id: Optional[str] = None
        self.is_completed: bool = False

        self.task_list = self.generate_tasks()

    def resolve_recipes_path(self, recipes_path: Optional[str | Path]) -> Path:
        if recipes_path is not None:
            p = Path(recipes_path)
            if p.exists():
                return p

        # Check default paths: single recipes.json > compressed recipes.json.gz
        candidates = [
            Path(__file__).resolve().parent / "recipes.json",
            Path(__file__).resolve().parent / "recipes.json.gz",
            Path("data/textcraft/recipes.json"),
        ]
        for c in candidates:
            if c.exists():
                return c

        raise FileNotFoundError(
            "Could not locate TextCraft recipes. "
            "Please ensure 'src/benchmark/textcraft/recipes.json' exists."
        )

    def generate_tasks(self) -> list[TaskInstance]:
        items_with_depth = list(self.crafting_tree.item_recipes_min_depth(self.min_depth))
        if self.max_depth is not None:
            items_with_depth = [x for x in items_with_depth if x[1] <= self.max_depth]

        if not items_with_depth:
            raise ValueError(
                f"No recipe items found with min_depth={self.min_depth} and max_depth={self.max_depth}"
            )

        if self.balanced:
            # Group items by recipe depth
            by_depth: dict[int, list[str]] = {}
            for item, d in items_with_depth:
                by_depth.setdefault(d, []).append(item)

            depths = sorted(by_depth.keys())
            rng_select = random.Random(self.seed)
            for d in depths:
                rng_select.shuffle(by_depth[d])

            # Interleave items across depths for balanced difficulty (Depths 1, 2, 3, 4)
            sorted_goals: list[tuple[str, int]] = []
            ptrs = {d: 0 for d in depths}
            while len(sorted_goals) < self.num_tasks:
                added_any = False
                for d in depths:
                    if ptrs[d] < len(by_depth[d]) and len(sorted_goals) < self.num_tasks:
                        sorted_goals.append((by_depth[d][ptrs[d]], d))
                        ptrs[d] += 1
                        added_any = True
                if not added_any:
                    break
        elif self.easy_first:
            sorted_goals = sorted(items_with_depth, key=lambda x: (x[1], x[0]))
        else:
            sorted_goals = sorted(items_with_depth, key=lambda x: (-x[1], x[0]))

        tasks: list[TaskInstance] = []
        for task_idx in range(self.num_tasks):
            goal_item_id, depth = sorted_goals[task_idx % len(sorted_goals)]
            goal_name = item_id_to_str(goal_item_id)
            task_seed = self.seed + task_idx
            rng = random.Random(task_seed)

            recipes, distractors = self.crafting_tree.create_recipe_set(
                goal_item_id,
                max_distractors=self.max_distractors,
                rng=rng,
            )

            recipe_set = set(r.recipe_str for r in recipes)
            distractor_set = set(d.recipe_str for d in distractors if d.recipe_str not in recipe_set)
            combined_recipes = list(recipe_set) + rng.sample(
                list(distractor_set),
                min(len(distractor_set), self.max_distractors),
            )
            rng.shuffle(combined_recipes)

            initial_observation = (
                f"Crafting commands:\n"
                + "\n".join(combined_recipes)
                + f"\n\nGoal: craft {goal_name}."
            )

            task_id = f"textcraft/{goal_name.replace(' ', '_')}-s{task_seed}"
            tasks.append(
                TaskInstance(
                    task_id=task_id,
                    instruction=f"Craft {goal_name}.",
                    system_prompt=self.DEFAULT_SYSTEM_PROMPT,
                    info={
                        "goal_item_id": goal_item_id,
                        "goal_name": goal_name,
                        "goal_count": 1,
                        "depth": depth,
                        "task_idx": task_idx,
                        "initial_observation": initial_observation,
                        "recipes": combined_recipes,
                    },
                )
            )

        return tasks

    def list_tasks(self, split: str = "train") -> list[TaskInstance]:
        return self.task_list

    def reset(self, task: TaskInstance, seed: Optional[int] = None) -> StepObservation:
        self.active_task_id = task.task_id
        self.inventory = Counter()
        self.goal_item_id = task.info.get("goal_item_id", "")
        self.goal_name = task.info.get("goal_name", "")
        self.goal_count = task.info.get("goal_count", 1)
        self.is_completed = False

        obs_text = task.info.get("initial_observation", f"Goal: craft {self.goal_name}.")
        return StepObservation(
            observation_text=obs_text,
            step_reward=0.0,
            is_done=False,
            info={"inventory": dict(self.inventory)},
        )

    def item_str_to_obj(self, item_str: str) -> ItemTag:
        clean = item_str.strip().lower().replace(" ", "_")
        item_id = clean if clean.startswith("minecraft:") else f"minecraft:{clean}"

        # Plural/singular normalization
        if item_id not in self.crafting_tree.itemid_set:
            if item_id + "s" in self.crafting_tree.itemid_set:
                item_id = item_id + "s"
            elif item_id.endswith("s") and item_id[:-1] in self.crafting_tree.itemid_set:
                item_id = item_id[:-1]

        if self.crafting_tree.is_tag(item_id):
            return ItemTag(tag=item_id)
        return ItemTag(item_id=item_id)

    def step(self, action_str: str) -> StepObservation:
        parsed = self.parser.parse(action_str)
        obs_text = ""
        reward = 0.0
        done = False

        if parsed.action_type == "think":
            thought = parsed.raw_text.lower()
            if "task completed" in thought:
                done = True
                reward = 1.0 if self.is_completed else 0.0
            elif "task failed" in thought:
                done = True
                reward = 0.0
            obs_text = "OK."

        elif parsed.action_type == "inventory":
            obs_text = "Inventory: "
            items = [
                f"[{item_id_to_str(item)}] ({cnt})"
                for item, cnt in sorted(self.inventory.items())
                if cnt > 0
            ]
            obs_text += " ".join(items) if items else "You are not carrying anything."

        elif parsed.action_type == "get":
            item_name = parsed.target or ""
            count = max(1, parsed.target_count)
            item_obj = self.item_str_to_obj(item_name)
            target_id = item_obj.item_id or item_obj.tag or ""

            if self.crafting_tree.is_craftable(item_obj.name):
                obs_text = (
                    f"Could not find {item_name}. '{item_name}' is a crafted item; "
                    f"you cannot 'get' it directly. You must get its ingredients and craft it."
                )
            elif self.crafting_tree.is_tag(target_id) or not self.crafting_tree.is_valid_item(target_id):
                obs_text = f"Could not find {item_name}. '{item_name}' is not a valid raw material."
            else:
                self.inventory[target_id] += count
                obs_text = f"Got {count} {item_name}"

        elif parsed.action_type == "craft":
            target_str = parsed.target or ""
            target_count = max(1, parsed.target_count)
            out_obj = self.item_str_to_obj(target_str)
            out_with_cnt = ItemTagWithCount(out_obj, target_count)

            input_items: list[ItemTagWithCount] = []
            for ing_str, ing_cnt in (parsed.ingredients or []):
                ing_obj = self.item_str_to_obj(ing_str)
                input_items.append(ItemTagWithCount(ing_obj, max(1, ing_cnt)))

            recipe_candidate = Recipe(input_items=input_items, output_item=out_with_cnt)

            # Check inventory sufficiency
            has_enough = True
            for inp in input_items:
                req_id = inp.item_tag.item_id
                req_tag = inp.item_tag.tag
                curr_inv = 0
                if req_id and req_id in self.inventory:
                    curr_inv = self.inventory[req_id]
                elif req_id and req_id + "s" in self.inventory:
                    curr_inv = self.inventory[req_id + "s"]
                elif req_id and req_id.endswith("s") and req_id[:-1] in self.inventory:
                    curr_inv = self.inventory[req_id[:-1]]
                elif req_tag:
                    curr_inv = sum(
                        cnt for itm, cnt in self.inventory.items()
                        if self.crafting_tree.item_id_to_tag.get(itm) == req_tag
                    )
                if curr_inv < inp.count:
                    has_enough = False
                    break

            if not has_enough:
                obs_text = f"Could not find enough items to craft {target_str}"
            else:
                crafted_itemtag = self.crafting_tree.craft(recipe_candidate)
                if crafted_itemtag is None:
                    obs_text = f"Could not find a valid recipe for {target_str}"
                else:
                    # Deduct ingredients
                    for inp in input_items:
                        req_id = inp.item_tag.item_id
                        req_tag = inp.item_tag.tag
                        needed = inp.count
                        if req_id and self.inventory.get(req_id, 0) >= needed:
                            self.inventory[req_id] -= needed
                            if self.inventory[req_id] <= 0:
                                del self.inventory[req_id]
                        elif req_id and self.inventory.get(req_id + "s", 0) >= needed:
                            self.inventory[req_id + "s"] -= needed
                            if self.inventory[req_id + "s"] <= 0:
                                del self.inventory[req_id + "s"]
                        elif req_id and req_id.endswith("s") and self.inventory.get(req_id[:-1], 0) >= needed:
                            self.inventory[req_id[:-1]] -= needed
                            if self.inventory[req_id[:-1]] <= 0:
                                del self.inventory[req_id[:-1]]
                        elif req_tag:
                            for itm in list(self.inventory.keys()):
                                if self.crafting_tree.item_id_to_tag.get(itm) == req_tag:
                                    take = min(needed, self.inventory[itm])
                                    self.inventory[itm] -= take
                                    needed -= take
                                    if self.inventory[itm] <= 0:
                                        del self.inventory[itm]
                                    if needed <= 0:
                                        break

                    # Add crafted output
                    crafted_id = crafted_itemtag.item_tag.item_id or out_obj.item_id or ""
                    self.inventory[crafted_id] += crafted_itemtag.count
                    obs_text = f"Crafted {crafted_itemtag.count} {item_id_to_str(crafted_id)}"

                    # Check goal completion
                    if (
                        crafted_id == self.goal_item_id
                        or self.inventory.get(self.goal_item_id, 0) >= self.goal_count
                    ):
                        self.is_completed = True
                        reward = 1.0
                        done = True

        else:
            obs_text = f"Could not execute '{action_str}'"

        return StepObservation(
            observation_text=obs_text,
            step_reward=reward,
            is_done=done,
            info={"inventory": dict(self.inventory), "is_completed": self.is_completed},
        )

    def evaluate(self, task: TaskInstance) -> bool:
        return self.is_completed

    def close(self) -> None:
        pass
