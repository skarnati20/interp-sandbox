"""
TextCraft Benchmark: Minecraft sequential recipe crafting environment.
Faithful implementation of ADaPT (Prasad et al., 2023).
Supports dynamic recipe DAG dependency traversal, procedural distractors,
and standard interactive actions ('get', 'craft', 'inventory', 'think:').
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import os
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
    """Parses agent textual action strings into environment actions."""

    @staticmethod
    def parse(action_str: str) -> ParsedAction:
        clean = action_str.strip()
        clean = re.sub(r"^>\s*", "", clean)  # Strip leading prompt markers

        # 1. Think
        if clean.lower().startswith("think:") or clean.lower().startswith("think"):
            thought_text = re.sub(r"^think:?\s*", "", clean, flags=re.IGNORECASE)
            return ParsedAction(action_type="think", raw_text=thought_text)

        # 2. Inventory
        if clean.lower().startswith("inventory"):
            return ParsedAction(action_type="inventory")

        # 3. Get action: get <count> <item> or get <item>
        get_match = re.match(r"^get\s+(\d+)\s+(.+)$", clean, re.IGNORECASE)
        if get_match:
            count = int(get_match.group(1))
            item = get_match.group(2).strip().rstrip(".")
            return ParsedAction(
                action_type="get",
                target=item,
                target_count=count,
            )

        get_simple_match = re.match(r"^get\s+(.+)$", clean, re.IGNORECASE)
        if get_simple_match:
            item = get_simple_match.group(1).strip().rstrip(".")
            return ParsedAction(
                action_type="get",
                target=item,
                target_count=1,
            )

        # 4. Craft action: craft <count> <item> using <count1> <ing1>, <count2> <ing2>...
        craft_match = re.match(
            r"^craft\s+(\d+)\s+(.+?)\s+using\s+(.+)$",
            clean,
            re.IGNORECASE,
        )
        if craft_match:
            target_count = int(craft_match.group(1))
            target_item = craft_match.group(2).strip()
            ing_raw = craft_match.group(3).strip()

            ingredients = []
            for part in ing_raw.split(","):
                part = part.strip().rstrip(".")
                part_match = re.match(r"^(\d+)\s+(.+)$", part)
                if part_match:
                    ingredients.append((part_match.group(2).strip(), int(part_match.group(1))))
                elif part:
                    ingredients.append((part, 1))

            return ParsedAction(
                action_type="craft",
                target=target_item,
                target_count=target_count,
                ingredients=ingredients,
            )

        # Fallback craft without count: craft <item> using <ingredients>
        craft_no_cnt = re.match(
            r"^craft\s+(.+?)\s+using\s+(.+)$",
            clean,
            re.IGNORECASE,
        )
        if craft_no_cnt:
            target_item = craft_no_cnt.group(1).strip()
            ing_raw = craft_no_cnt.group(2).strip()
            ingredients = []
            for part in ing_raw.split(","):
                part = part.strip().rstrip(".")
                part_match = re.match(r"^(\d+)\s+(.+)$", part)
                if part_match:
                    ingredients.append((part_match.group(2).strip(), int(part_match.group(1))))
                elif part:
                    ingredients.append((part, 1))
            return ParsedAction(
                action_type="craft",
                target=target_item,
                target_count=1,
                ingredients=ingredients,
            )

        return ParsedAction(action_type="unknown", raw_text=clean)


class TextCraftBenchmark(BaseBenchmark):
    """
    TextCraft sequential crafting benchmark environment.
    Official implementation aligned with ADaPT (Prasad et al., 2023).
    """

    DEFAULT_SYSTEM_PROMPT = (
        "You are an expert player playing a crafting game in Minecraft.\n"
        "You interact with the environment using the following actions:\n"
        "  - get <count> <item>: fetch raw base materials from the world\n"
        "  - craft <count> <target> using <count1> <ingredient1>, <count2> <ingredient2>, ...: craft items\n"
        "  - inventory: inspect your current inventory\n"
        "  - think: <thought>: perform reasoning steps (returns 'OK.')\n\n"
        "Rules:\n"
        "  1. You start with an empty inventory. Fetch raw base materials before crafting.\n"
        "  2. You cannot 'get' items that are craftable (you must craft them from raw materials).\n"
        "  3. Follow exact crafting recipes and counts provided in the command list."
    )

    def __init__(
        self,
        recipes_path: Optional[str | Path] = None,
        num_tasks: int = 50,
        min_depth: int = 2,
        max_depth: Optional[int] = None,
        max_distractors: int = 10,
        seed: int = 42,
        easy_first: bool = False,
    ):
        self.recipes_path = self._resolve_recipes_path(recipes_path)
        self.num_tasks = num_tasks
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.max_distractors = max_distractors
        self.seed = seed
        self.easy_first = easy_first

        self.crafting_tree = CraftingTree(self.recipes_path)
        self.parser = TextCraftActionParser()

        self.inventory: Counter = Counter()
        self.goal_item_id: str = ""
        self.goal_name: str = ""
        self.goal_count: int = 1
        self.active_task_id: Optional[str] = None
        self.is_completed: bool = False

        self.task_list = self._generate_tasks()

    def _resolve_recipes_path(self, recipes_path: Optional[str | Path]) -> Path:
        if recipes_path is not None:
            p = Path(recipes_path)
            if p.exists():
                return p

        # Check default paths in priority order: single recipes.json > directory
        candidates = [
            Path(__file__).resolve().parent / "recipes.json",
            Path(__file__).resolve().parent / "recipes.json.gz",
            Path(__file__).resolve().parent / "recipes",
            Path(__file__).resolve().parent.parent.parent.parent / "data" / "textcraft" / "recipes.json",
            Path(__file__).resolve().parent.parent.parent.parent / "data" / "textcraft" / "recipes",
            Path("data/textcraft/recipes.json"),
            Path("data/textcraft/recipes"),
        ]
        for c in candidates:
            if c.exists():
                return c

        raise FileNotFoundError(
            "Could not locate TextCraft recipes. "
            "Please ensure 'src/benchmark/textcraft/recipes.json' exists."
        )

    def _generate_tasks(self) -> list[TaskInstance]:
        items_with_depth = list(self.crafting_tree.item_recipes_min_depth(self.min_depth))
        if self.max_depth is not None:
            items_with_depth = [x for x in items_with_depth if x[1] <= self.max_depth]

        # Sort items deterministically: ascending if easy_first else descending by depth
        if self.easy_first:
            sorted_goals = sorted(items_with_depth, key=lambda x: (x[1], x[0]))
        else:
            sorted_goals = sorted(items_with_depth, key=lambda x: (-x[1], x[0]))

        if not sorted_goals:
            raise ValueError(
                f"No recipe items found with min_depth={self.min_depth} and max_depth={self.max_depth}"
            )

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

    def _item_str_to_obj(self, item_str: str) -> ItemTag:
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
            item_obj = self._item_str_to_obj(item_name)
            target_id = item_obj.item_id or item_obj.tag or ""

            if (
                self.crafting_tree.is_craftable(item_obj.name)
                or self.crafting_tree.is_tag(target_id)
                or not self.crafting_tree.is_valid_item(target_id)
            ):
                obs_text = f"Could not find {item_name}"
            else:
                self.inventory[target_id] += count
                obs_text = f"Got {count} {item_name}"

        elif parsed.action_type == "craft":
            target_str = parsed.target or ""
            target_count = max(1, parsed.target_count)
            out_obj = self._item_str_to_obj(target_str)
            out_with_cnt = ItemTagWithCount(out_obj, target_count)

            input_items: list[ItemTagWithCount] = []
            for ing_str, ing_cnt in (parsed.ingredients or []):
                ing_obj = self._item_str_to_obj(ing_str)
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
            obs_text = f"Could not execute {action_str}"

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
