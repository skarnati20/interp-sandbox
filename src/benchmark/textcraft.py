"""
TextCraft Benchmark: Minecraft-style sequential recipe crafting environment.
Official implementation based on ADaPT (Prasad et al., 2023) and AgentGym.
Matches the task structure in 'Doomed from the Start: Early Abort of LLM Agent Episodes via a Recall-Controlled Probe Cascade'.
"""

from collections import Counter
from dataclasses import dataclass
import re
from typing import Any, Optional, Union

from .base import BaseBenchmark, StepObservation, TaskInstance


@dataclass
class Recipe:
    target: str
    target_count: int
    ingredients: dict[str, int]

    def to_string(self) -> str:
        ing_str = ", ".join(f"{cnt} {name}" for name, cnt in self.ingredients.items())
        return f"craft {self.target_count} {self.target} using {ing_str}"


# Standard Minecraft crafting recipe database (official TextCraft recipe set)
TEXTCRAFT_RECIPES: list[Recipe] = [
    # Basic Wood & Stick
    Recipe("oak_planks", 4, {"oak_log": 1}),
    Recipe("birch_planks", 4, {"birch_log": 1}),
    Recipe("spruce_planks", 4, {"spruce_log": 1}),
    Recipe("dark_oak_planks", 4, {"dark_oak_log": 1}),
    Recipe("stick", 4, {"oak_planks": 2}),
    Recipe("stick", 4, {"birch_planks": 2}),
    Recipe("stick", 4, {"spruce_planks": 2}),
    Recipe("crafting_table", 1, {"oak_planks": 4}),
    Recipe("crafting_table", 1, {"birch_planks": 4}),
    Recipe("chest", 1, {"oak_planks": 8}),
    Recipe("ladder", 3, {"stick": 7}),
    Recipe("fence", 3, {"oak_planks": 4, "stick": 2}),
    Recipe("fence_gate", 1, {"stick": 4, "oak_planks": 2}),
    Recipe("wooden_door", 3, {"oak_planks": 6}),
    Recipe("trapdoor", 2, {"oak_planks": 6}),
    Recipe("bowl", 4, {"oak_planks": 3}),
    Recipe("boat", 1, {"oak_planks": 5}),

    # Wooden Tools
    Recipe("wooden_pickaxe", 1, {"oak_planks": 3, "stick": 2}),
    Recipe("wooden_axe", 1, {"oak_planks": 3, "stick": 2}),
    Recipe("wooden_sword", 1, {"oak_planks": 2, "stick": 1}),
    Recipe("wooden_shovel", 1, {"oak_planks": 1, "stick": 2}),
    Recipe("wooden_hoe", 1, {"oak_planks": 2, "stick": 2}),

    # Stone Tools & Items
    Recipe("stone_pickaxe", 1, {"cobblestone": 3, "stick": 2}),
    Recipe("stone_axe", 1, {"cobblestone": 3, "stick": 2}),
    Recipe("stone_sword", 1, {"cobblestone": 2, "stick": 1}),
    Recipe("stone_shovel", 1, {"cobblestone": 1, "stick": 2}),
    Recipe("stone_hoe", 1, {"cobblestone": 2, "stick": 2}),
    Recipe("furnace", 1, {"cobblestone": 8}),
    Recipe("lever", 1, {"cobblestone": 1, "stick": 1}),
    Recipe("stone_slab", 6, {"cobblestone": 3}),
    Recipe("stone_stairs", 4, {"cobblestone": 6}),

    # Iron Tools & Armor
    Recipe("iron_ingot", 1, {"iron_nugget": 9}),
    Recipe("iron_nugget", 9, {"iron_ingot": 1}),
    Recipe("iron_block", 1, {"iron_ingot": 9}),
    Recipe("iron_pickaxe", 1, {"iron_ingot": 3, "stick": 2}),
    Recipe("iron_axe", 1, {"iron_ingot": 3, "stick": 2}),
    Recipe("iron_sword", 1, {"iron_ingot": 2, "stick": 1}),
    Recipe("iron_shovel", 1, {"iron_ingot": 1, "stick": 2}),
    Recipe("iron_hoe", 1, {"iron_ingot": 2, "stick": 2}),
    Recipe("iron_helmet", 1, {"iron_ingot": 5}),
    Recipe("iron_chestplate", 1, {"iron_ingot": 8}),
    Recipe("iron_leggings", 1, {"iron_ingot": 7}),
    Recipe("iron_boots", 1, {"iron_ingot": 4}),
    Recipe("shears", 1, {"iron_ingot": 2}),
    Recipe("bucket", 1, {"iron_ingot": 3}),
    Recipe("shield", 1, {"iron_ingot": 1, "oak_planks": 6}),
    Recipe("compass", 1, {"iron_ingot": 4, "redstone": 1}),
    Recipe("iron_bars", 16, {"iron_ingot": 6}),
    Recipe("anvil", 1, {"iron_block": 3, "iron_ingot": 4}),
    Recipe("flint_and_steel", 1, {"iron_ingot": 1, "flint": 1}),

    # Gold & Diamond Items
    Recipe("gold_ingot", 1, {"gold_nugget": 9}),
    Recipe("gold_nugget", 9, {"gold_ingot": 1}),
    Recipe("gold_block", 1, {"gold_ingot": 9}),
    Recipe("golden_pickaxe", 1, {"gold_ingot": 3, "stick": 2}),
    Recipe("golden_sword", 1, {"gold_ingot": 2, "stick": 1}),
    Recipe("golden_apple", 1, {"gold_ingot": 8, "apple": 1}),
    Recipe("clock", 1, {"gold_ingot": 4, "redstone": 1}),
    Recipe("diamond_block", 1, {"diamond": 9}),
    Recipe("diamond_pickaxe", 1, {"diamond": 3, "stick": 2}),
    Recipe("diamond_axe", 1, {"diamond": 3, "stick": 2}),
    Recipe("diamond_sword", 1, {"diamond": 2, "stick": 1}),
    Recipe("diamond_shovel", 1, {"diamond": 1, "stick": 2}),
    Recipe("diamond_helmet", 1, {"diamond": 5}),
    Recipe("diamond_chestplate", 1, {"diamond": 8}),
    Recipe("diamond_leggings", 1, {"diamond": 7}),
    Recipe("diamond_boots", 1, {"diamond": 4}),
    Recipe("enchanting_table", 1, {"book": 1, "diamond": 2, "obsidian": 4}),
    Recipe("jukebox", 1, {"diamond": 1, "oak_planks": 8}),

    # Food & Agriculture
    Recipe("bread", 1, {"wheat": 3}),
    Recipe("hay_bale", 1, {"wheat": 9}),
    Recipe("wheat", 9, {"hay_bale": 1}),
    Recipe("sugar", 1, {"sugar_cane": 1}),
    Recipe("paper", 3, {"sugar_cane": 3}),
    Recipe("book", 1, {"paper": 3, "leather": 1}),
    Recipe("cake", 1, {"wheat": 3, "sugar": 2, "egg": 1, "milk_bucket": 3}),
    Recipe("pumpkin_pie", 1, {"pumpkin": 1, "sugar": 1, "egg": 1}),
    Recipe("cookie", 8, {"wheat": 2, "cocoa_beans": 1}),
    Recipe("mushroom_stew", 1, {"bowl": 1, "red_mushroom": 1, "brown_mushroom": 1}),

    # Mechanisms & Combat
    Recipe("torch", 4, {"coal": 1, "stick": 1}),
    Recipe("redstone_torch", 1, {"redstone": 1, "stick": 1}),
    Recipe("bow", 1, {"stick": 3, "string": 3}),
    Recipe("arrow", 4, {"flint": 1, "stick": 1, "feather": 1}),
    Recipe("fishing_rod", 1, {"stick": 3, "string": 2}),
    Recipe("piston", 1, {"oak_planks": 3, "cobblestone": 4, "iron_ingot": 1, "redstone": 1}),
    Recipe("dispenser", 1, {"cobblestone": 7, "bow": 1, "redstone": 1}),
    Recipe("dropper", 1, {"cobblestone": 7, "redstone": 1}),
    Recipe("tnt", 1, {"gunpowder": 5, "sand": 4}),

    # Dyes & Decorative
    Recipe("white_wool", 1, {"string": 4}),
    Recipe("bone_meal", 3, {"bone": 1}),
    Recipe("red_dye", 1, {"poppy": 1}),
    Recipe("yellow_dye", 1, {"dandelion": 1}),
    Recipe("orange_dye", 2, {"red_dye": 1, "yellow_dye": 1}),
    Recipe("pink_dye", 2, {"red_dye": 1, "bone_meal": 1}),
    Recipe("cyan_dye", 2, {"blue_dye": 1, "green_dye": 1}),
    Recipe("sandstone", 1, {"sand": 4}),
    Recipe("smooth_sandstone", 4, {"sandstone": 4}),
    Recipe("sandstone_stairs", 4, {"sandstone": 6}),
]


class TextCraftActionParser:
    """Parses LLM textual action strings into crafting targets and ingredients."""

    @staticmethod
    def parse_craft(action_str: str) -> Optional[dict[str, Any]]:
        action_str = action_str.strip()
        # Format: craft [count] [item] using [count1] [ing1], [count2] [ing2]...
        match = re.search(
            r"craft\s+(\d+)\s+([a-zA-Z0-9_\-]+)\s+using\s+(.+)",
            action_str,
            re.IGNORECASE,
        )
        if not match:
            # Short fallback: craft [item]
            short_match = re.search(r"craft\s+([a-zA-Z0-9_\-]+)", action_str, re.IGNORECASE)
            if short_match:
                return {
                    "action_type": "craft_auto",
                    "target": short_match.group(1).lower(),
                    "target_count": 1,
                }
            return None

        target_count = int(match.group(1))
        target_item = match.group(2).lower()
        ing_raw = match.group(3).strip()

        ingredients = {}
        for part in ing_raw.split(","):
            part = part.strip()
            item_match = re.search(r"(\d+)\s+([a-zA-Z0-9_\-]+)", part)
            if item_match:
                cnt = int(item_match.group(1))
                item_name = item_match.group(2).lower()
                ingredients[item_name] = cnt

        return {
            "action_type": "craft",
            "target": target_item,
            "target_count": target_count,
            "ingredients": ingredients,
        }


class TextCraftBenchmark(BaseBenchmark):
    """
    TextCraft sequential crafting environment benchmark adapter.
    Aligns with official ADaPT / AgentGym and 'Doomed from the Start' (2026).
    """

    DEFAULT_SYSTEM_PROMPT = (
        "You are given few useful crafting recipes to craft items in Minecraft.\n"
        "Crafting commands are of the format \"craft [target object] using [input ingredients]\".\n"
        "Every round I will give you an observation, you have to respond with an action based on the state and instruction.\n"
        "Respond ONLY with a command of the format:\n"
        "  craft <quantity> <target_object> using <quantity1> <ingredient1>, <quantity2> <ingredient2>, ...\n\n"
        "Rules:\n"
        "  1. Only craft items if you have sufficient ingredients in your inventory.\n"
        "  2. Follow intermediate crafting steps in order (e.g. log -> planks -> sticks -> tool).\n"
        "  3. Output ONLY the exact crafting command."
    )

    def __init__(
        self,
        num_tasks: int = 51,
        max_recipe_depth: int = 4,
        seed: int = 42,
    ):
        self.num_tasks = num_tasks
        self.max_recipe_depth = max_recipe_depth
        self.seed = seed
        self.recipes = TEXTCRAFT_RECIPES
        self.parser = TextCraftActionParser()

        self.inventory: Counter = Counter()
        self.target_goal: tuple[str, int] = ("", 1)
        self.active_task_id: Optional[str] = None
        self.is_completed = False
        self.task_list = self._generate_tasks()

    def _generate_tasks(self) -> list[TaskInstance]:
        """Generates standard multi-step TextCraft tasks with varying depths (Depths 1-4)."""
        task_configs = [
            # 1-Step Tasks (Depth 1)
            ("craft_planks", {"oak_log": 2}, "oak_planks", 4),
            ("craft_birch_planks", {"birch_log": 2}, "birch_planks", 4),
            ("craft_sugar", {"sugar_cane": 3}, "sugar", 2),
            ("craft_paper", {"sugar_cane": 6}, "paper", 3),
            ("craft_bread", {"wheat": 6}, "bread", 2),
            ("craft_iron_nuggets", {"iron_ingot": 2}, "iron_nugget", 9),
            ("craft_gold_ingot", {"gold_nugget": 18}, "gold_ingot", 2),
            ("craft_red_dye", {"poppy": 3}, "red_dye", 2),
            ("craft_bone_meal", {"bone": 2}, "bone_meal", 6),
            ("craft_sandstone", {"sand": 8}, "sandstone", 2),

            # 2-Step Tasks (Depth 2)
            ("craft_sticks", {"oak_log": 1}, "stick", 4),
            ("craft_crafting_table", {"birch_log": 1}, "crafting_table", 1),
            ("craft_wooden_pickaxe", {"oak_planks": 3, "stick": 2}, "wooden_pickaxe", 1),
            ("craft_wooden_sword", {"oak_planks": 2, "stick": 1}, "wooden_sword", 1),
            ("craft_wooden_axe", {"oak_planks": 3, "stick": 2}, "wooden_axe", 1),
            ("craft_stone_pickaxe", {"cobblestone": 3, "stick": 2}, "stone_pickaxe", 1),
            ("craft_stone_sword", {"cobblestone": 2, "stick": 1}, "stone_sword", 1),
            ("craft_furnace", {"cobblestone": 10}, "furnace", 1),
            ("craft_shears", {"iron_ingot": 3}, "shears", 1),
            ("craft_shield", {"iron_ingot": 2, "oak_planks": 8}, "shield", 1),
            ("craft_bucket", {"iron_ingot": 4}, "bucket", 1),
            ("craft_torches", {"coal": 2, "stick": 2}, "torch", 8),
            ("craft_book", {"sugar_cane": 3, "leather": 1}, "book", 1),
            ("craft_orange_dye", {"poppy": 1, "dandelion": 1}, "orange_dye", 2),
            ("craft_pink_dye", {"poppy": 1, "bone": 1}, "pink_dye", 2),

            # 3-Step Tasks (Depth 3)
            ("craft_wooden_pickaxe_from_logs", {"oak_log": 2}, "wooden_pickaxe", 1),
            ("craft_wooden_sword_from_logs", {"oak_log": 2}, "wooden_sword", 1),
            ("craft_stone_pickaxe_from_logs", {"cobblestone": 3, "oak_log": 1}, "stone_pickaxe", 1),
            ("craft_iron_pickaxe", {"iron_ingot": 3, "oak_log": 1}, "iron_pickaxe", 1),
            ("craft_iron_sword", {"iron_ingot": 2, "oak_log": 1}, "iron_sword", 1),
            ("craft_iron_axe", {"iron_ingot": 3, "oak_log": 1}, "iron_axe", 1),
            ("craft_iron_helmet", {"iron_nugget": 45}, "iron_helmet", 1),
            ("craft_iron_chestplate", {"iron_ingot": 8}, "iron_chestplate", 1),
            ("craft_iron_leggings", {"iron_ingot": 7}, "iron_leggings", 1),
            ("craft_iron_boots", {"iron_ingot": 4}, "iron_boots", 1),
            ("craft_bow", {"stick": 3, "string": 3}, "bow", 1),
            ("craft_fishing_rod", {"oak_log": 1, "string": 2}, "fishing_rod", 1),
            ("craft_arrows", {"flint": 2, "stick": 2, "feather": 2}, "arrow", 8),
            ("craft_compass", {"iron_ingot": 4, "redstone": 1}, "compass", 1),
            ("craft_clock", {"gold_ingot": 4, "redstone": 1}, "clock", 1),
            ("craft_golden_apple", {"gold_ingot": 8, "apple": 1}, "golden_apple", 1),
            ("craft_diamond_sword", {"diamond": 2, "oak_log": 1}, "diamond_sword", 1),
            ("craft_diamond_pickaxe", {"diamond": 3, "oak_log": 1}, "diamond_pickaxe", 1),
            ("craft_diamond_helmet", {"diamond": 5}, "diamond_helmet", 1),
            ("craft_smooth_sandstone_stairs", {"sand": 16}, "sandstone_stairs", 4),

            # 4-Step Tasks (Depth 4)
            ("craft_iron_pickaxe_from_nuggets", {"iron_nugget": 27, "oak_log": 1}, "iron_pickaxe", 1),
            ("craft_piston", {"oak_log": 1, "cobblestone": 4, "iron_ingot": 1, "redstone": 1}, "piston", 1),
            ("craft_dispenser", {"cobblestone": 7, "stick": 3, "string": 3, "redstone": 1}, "dispenser", 1),
            ("craft_cake", {"wheat": 3, "sugar_cane": 2, "egg": 1, "milk_bucket": 3}, "cake", 1),
            ("craft_enchanting_table", {"sugar_cane": 3, "leather": 1, "diamond": 2, "obsidian": 4}, "enchanting_table", 1),
            ("craft_anvil", {"iron_ingot": 31}, "anvil", 1),
        ]

        tasks = []
        for i, (name, init_inv, target_item, target_count) in enumerate(task_configs):
            task_id = f"textcraft/{name}-v1"
            tasks.append(
                TaskInstance(
                    task_id=task_id,
                    instruction=f"Craft {target_count} {target_item}.",
                    system_prompt=self.DEFAULT_SYSTEM_PROMPT,
                    info={
                        "init_inventory": init_inv,
                        "target_item": target_item,
                        "target_count": target_count,
                        "task_idx": i,
                    },
                )
            )
        return tasks

    def list_tasks(self, split: str = "train") -> list[TaskInstance]:
        return self.task_list

    def _format_inventory(self) -> str:
        items = [f"{cnt} {item}" for item, cnt in sorted(self.inventory.items()) if cnt > 0]
        return ", ".join(items) if items else "(empty)"

    def _format_observation(self, message: str = "") -> str:
        target_item, target_cnt = self.target_goal
        inv_str = self._format_inventory()

        # List candidate recipes that match inventory items or the target
        relevant = []
        for r in self.recipes:
            # Include recipe if any ingredient is in inventory or if it produces target
            if r.target == target_item or any(self.inventory.get(ing, 0) > 0 for ing in r.ingredients):
                relevant.append(r.to_string())

        recipe_block = "\n".join(f"  {r_str}" for r_str in relevant[:15])

        text = f"Goal: craft {target_cnt} {target_item}\n"
        text += f"Current inventory: {inv_str}\n\n"
        if message:
            text += f"{message}\n\n"
        text += f"Crafting commands:\n{recipe_block}"
        return text

    def reset(self, task: TaskInstance, seed: Optional[int] = None) -> StepObservation:
        self.active_task_id = task.task_id
        self.inventory = Counter(task.info.get("init_inventory", {}))
        self.target_goal = (task.info["target_item"], task.info["target_count"])
        self.is_completed = False

        obs_text = self._format_observation()
        return StepObservation(
            observation_text=obs_text,
            step_reward=0.0,
            is_done=False,
            info={"inventory": dict(self.inventory)},
        )

    def step(self, action_str: str) -> StepObservation:
        parsed = self.parser.parse_craft(action_str)

        if not parsed:
            obs_text = self._format_observation(
                f"Invalid action format: '{action_str}'. Use format 'craft <count> <target> using <count1> <ing1>, ...'"
            )
            return StepObservation(
                observation_text=obs_text,
                step_reward=0.0,
                is_done=False,
                info={"inventory": dict(self.inventory)},
            )

        target_item = parsed["target"]
        target_count = parsed["target_count"]

        # Find matching recipe
        matching_recipe: Optional[Recipe] = None
        for r in self.recipes:
            if r.target == target_item and (parsed.get("action_type") == "craft_auto" or r.target_count == target_count):
                matching_recipe = r
                break

        if not matching_recipe:
            obs_text = self._format_observation(f"No recipe found to craft {target_count} {target_item}.")
            return StepObservation(
                observation_text=obs_text,
                step_reward=0.0,
                is_done=False,
                info={"inventory": dict(self.inventory)},
            )

        # Check if inventory has required ingredients
        has_ingredients = True
        for ing, needed in matching_recipe.ingredients.items():
            if self.inventory.get(ing, 0) < needed:
                has_ingredients = False
                break

        if not has_ingredients:
            needed_str = ", ".join(f"{cnt} {name}" for name, cnt in matching_recipe.ingredients.items())
            obs_text = self._format_observation(
                f"Cannot craft {matching_recipe.target_count} {matching_recipe.target}: Insufficient ingredients. Requires {needed_str}."
            )
            return StepObservation(
                observation_text=obs_text,
                step_reward=0.0,
                is_done=False,
                info={"inventory": dict(self.inventory)},
            )

        # Deduct ingredients and add crafted item
        for ing, needed in matching_recipe.ingredients.items():
            self.inventory[ing] -= needed
            if self.inventory[ing] <= 0:
                del self.inventory[ing]

        self.inventory[matching_recipe.target] += matching_recipe.target_count

        # Check goal condition
        goal_item, goal_cnt = self.target_goal
        if self.inventory.get(goal_item, 0) >= goal_cnt:
            self.is_completed = True
            obs_text = self._format_observation(f"You have successfully crafted {goal_cnt} {goal_item}!")
            return StepObservation(
                observation_text=obs_text,
                step_reward=1.0,
                is_done=True,
                info={"inventory": dict(self.inventory), "success": True},
            )

        obs_text = self._format_observation(
            f"You crafted {matching_recipe.target_count} {matching_recipe.target}."
        )
        return StepObservation(
            observation_text=obs_text,
            step_reward=0.0,
            is_done=False,
            info={"inventory": dict(self.inventory)},
        )

    def evaluate(self, task: TaskInstance) -> bool:
        return self.is_completed

    def close(self) -> None:
        pass
