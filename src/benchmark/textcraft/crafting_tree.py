from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import gzip
import json
import os
from pathlib import Path
import random
import re
from typing import Iterator, Optional


class ActionFailed(Exception):
    """Custom exception raised when an environment action cannot be performed."""
    pass


def item_id_to_str(item_id: str) -> str:
    """Converts 'minecraft:dark_oak_planks' -> 'dark oak planks'."""
    return item_id.replace("minecraft:", "").replace("_", " ")


def str_to_item_id(item_str: str) -> str:
    """Converts 'dark oak planks' or 'dark_oak_planks' -> 'minecraft:dark_oak_planks'."""
    clean = item_str.strip().lower()
    if clean.startswith("minecraft:"):
        return clean
    return f"minecraft:{clean.replace(' ', '_')}"


@dataclass(frozen=True)
class ItemTag:
    tag: Optional[str] = None
    item_id: Optional[str] = None

    @property
    def name(self) -> str:
        return self.item_id or self.tag or ""


@dataclass
class ItemTagWithCount:
    item_tag: ItemTag
    count: int


@dataclass(frozen=True)
class Recipe:
    input_items: list[ItemTagWithCount]
    output_item: ItemTagWithCount

    @property
    def recipe_str(self) -> str:
        output_str = f"craft {self.output_item.count} {item_id_to_str(self.output_item.item_tag.name)} using "
        input_strs = [
            f"{inp.count} {item_id_to_str(inp.item_tag.name)}"
            for inp in self.input_items
        ]
        return output_str + ", ".join(input_strs)


class CraftingTree:
    """
    Minecraft recipe dependency graph and dynamic crafting resolver.
    Faithfully implements the ADaPT (Prasad et al., 2023) TextCraft recipe system.
    Loads from a single consolidated recipes.json (or fallback directory).
    """

    def __init__(self, recipes_path: str | Path):
        self.recipes_path = Path(recipes_path)
        self.tag_recipes: dict[str, list[Recipe]] = {}
        self.itemid_recipes: dict[str, list[Recipe]] = {}
        self.tag_set: set[str] = set()
        self.itemid_set: set[str] = set()
        self.item_id_to_tag: dict[str, str] = {}
        self.transitive_dependencies: dict[str, set[str]] = {}
        self.min_depth: dict[str, int] = {}

        self._load_recipes(self.recipes_path)
        self.clean_up_recipes()

    def _load_recipes(self, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(f"Recipes file or directory not found: {path}")

        # Case 1: Single consolidated JSON or GZ file
        if path.is_file():
            if str(path).endswith(".gz"):
                with gzip.open(path, "rt", encoding="utf-8") as fp:
                    data = json.load(fp)
            else:
                with open(path, "r", encoding="utf-8") as fp:
                    data = json.load(fp)

            recipe_list = list(data.values()) if isinstance(data, dict) else data
            for recipe_details in recipe_list:
                self._parse_recipe(recipe_details)
            return

        # Case 2: Directory of individual JSON files
        for f in os.listdir(path):
            if not f.endswith(".json"):
                continue
            recipe_file = path / f
            try:
                with open(recipe_file, "r", encoding="utf-8") as fp:
                    recipe_details = json.load(fp)
                self._parse_recipe(recipe_details)
            except Exception:
                continue

    def _parse_recipe(self, recipe_details: dict) -> None:
        input_items: list[ItemTagWithCount] = []
        recipe_type = recipe_details.get("type", "")

        if recipe_type == "minecraft:crafting_shaped":
            pattern = recipe_details.get("pattern", [])
            keys = recipe_details.get("key", {})
            for key_char, item_spec in keys.items():
                count = 0
                if isinstance(item_spec, list):
                    item_spec = item_spec[0]
                for line in pattern:
                    count += line.count(key_char)
                if count == 0:
                    continue

                if "item" in item_spec:
                    item_name = item_spec["item"]
                    input_item = ItemTag(item_id=item_name)
                    self.itemid_set.add(item_name)
                elif "tag" in item_spec:
                    tag_name = item_spec["tag"]
                    if not tag_name.startswith("minecraft:"):
                        tag_name = "minecraft:" + tag_name
                    input_item = ItemTag(tag=tag_name)
                    self.tag_set.add(tag_name)
                else:
                    continue
                input_items.append(ItemTagWithCount(input_item, count))

        elif recipe_type == "minecraft:crafting_shapeless":
            item_name_idx: dict[str, int] = {}
            for ingredient in recipe_details.get("ingredients", []):
                if isinstance(ingredient, list):
                    ingredient = ingredient[0]
                if "item" in ingredient:
                    item_name = ingredient["item"]
                    input_item = ItemTag(item_id=item_name)
                    self.itemid_set.add(item_name)
                elif "tag" in ingredient:
                    tag_name = ingredient["tag"]
                    if not tag_name.startswith("minecraft:"):
                        tag_name = "minecraft:" + tag_name
                    input_item = ItemTag(tag=tag_name)
                    item_name = tag_name
                    self.tag_set.add(tag_name)
                else:
                    continue

                if item_name not in item_name_idx:
                    item_name_idx[item_name] = len(input_items)
                    input_items.append(ItemTagWithCount(input_item, 1))
                else:
                    idx = item_name_idx[item_name]
                    curr_count = input_items[idx].count
                    input_items[idx] = ItemTagWithCount(input_item, curr_count + 1)
        else:
            return

        if not input_items:
            return

        recipe_result = recipe_details.get("result")
        if not recipe_result:
            return

        if isinstance(recipe_result, str):
            output_item_id = recipe_result
            output_item_count = 1
        elif isinstance(recipe_result, dict):
            output_item_id = recipe_result.get("item") or recipe_result.get("result")
            output_item_count = recipe_result.get("count", 1)
        else:
            return

        if not output_item_id:
            return

        self.itemid_set.add(output_item_id)

        # Filter single item decompression from block
        if len(input_items) == 1 and input_items[0].item_tag.name.endswith("_block") and not output_item_id.endswith("_ingot") and not output_item_id.endswith("_nugget"):
            return

        output_tag = None
        if "group" in recipe_details:
            grp = recipe_details["group"]
            output_tag = grp if grp.startswith("minecraft:") else "minecraft:" + grp
            if output_tag != output_item_id:
                self.tag_set.add(output_tag)
                self.item_id_to_tag[output_item_id] = output_tag

        output_item = ItemTagWithCount(
            ItemTag(tag=output_tag, item_id=output_item_id),
            output_item_count,
        )
        recipe = Recipe(input_items, output_item)

        if output_item_id not in self.transitive_dependencies:
            self.transitive_dependencies[output_item_id] = set()

        skip_recipe = False
        for input_itemtag_count in input_items:
            input_item_name = input_itemtag_count.item_tag.name
            if input_item_name in self.transitive_dependencies:
                if output_item_id in self.transitive_dependencies[input_item_name]:
                    skip_recipe = True
                    break

        if not skip_recipe:
            if output_item_id not in self.itemid_recipes:
                self.itemid_recipes[output_item_id] = [recipe]
            else:
                self.itemid_recipes[output_item_id].append(recipe)

            for input_itemtag_count in input_items:
                input_item_name = input_itemtag_count.item_tag.name
                self.transitive_dependencies[output_item_id].add(input_item_name)
                if input_item_name in self.transitive_dependencies:
                    self.transitive_dependencies[output_item_id].update(
                        self.transitive_dependencies[input_item_name]
                    )

            if output_tag is not None:
                if output_tag not in self.tag_recipes:
                    self.tag_recipes[output_tag] = [recipe]
                else:
                    self.tag_recipes[output_tag].append(recipe)

    def clean_up_recipes(self) -> None:
        new_items = set()
        for item, recipes in self.itemid_recipes.items():
            for recipe in recipes:
                for input_item in recipe.input_items:
                    input_tag = input_item.item_tag.tag
                    if input_item.item_tag.item_id is None and input_tag is not None:
                        item_list = list(self.get_items_with_tags(input_tag))
                        success = any(i in self.itemid_recipes for i in item_list)
                        if not success:
                            input_item.item_tag = ItemTag(item_id=input_tag)
                            new_items.add(input_tag)

        for item in new_items:
            self.itemid_set.add(item)
            self.tag_set.discard(item)

    def get_items_with_tags(self, input_tag: str) -> Iterator[str]:
        for item_id, tag in self.item_id_to_tag.items():
            if input_tag == tag:
                yield item_id

    def _get_matching_target_recipes(self, target_id: str) -> list[Recipe]:
        if target_id in self.itemid_recipes:
            return self.itemid_recipes[target_id]
        if target_id + "s" in self.itemid_recipes:
            return self.itemid_recipes[target_id + "s"]
        if target_id.endswith("s") and target_id[:-1] in self.itemid_recipes:
            return self.itemid_recipes[target_id[:-1]]
        if target_id in self.tag_recipes:
            return self.tag_recipes[target_id]
        return []

    def craft(self, recipe: Recipe) -> Optional[ItemTagWithCount]:
        target_id = recipe.output_item.item_tag.item_id or ""
        target_recipes = self._get_matching_target_recipes(target_id)
        if not target_recipes:
            return None

        for target_recipe in target_recipes:
            success = True
            input_recipe_items_clone = deepcopy(recipe.input_items)

            for itemtag_count in target_recipe.input_items:
                itemtag = itemtag_count.item_tag
                input_itemtag_count = self.find_matching_item(itemtag, input_recipe_items_clone)
                if input_itemtag_count is None or input_itemtag_count.count != itemtag_count.count:
                    success = False
                    break
                input_recipe_items_clone.remove(input_itemtag_count)

            if len(input_recipe_items_clone) > 0:
                success = False

            if success:
                return target_recipe.output_item

        return None

    def find_matching_item(
        self,
        itemtag: ItemTag,
        input_recipe_items: list[ItemTagWithCount],
    ) -> Optional[ItemTagWithCount]:
        def match_name(n1: Optional[str], n2: Optional[str]) -> bool:
            if not n1 or not n2:
                return False
            if n1 == n2:
                return True
            if n1 + "s" == n2 or n2 + "s" == n1:
                return True
            return False

        for input_itemtag_count in input_recipe_items:
            inp_tag = input_itemtag_count.item_tag
            # 1. Exact or plural/singular item_id match
            if itemtag.item_id is not None:
                if match_name(inp_tag.item_id, itemtag.item_id):
                    return input_itemtag_count
                if match_name(inp_tag.tag, itemtag.item_id):
                    return input_itemtag_count
                if inp_tag.item_id and match_name(self.item_id_to_tag.get(inp_tag.item_id), itemtag.item_id):
                    return input_itemtag_count

            # 2. Tag match
            if itemtag.tag is not None:
                if match_name(inp_tag.tag, itemtag.tag):
                    return input_itemtag_count
                if inp_tag.item_id and match_name(self.item_id_to_tag.get(inp_tag.item_id), itemtag.tag):
                    return input_itemtag_count
                if match_name(inp_tag.item_id, itemtag.tag):
                    return input_itemtag_count

        return None

    def is_craftable(self, item: str) -> bool:
        clean = item if item.startswith("minecraft:") else f"minecraft:{item}"
        if clean in self.itemid_recipes or clean in self.tag_recipes:
            return True
        if clean + "s" in self.itemid_recipes or clean + "s" in self.tag_recipes:
            return True
        if clean.endswith("s") and (clean[:-1] in self.itemid_recipes or clean[:-1] in self.tag_recipes):
            return True
        return False

    def is_valid_item(self, item: str) -> bool:
        clean = item if item.startswith("minecraft:") else f"minecraft:{item}"
        if clean in self.itemid_set:
            return True
        if clean + "s" in self.itemid_set or (clean.endswith("s") and clean[:-1] in self.itemid_set):
            return True
        return False

    def is_tag(self, input_str: str) -> bool:
        clean = input_str if input_str.startswith("minecraft:") else f"minecraft:{input_str}"
        return clean in self.tag_set

    def collect_item_uses(self) -> dict[str, list[Recipe]]:
        item_uses: dict[str, list[Recipe]] = {}
        for item, recipes in self.itemid_recipes.items():
            for recipe in recipes:
                for input_itemtag in recipe.input_items:
                    name = input_itemtag.item_tag.name
                    item_uses.setdefault(name, []).append(recipe)
        for tag, recipes in self.tag_recipes.items():
            for recipe in recipes:
                for input_itemtag in recipe.input_items:
                    name = input_itemtag.item_tag.name
                    item_uses.setdefault(name, []).append(recipe)
        return item_uses

    def get_min_depth(self, item_tag: str) -> int:
        if item_tag in self.min_depth:
            return self.min_depth[item_tag]

        if item_tag in self.itemid_recipes:
            self.min_depth[item_tag] = self._get_min_depth_recipes(self.itemid_recipes[item_tag])
        elif item_tag in self.tag_recipes:
            self.min_depth[item_tag] = self._get_min_depth_recipes(self.tag_recipes[item_tag])
        else:
            self.min_depth[item_tag] = 0

        return self.min_depth[item_tag]

    def _get_min_depth_recipes(self, recipes: list[Recipe]) -> int:
        depths = []
        for recipe in recipes:
            recipe_depths = []
            for input_itemtag_count in recipe.input_items:
                recipe_depths.append(self.get_min_depth(input_itemtag_count.item_tag.name) + 1)
            depths.append(max(recipe_depths) if recipe_depths else 1)
        return min(depths) if depths else 0

    def item_recipes_min_depth(self, min_depth: int) -> Iterator[tuple[str, int]]:
        for item in sorted(self.itemid_recipes.keys()):
            item_depth = self.get_min_depth(item)
            if item_depth >= min_depth:
                yield item, item_depth

    def traverse_recipe_tree(self, item_name: str, visited_names: set[str]) -> list[Recipe]:
        if item_name in visited_names:
            return []
        recipes = list(self.itemid_recipes.get(item_name) or self.tag_recipes.get(item_name) or [])
        res = list(recipes)
        for recipe in recipes:
            new_visited = set(visited_names)
            new_visited.add(item_name)
            for input_itemtag_count in recipe.input_items:
                input_item_name = input_itemtag_count.item_tag.name
                res.extend(self.traverse_recipe_tree(input_item_name, new_visited))
        return res

    def create_recipe_set(
        self,
        item_name: str,
        max_distractors: int = 10,
        rng: Optional[random.Random] = None,
    ) -> tuple[list[Recipe], list[Recipe]]:
        r = rng or random.Random()
        item_uses = self.collect_item_uses()
        recipes = self.traverse_recipe_tree(item_name, set())
        distractors: list[Recipe] = []

        for recipe in recipes:
            for item in recipe.input_items:
                input_item_name = item.item_tag.name
                if input_item_name in item_uses:
                    input_item_uses_recipes = item_uses[input_item_name]
                    sample_k = min(len(input_item_uses_recipes), max_distractors)
                    distractors.extend(r.sample(input_item_uses_recipes, sample_k))

        return recipes, distractors
