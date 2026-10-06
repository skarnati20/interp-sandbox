"""
Unit and integration tests for the TextCraft benchmark implementation.
"""

from pathlib import Path
import unittest

from src.benchmark.textcraft import (
    CraftingTree,
    ItemTag,
    ItemTagWithCount,
    Recipe,
    TextCraftActionParser,
    TextCraftBenchmark,
    item_id_to_str,
    str_to_item_id,
)


class TestCraftingTree(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.recipes_file = (
            Path(__file__).resolve().parent.parent / "src" / "benchmark" / "textcraft" / "recipes.json"
        )
        cls.tree = CraftingTree(cls.recipes_file)

    def test_recipe_loading(self):
        self.assertGreater(len(self.tree.itemid_recipes), 400)
        self.assertGreater(len(self.tree.tag_recipes), 20)
        self.assertIn("minecraft:oak_planks", self.tree.itemid_recipes)

    def test_min_depth_calculation(self):
        items_d2 = list(self.tree.item_recipes_min_depth(2))
        self.assertGreater(len(items_d2), 300)
        planks_depth = self.tree.get_min_depth("minecraft:oak_planks")
        self.assertEqual(planks_depth, 1)

    def test_create_recipe_set(self):
        recipes, distractors = self.tree.create_recipe_set("minecraft:oak_boat", max_distractors=5)
        self.assertGreaterEqual(len(recipes), 2)
        recipe_targets = [r.output_item.item_tag.name for r in recipes]
        self.assertIn("minecraft:oak_boat", recipe_targets)


class TestTextCraftActionParser(unittest.TestCase):
    def test_parse_get(self):
        res1 = TextCraftActionParser.parse("get 2 oak_log")
        self.assertEqual(res1.action_type, "get")
        self.assertEqual(res1.target_count, 2)
        self.assertEqual(res1.target, "oak_log")

        res2 = TextCraftActionParser.parse("> get 1 dark oak log.")
        self.assertEqual(res2.action_type, "get")
        self.assertEqual(res2.target_count, 1)
        self.assertEqual(res2.target, "dark oak log")

    def test_parse_craft(self):
        res = TextCraftActionParser.parse("craft 4 oak planks using 1 oak log")
        self.assertEqual(res.action_type, "craft")
        self.assertEqual(res.target, "oak planks")
        self.assertEqual(res.target_count, 4)
        self.assertEqual(res.ingredients, [("oak log", 1)])

    def test_parse_llm_formats(self):
        # Action with <think> tag
        res_think = TextCraftActionParser.parse(
            "<think>\nI need oak logs first.\n</think>\nAction: get 1 oak logs"
        )
        self.assertEqual(res_think.action_type, "get")
        self.assertEqual(res_think.target, "oak logs")

        # Markdown block with Action prefix
        res_md = TextCraftActionParser.parse(
            "```bash\nAction: craft 4 oak planks using 1 oak logs\n```"
        )
        self.assertEqual(res_md.action_type, "craft")
        self.assertEqual(res_md.target, "oak planks")
        self.assertEqual(res_md.target_count, 4)

    def test_parse_inventory_and_think(self):
        res_inv = TextCraftActionParser.parse("inventory")
        self.assertEqual(res_inv.action_type, "inventory")

        res_think = TextCraftActionParser.parse("think: I need 2 sticks next.")
        self.assertEqual(res_think.action_type, "think")
        self.assertEqual(res_think.raw_text, "I need 2 sticks next.")


class TestTextCraftBenchmark(unittest.TestCase):
    def setUp(self):
        self.bench = TextCraftBenchmark(num_tasks=10, min_depth=2, seed=42)

    def test_task_list(self):
        tasks = self.bench.list_tasks()
        self.assertEqual(len(tasks), 10)
        self.assertTrue(tasks[0].task_id.startswith("textcraft/"))
        self.assertIn("Crafting commands:", tasks[0].info["initial_observation"])

    def test_workflow_execution(self):
        tasks = self.bench.list_tasks()
        task = tasks[0]
        obs = self.bench.reset(task)
        self.assertFalse(obs.is_done)

        # 1. Initial inventory should be empty
        obs_inv = self.bench.step("inventory")
        self.assertIn("You are not carrying anything", obs_inv.observation_text)

        # 2. Cannot get craftable item
        obs_fail = self.bench.step("get 1 oak_planks")
        self.assertIn("Could not find oak_planks", obs_fail.observation_text)

        # 3. Can get raw logs
        obs_get = self.bench.step("get 2 oak logs")
        self.assertIn("Got 2 oak logs", obs_get.observation_text)

        # 4. Craft planks
        obs_craft = self.bench.step("craft 4 oak planks using 1 oak logs")
        self.assertIn("Crafted 4 oak planks", obs_craft.observation_text)

        # 5. Inventory check reflects state
        obs_inv2 = self.bench.step("inventory")
        self.assertIn("oak planks", obs_inv2.observation_text)
        self.assertIn("oak logs", obs_inv2.observation_text)


if __name__ == "__main__":
    unittest.main()
