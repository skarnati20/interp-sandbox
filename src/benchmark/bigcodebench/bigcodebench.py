"""
BigCodeBench Benchmark Adapter.
Evaluates Python code generation and iterative software-engineering tasks
against the BigCodeBench suite (ICLR 2025).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
from typing import Any, Optional

from ..base import BaseBenchmark, StepObservation, TaskInstance


@dataclass
class BigCodeBenchTaskSpec:
    task_id: str
    entry_point: str
    complete_prompt: str
    instruct_prompt: str
    code_prompt: str
    canonical_solution: str
    test: str
    doc_struct: str = ""
    libs: list[str] = field(default_factory=list)


class BigCodeBenchmark(BaseBenchmark):
    """
    BigCodeBench benchmark environment adapter.
    Executes Python solutions against unittest test harnesses in isolated subprocesses.
    Loads tasks exclusively from the official HuggingFace Hub dataset.
    """

    DEFAULT_SYSTEM_PROMPT = (
        "You are an expert Python software engineer.\n"
        "Your task is to write clean, complete, and correct Python code that satisfies the given specification.\n"
        "Respond strictly with the Python code solving the task inside a markdown code block (```python ... ```) or as raw Python code."
    )

    def __init__(
        self,
        subset: str = "full",  # "full" or "hard"
        split: str = "instruct",  # "instruct" or "complete"
        num_tasks: Optional[int] = None,
        seed: int = 42,
        timeout: float = 15.0,
    ):
        self.subset = subset.lower()
        self.split = split.lower()
        self.num_tasks = num_tasks
        self.seed = seed
        self.timeout = timeout

        # Load task specs exclusively from HuggingFace
        self.task_specs = self.load_task_specs()
        if self.num_tasks is not None:
            self.task_specs = self.task_specs[: self.num_tasks]

        self.tasks: list[TaskInstance] = self.create_task_instances()

        # Active episode state
        self.active_task_spec: Optional[BigCodeBenchTaskSpec] = None
        self.last_code: str = ""
        self.is_completed: bool = False
        self.last_reward: float = 0.0

    @staticmethod
    def extract_code(
        action_str: str,
        entry_point: Optional[str] = None,
        code_prompt: Optional[str] = None,
    ) -> str:
        """
        Parses and sanitizes LLM completions into valid executable Python code.
        Strips thinking tags, extracts markdown blocks or raw code, and prepends
        function headers if the model only generated the body.
        """
        # 1. Strip reasoning traces (<think>...</think>)
        text = re.sub(r"<think>.*?</think>", "", action_str, flags=re.DOTALL | re.IGNORECASE).strip()

        # 2. Extract code from markdown blocks, preferring the block with entry_point
        blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
        if blocks:
            code = next((b for b in blocks if entry_point and (f"def {entry_point}" in b or f"{entry_point}(" in b)), blocks[0]).strip()
        else:
            # For raw code, strip any conversational preamble before code keywords
            match = re.search(r"^(?:import |from |def |class |@)", text, re.MULTILINE)
            code = text[match.start():].strip() if match else text

        # 3. If model generated only the function body, prepend function signature
        if entry_point and code and code_prompt and not re.search(rf"\bdef\s+{re.escape(entry_point)}\b", code):
            code = f"{code_prompt.rstrip()}\n{code}"

        return code.strip()

    def load_task_specs(self) -> list[BigCodeBenchTaskSpec]:
        """Loads tasks exclusively from HuggingFace Hub."""
        repo_id = "bigcode/bigcodebench-hard" if self.subset == "hard" else "bigcode/bigcodebench"
        try:
            from huggingface_hub import hf_hub_download
            import pandas as pd

            parquet_filename = "data/v0.1.2-00000-of-00001.parquet"
            fpath = hf_hub_download(repo_id=repo_id, filename=parquet_filename, repo_type="dataset")
            df = pd.read_parquet(fpath)
            specs = []
            for row_idx, row in df.iterrows():
                libs_val = row.get("libs", [])
                if isinstance(libs_val, str):
                    try:
                        libs_val = json.loads(libs_val.replace("'", '"'))
                    except Exception:
                        libs_val = [lib.strip() for lib in libs_val.strip("[]").split(",") if lib.strip()]
                elif not isinstance(libs_val, list):
                    libs_val = []

                specs.append(
                    BigCodeBenchTaskSpec(
                        task_id=str(row["task_id"]),
                        entry_point=str(row["entry_point"]),
                        complete_prompt=str(row["complete_prompt"]),
                        instruct_prompt=str(row["instruct_prompt"]),
                        code_prompt=str(row["code_prompt"]),
                        canonical_solution=str(row["canonical_solution"]),
                        test=str(row["test"]),
                        doc_struct=str(row.get("doc_struct", "")),
                        libs=libs_val,
                    )
                )
            return specs
        except Exception as err:
            raise RuntimeError(
                f"Could not load BigCodeBench dataset from {repo_id}: {err}"
            )

    def create_task_instances(self) -> list[TaskInstance]:
        instances = []
        for i, spec in enumerate(self.task_specs):
            if self.split == "instruct":
                instruction = spec.instruct_prompt
            else:
                instruction = f"Complete the Python function '{spec.entry_point}' according to its specification."

            task_id = f"bigcodebench/{spec.task_id.split('/')[-1] if '/' in spec.task_id else spec.task_id}"

            instances.append(
                TaskInstance(
                    task_id=task_id,
                    instruction=instruction,
                    system_prompt=self.DEFAULT_SYSTEM_PROMPT,
                    info={
                        "task_idx": i,
                        "raw_task_id": spec.task_id,
                        "entry_point": spec.entry_point,
                        "libs": spec.libs,
                        "subset": self.subset,
                        "split": self.split,
                    },
                )
            )
        return instances

    def list_tasks(self, split: str = "train") -> list[TaskInstance]:
        return self.tasks

    def reset(self, task: TaskInstance, seed: Optional[int] = None) -> StepObservation:
        self.close()

        task_idx = task.info.get("task_idx", 0)
        self.active_task_spec = self.task_specs[task_idx]
        self.last_code = ""
        self.is_completed = False
        self.last_reward = 0.0

        if self.split == "instruct":
            obs_text = (
                f"Specification:\n{self.active_task_spec.instruct_prompt}\n\n"
                f"Entry point function: {self.active_task_spec.entry_point}\n\n"
                f"Function signature:\n{self.active_task_spec.code_prompt.strip()}"
            )
        else:
            obs_text = (
                f"Complete the following Python function:\n\n"
                f"{self.active_task_spec.complete_prompt.strip()}"
            )

        return StepObservation(
            observation_text=obs_text,
            step_reward=0.0,
            is_done=False,
            info={"task_id": task.task_id, "entry_point": self.active_task_spec.entry_point},
        )

    def step(self, action_str: str) -> StepObservation:
        if not self.active_task_spec:
            raise RuntimeError("Environment not initialized. Call reset() before step().")

        # Handle submission / exit commands
        action_clean = action_str.strip()
        if action_clean.lower() in ("submit", "done", "exit") and self.last_code:
            return StepObservation(
                observation_text=f"Submitted solution. Success: {self.is_completed}",
                step_reward=1.0 if self.is_completed else 0.0,
                is_done=True,
                info={"success": self.is_completed},
            )

        # Parse and sanitize candidate Python solution
        code = self.extract_code(
            action_clean,
            entry_point=self.active_task_spec.entry_point,
            code_prompt=self.active_task_spec.code_prompt,
        )
        self.last_code = code

        if not code:
            return StepObservation(
                observation_text="[Observation: Empty code received. Please output executable Python code.]",
                step_reward=0.0,
                is_done=False,
                info={"warning": "empty_code"},
            )

        # Execute code against unit tests in isolated subprocess
        is_succ, error_msg = self.run_tests(code, self.active_task_spec)
        self.is_completed = is_succ
        self.last_reward = 1.0 if is_succ else 0.0

        if is_succ:
            obs_text = "All unit tests passed successfully!"
            is_done = True
        else:
            # Truncate error message if too long
            truncated_err = error_msg[:1200] + ("\n...[truncated]" if len(error_msg) > 1200 else "")
            obs_text = f"Test execution failed:\n{truncated_err}"
            is_done = False

        return StepObservation(
            observation_text=obs_text,
            step_reward=self.last_reward,
            is_done=is_done,
            info={"success": is_succ, "error": error_msg if not is_succ else ""},
        )

    def run_tests(self, solution_code: str, spec: BigCodeBenchTaskSpec) -> tuple[bool, str]:
        """Runs the solution code with the task's unittest suite in a subprocess."""
        test_script = (
            f"import unittest\n"
            f"import sys\n"
            f"\n"
            f"# Solution Code\n"
            f"{solution_code}\n"
            f"\n"
            f"# Unit Tests\n"
            f"{spec.test}\n"
            f"\n"
            f"if __name__ == '__main__':\n"
            f"    unittest.main()\n"
        )

        with tempfile.TemporaryDirectory(prefix="bigcodebench_eval_") as tmp_dir:
            run_file = Path(tmp_dir) / "test_run.py"
            run_file.write_text(test_script, encoding="utf-8")

            try:
                res = subprocess.run(
                    [sys.executable, str(run_file)],
                    cwd=tmp_dir,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                )
                if res.returncode == 0:
                    return True, ""
                output = res.stderr.strip() or res.stdout.strip()
                return False, output
            except subprocess.TimeoutExpired:
                return False, f"TimeoutExpired: Execution exceeded {self.timeout} seconds."
            except Exception as err:
                return False, f"Execution error: {str(err)}"

    def evaluate(self, task: TaskInstance) -> bool:
        return self.is_completed

    def close(self) -> None:
        self.active_task_spec = None
        self.last_code = ""
        self.is_completed = False
        self.last_reward = 0.0
