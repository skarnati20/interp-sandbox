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


class BigCodeBenchActionParser:
    """
    Parses and sanitizes LLM completions into valid executable Python code.
    Handles thinking tags (<think>...</think>), markdown code blocks,
    conversational text, and docstring function headers.
    """

    @staticmethod
    def extract_code(
        action_str: str,
        entry_point: Optional[str] = None,
        code_prompt: Optional[str] = None,
    ) -> str:
        clean = action_str.strip()

        # 1. Strip reasoning / thinking traces (e.g. <think>...</think>)
        clean = re.sub(r"<think>.*?</think>", "", clean, flags=re.DOTALL | re.IGNORECASE).strip()

        # 2. Extract markdown code blocks (```python ... ``` or ``` ... ```)
        md_pattern = r"```(?:python|py)?\s*\n(.*?)```"
        matches = re.findall(md_pattern, clean, flags=re.DOTALL | re.IGNORECASE)

        candidate_code = ""
        if matches:
            if entry_point:
                for block in matches:
                    if f"def {entry_point}" in block or (f"{entry_point}(" in block):
                        candidate_code = block.strip()
                        break
            if not candidate_code:
                candidate_code = matches[0].strip()
        else:
            # 3. Handle raw code or conversational output
            lines = clean.splitlines()
            code_lines = []
            capturing = False

            for line in lines:
                stripped = line.strip()
                # Check for standard Python start lines
                if stripped.startswith(("import ", "from ", "def ", "class ", "@")):
                    capturing = True
                if capturing:
                    code_lines.append(line)

            if code_lines:
                candidate_code = "\n".join(code_lines).strip()
            else:
                candidate_code = clean

        # 4. If only the function body was generated and entry_point definition is missing,
        # prepend code_prompt if available
        if entry_point and candidate_code:
            has_func_def = re.search(rf"\bdef\s+{re.escape(entry_point)}\b", candidate_code)
            if not has_func_def and code_prompt:
                # Prepend the signature and imports from code_prompt
                candidate_code = f"{code_prompt.rstrip()}\n{candidate_code}"

        return candidate_code.strip()


class BigCodeBenchmark(BaseBenchmark):
    """
    BigCodeBench benchmark environment adapter.
    Executes Python solutions against unittest test harnesses in isolated subprocesses.
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
        local_data_path: Optional[str | Path] = None,
        sample_only: bool = False,
        single_turn: bool = True,
    ):
        self.subset = subset.lower()
        self.split = split.lower()
        self.num_tasks = num_tasks
        self.seed = seed
        self.timeout = timeout
        self.sample_only = sample_only
        self.single_turn = single_turn
        self.parser = BigCodeBenchActionParser()

        # Load task specs
        self.task_specs = self._load_task_specs(local_data_path)
        if self.num_tasks is not None:
            self.task_specs = self.task_specs[: self.num_tasks]

        self.tasks: list[TaskInstance] = self._create_task_instances()

        # Active episode state
        self._active_task_spec: Optional[BigCodeBenchTaskSpec] = None
        self._last_code: str = ""
        self._is_completed: bool = False
        self._last_reward: float = 0.0

    def _load_task_specs(self, local_path: Optional[str | Path]) -> list[BigCodeBenchTaskSpec]:
        """Loads tasks from local file, HuggingFace dataset cache, or bundled sample tasks."""
        # 1. User specified local path
        if local_path and Path(local_path).exists():
            return self._load_from_file(Path(local_path))

        # 2. Bundled sample tasks if sample_only requested
        sample_file = Path(__file__).parent / "sample_tasks.json"
        if self.sample_only:
            if sample_file.exists():
                return self._load_from_file(sample_file)

        # 3. Try loading via HuggingFace Hub (parquet)
        repo_id = "bigcode/bigcodebench-hard" if self.subset == "hard" else "bigcode/bigcodebench"
        try:
            from huggingface_hub import hf_hub_download
            import pandas as pd

            # Prefer latest version, fallback to available parquets
            parquet_filename = "data/v0.1.2-00000-of-00001.parquet"
            fpath = hf_hub_download(repo_id=repo_id, filename=parquet_filename, repo_type="dataset")
            df = pd.read_parquet(fpath)
            specs = []
            for _, row in df.iterrows():
                libs_val = row.get("libs", [])
                if isinstance(libs_val, str):
                    try:
                        libs_val = json.loads(libs_val.replace("'", '"'))
                    except Exception:
                        libs_val = [l.strip() for l in libs_val.strip("[]").split(",") if l.strip()]
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
        except Exception:
            # Fallback to local sample tasks
            if sample_file.exists():
                return self._load_from_file(sample_file)
            raise RuntimeError(
                f"Could not load BigCodeBench dataset from {repo_id} and sample file {sample_file} not found."
            )

    def _load_from_file(self, filepath: Path) -> list[BigCodeBenchTaskSpec]:
        """Loads task specs from JSON or Parquet file."""
        if filepath.suffix == ".parquet":
            import pandas as pd

            df = pd.read_parquet(filepath)
            records = df.to_dict(orient="records")
        else:
            with open(filepath, "r", encoding="utf-8") as fp:
                records = json.load(fp)

        specs = []
        for r in records:
            libs = r.get("libs", [])
            if isinstance(libs, str):
                try:
                    libs = json.loads(libs.replace("'", '"'))
                except Exception:
                    libs = [l.strip() for l in libs.strip("[]").split(",") if l.strip()]

            specs.append(
                BigCodeBenchTaskSpec(
                    task_id=str(r["task_id"]),
                    entry_point=str(r["entry_point"]),
                    complete_prompt=str(r["complete_prompt"]),
                    instruct_prompt=str(r["instruct_prompt"]),
                    code_prompt=str(r["code_prompt"]),
                    canonical_solution=str(r["canonical_solution"]),
                    test=str(r["test"]),
                    doc_struct=str(r.get("doc_struct", "")),
                    libs=libs if isinstance(libs, list) else [],
                )
            )
        return specs

    def _create_task_instances(self) -> list[TaskInstance]:
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
        self._active_task_spec = self.task_specs[task_idx]
        self._last_code = ""
        self._is_completed = False
        self._last_reward = 0.0

        if self.split == "instruct":
            obs_text = (
                f"Specification:\n{self._active_task_spec.instruct_prompt}\n\n"
                f"Entry point function: {self._active_task_spec.entry_point}\n\n"
                f"Function signature:\n{self._active_task_spec.code_prompt.strip()}"
            )
        else:
            obs_text = (
                f"Complete the following Python function:\n\n"
                f"{self._active_task_spec.complete_prompt.strip()}"
            )

        return StepObservation(
            observation_text=obs_text,
            step_reward=0.0,
            is_done=False,
            info={"task_id": task.task_id, "entry_point": self._active_task_spec.entry_point},
        )

    def step(self, action_str: str) -> StepObservation:
        if not self._active_task_spec:
            raise RuntimeError("Environment not initialized. Call reset() before step().")

        # Handle submission / exit commands
        action_clean = action_str.strip()
        if action_clean.lower() in ("submit", "done", "exit") and self._last_code:
            return StepObservation(
                observation_text=f"Submitted solution. Success: {self._is_completed}",
                step_reward=1.0 if self._is_completed else 0.0,
                is_done=True,
                info={"success": self._is_completed},
            )

        # Parse and sanitize candidate Python solution
        code = self.parser.extract_code(
            action_clean,
            entry_point=self._active_task_spec.entry_point,
            code_prompt=self._active_task_spec.code_prompt,
        )
        self._last_code = code

        if not code:
            return StepObservation(
                observation_text="[Observation: Empty code received. Please output executable Python code.]",
                step_reward=0.0,
                is_done=self.single_turn,
                info={"warning": "empty_code"},
            )

        # Execute code against unit tests in isolated subprocess
        is_succ, error_msg = self._run_tests(code, self._active_task_spec)
        self._is_completed = is_succ
        self._last_reward = 1.0 if is_succ else 0.0

        if is_succ:
            obs_text = "All unit tests passed successfully!"
            is_done = True
        else:
            # Truncate error message if too long
            truncated_err = error_msg[:1200] + ("\n...[truncated]" if len(error_msg) > 1200 else "")
            obs_text = f"Test execution failed:\n{truncated_err}"
            is_done = self.single_turn

        return StepObservation(
            observation_text=obs_text,
            step_reward=self._last_reward,
            is_done=is_done,
            info={"success": is_succ, "error": error_msg if not is_succ else ""},
        )

    def _run_tests(self, solution_code: str, spec: BigCodeBenchTaskSpec) -> tuple[bool, str]:
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
            except Exception as e:
                return False, f"Execution error: {str(e)}"

    def evaluate(self, task: TaskInstance) -> bool:
        return self._is_completed

    def close(self) -> None:
        self._active_task_spec = None
        self._last_code = ""
        self._is_completed = False
        self._last_reward = 0.0
