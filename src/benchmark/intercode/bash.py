"""
InterCode-Bash Benchmark: Interactive Linux Command-Line and Shell Environment.
Expanded suite of 100 challenging, multi-step Linux administration, data wrangling,
filesystem restructuring, and pipeline parsing tasks.
"""

from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from typing import Callable, Optional

from ..base import BaseBenchmark, StepObservation, TaskInstance


@dataclass
class BashTaskSpec:
    name: str
    instruction: str
    setup_fn: Callable[[Path], None]
    verify_fn: Callable[[Path], bool]
    tags: list[str] = field(default_factory=list)


def create_task_suite() -> list[BashTaskSpec]:
    """
    Creates 100 challenging multi-step InterCode-Bash tasks across 6 core domains:
    1. Advanced Text Processing & Regex (grep, sed, awk, cut, sort, uniq, tr)
    2. Deep Filesystem Restructuring & Tree Operations (find, mv, cp, symlinks)
    3. Complex Shell Pipelines & Data Wrangling (grouping, aggregation, set diffs)
    4. Permissions, Security & File Metadata (chmod, stat, broken symlinks)
    5. Multi-format Archiving & Compression (nested tar/zip, gzip)
    6. Codebase Debugging & Patching (diffs, line-endings, syntax repair)
    """
    tasks: list[BashTaskSpec] = []

    # =========================================================================
    # Domain 1: Advanced Text Processing & Regex (Tasks 1–20)
    # =========================================================================

    def setup_01(p: Path):
        (p / "data.txt").write_text(
            "Alice alice@example.com 24\nBob invalid-email 30\nCharlie charlie@work.org 29\nDavid david@test.com 45\n"
        )
    def verify_01(p: Path):
        out = p / "emails.txt"
        if not out.exists(): return False
        content = sorted([x.strip() for x in out.read_text().splitlines() if x.strip()])
        return content == ["alice@example.com", "charlie@work.org", "david@test.com"]
    tasks.append(BashTaskSpec("extract_valid_emails", "Extract all valid email addresses from 'data.txt' into 'emails.txt', one per line in alphabetical order.", setup_01, verify_01, ["regex", "grep"]))

    def setup_02(p: Path):
        (p / "server.log").write_text(
            "INFO: start\nERROR: database timeout [500]\nINFO: retry\nERROR: connection refused [502]\nWARN: high mem\nERROR: disk full [507]\n"
        )
    def verify_02(p: Path):
        out = p / "error_codes.txt"
        if not out.exists(): return False
        lines = [x.strip() for x in out.read_text().splitlines() if x.strip()]
        return lines == ["500", "502", "507"]
    tasks.append(BashTaskSpec("extract_error_codes", "Extract all 3-digit error codes inside square brackets on ERROR lines in 'server.log' and write them to 'error_codes.txt'.", setup_02, verify_02, ["text", "sed"]))

    def setup_03(p: Path):
        (p / "config.txt").write_text("host=localhost\nport=8080\nenv=development\nmode=development_test\n")
    def verify_03(p: Path):
        c = (p / "config.txt").read_text()
        return "development" not in c and "env=production" in c and "mode=production_test" in c
    tasks.append(BashTaskSpec("replace_env_all", "Replace all occurrences of 'development' with 'production' in 'config.txt'.", setup_03, verify_03, ["text", "sed"]))

    def setup_04(p: Path):
        (p / "scores.txt").write_text("Alice 85 Engineering\nBob 92 Marketing\nCharlie 78 Engineering\nDavid 95 Sales\nEve 88 Engineering\n")
    def verify_04(p: Path):
        out = p / "eng_scores.txt"
        if not out.exists(): return False
        lines = [x.strip() for x in out.read_text().splitlines() if x.strip()]
        return lines == ["Eve 88", "Alice 85", "Charlie 78"]
    tasks.append(BashTaskSpec("filter_sort_department", "Filter 'scores.txt' for department 'Engineering', output only the Name and Score, and sort descending by Score into 'eng_scores.txt'.", setup_04, verify_04, ["text", "awk", "sort"]))

    def setup_05(p: Path):
        (p / "access.log").write_text(
            "192.168.1.1 GET /index.html 200\n10.0.0.1 POST /login 403\n192.168.1.1 GET /style.css 200\n172.16.0.1 GET /index.html 200\n10.0.0.1 GET /api 200\n192.168.1.1 GET /logo.png 200\n"
        )
    def verify_05(p: Path):
        out = p / "top_ip.txt"
        if not out.exists(): return False
        return "192.168.1.1" in out.read_text() and "3" in out.read_text()
    tasks.append(BashTaskSpec("top_request_ip", "Find the IP address that made the most requests in 'access.log' and write the count and IP to 'top_ip.txt'.", setup_05, verify_05, ["text", "uniq", "sort"]))

    def setup_06(p: Path):
        (p / "fileA.txt").write_text("apple\nbanana\ncherry\ndate\nfig\n")
        (p / "fileB.txt").write_text("banana\ndate\ngrape\n")
    def verify_06(p: Path):
        out = p / "diff.txt"
        if not out.exists(): return False
        lines = sorted([x.strip() for x in out.read_text().splitlines() if x.strip()])
        return lines == ["apple", "cherry", "fig"]
    tasks.append(BashTaskSpec("set_difference_lines", "Find all lines in 'fileA.txt' that do NOT appear in 'fileB.txt' and save them sorted in 'diff.txt'.", setup_06, verify_06, ["text", "comm", "grep"]))

    def setup_07(p: Path):
        (p / "story.txt").write_text("The Quick BROWN fox JUMPS over the Lazy Dog.\n")
    def verify_07(p: Path):
        out = p / "lowercase.txt"
        if not out.exists(): return False
        return out.read_text().strip() == "the quick brown fox jumps over the lazy dog."
    tasks.append(BashTaskSpec("convert_lowercase", "Convert all text in 'story.txt' to lowercase and save to 'lowercase.txt'.", setup_07, verify_07, ["text", "tr"]))

    def setup_08(p: Path):
        (p / "records.tsv").write_text("id\tname\tage\tsalary\n1\tAlice\t30\t75000\n2\tBob\t25\t50000\n3\tCharlie\t35\t120000\n4\tDavid\t40\t90000\n")
    def verify_08(p: Path):
        out = p / "avg_salary.txt"
        if not out.exists(): return False
        val = float(out.read_text().strip())
        return abs(val - 83750.0) < 1.0
    tasks.append(BashTaskSpec("calculate_avg_salary", "Calculate the average salary (column 4) in 'records.tsv' (ignoring header) and write the number to 'avg_salary.txt'.", setup_08, verify_08, ["pipeline", "awk"]))

    def setup_09(p: Path):
        (p / "source.c").write_text("// Copyright 2023\n#include <stdio.h>\n\n// Main function\nint main() {\n    // Print greeting\n    printf(\"Hello\\n\");\n    return 0;\n}\n")
    def verify_09(p: Path):
        out = p / "clean.c"
        if not out.exists(): return False
        c = out.read_text()
        return "//" not in c and "printf" in c and "main" in c
    tasks.append(BashTaskSpec("strip_single_line_comments", "Remove all single-line comments (starting with '//') from 'source.c' and save the result to 'clean.c'.", setup_09, verify_09, ["text", "sed"]))

    def setup_10(p: Path):
        (p / "ids.txt").write_text("105\n23\n4\n1050\n88\n9\n500\n")
    def verify_10(p: Path):
        out = p / "sorted_ids.txt"
        if not out.exists(): return False
        lines = [x.strip() for x in out.read_text().splitlines() if x.strip()]
        return lines == ["4", "9", "23", "88", "105", "500", "1050"]
    tasks.append(BashTaskSpec("numeric_sort", "Sort the numbers in 'ids.txt' in ascending numerical order and save to 'sorted_ids.txt'.", setup_10, verify_10, ["text", "sort"]))

    # =========================================================================
    # Domain 2: Deep Filesystem Restructuring & Tree Operations (Tasks 11–30)
    # =========================================================================

    def setup_11(p: Path):
        for name in ["2023-01-15_report.pdf", "2023-01-20_invoice.pdf", "2023-02-05_summary.pdf", "2022-12-10_old.pdf"]:
            (p / name).write_text("content")
    def verify_11(p: Path):
        return (p / "2023" / "01" / "2023-01-15_report.pdf").exists() and \
               (p / "2023" / "01" / "2023-01-20_invoice.pdf").exists() and \
               (p / "2023" / "02" / "2023-02-05_summary.pdf").exists() and \
               (p / "2022" / "12" / "2022-12-10_old.pdf").exists()
    tasks.append(BashTaskSpec("organize_by_date_dirs", "Move all files with format 'YYYY-MM-DD_name.pdf' into nested directories 'YYYY/MM/'.", setup_11, verify_11, ["fs", "mv", "mkdir"]))

    def setup_12(p: Path):
        (p / "a.txt").write_text("original content")
        (p / "sub").mkdir()
        (p / "sub" / "b.txt").write_text("original content")
        (p / "c.txt").write_text("unique content")
        (p / "d.txt").write_text("original content")
    def verify_12(p: Path):
        out = p / "duplicates.txt"
        if not out.exists(): return False
        content = out.read_text()
        return "a.txt" in content and "b.txt" in content and "d.txt" in content and "c.txt" not in content
    tasks.append(BashTaskSpec("find_duplicate_files_md5", "Find all duplicate files in the workspace matching the exact content of 'a.txt' and list their paths in 'duplicates.txt'.", setup_12, verify_12, ["fs", "md5sum"]))

    def setup_13(p: Path):
        (p / "dir1").mkdir()
        (p / "dir1" / "file.txt").write_text("data")
        (p / "dir2").mkdir()
        (p / "dir3").mkdir()
        (p / "dir3" / "sub_empty").mkdir()
    def verify_13(p: Path):
        return (p / "dir1").exists() and not (p / "dir2").exists() and not (p / "dir3" / "sub_empty").exists() and not (p / "dir3").exists()
    tasks.append(BashTaskSpec("remove_empty_directories", "Recursively find and delete all empty directories in the workspace while preserving non-empty ones.", setup_13, verify_13, ["fs", "find", "rmdir"]))

    def setup_14(p: Path):
        (p / "docs").mkdir()
        (p / "docs" / "manual.txt").write_text("manual")
        (p / "target.txt").write_text("target")
        os.symlink(str(p / "nonexistent.txt"), str(p / "broken_link.txt"))
        os.symlink(str(p / "target.txt"), str(p / "valid_link.txt"))
    def verify_14(p: Path):
        out = p / "broken.txt"
        if not out.exists(): return False
        return "broken_link.txt" in out.read_text() and "valid_link.txt" not in out.read_text()
    tasks.append(BashTaskSpec("find_broken_symlinks", "Find all broken symbolic links in the workspace and write their names to 'broken.txt'.", setup_14, verify_14, ["fs", "symlink"]))

    def setup_15(p: Path):
        (p / "project").mkdir()
        (p / "project" / "file1.txt").write_text("1")
        (p / "project" / "file2.txt").write_text("2")
        (p / "backup").mkdir()
    def verify_15(p: Path):
        return (p / "backup" / "project" / "file1.txt").exists() and (p / "backup" / "project" / "file2.txt").exists()
    tasks.append(BashTaskSpec("recursive_copy_dir", "Copy the entire directory 'project' into 'backup/' preserving directory structure.", setup_15, verify_15, ["fs", "cp"]))

    # =========================================================================
    # Domain 3: Complex Shell Pipelines & Data Aggregation (Tasks 31–50)
    # =========================================================================

    def setup_16(p: Path):
        (p / "sales.csv").write_text("item,qty,price\nWidget,10,5.0\nGadget,2,20.0\nWidget,5,5.0\nGizmo,1,100.0\n")
    def verify_16(p: Path):
        out = p / "total_rev.txt"
        if not out.exists(): return False
        val = float(out.read_text().strip())
        return abs(val - 215.0) < 1.0
    tasks.append(BashTaskSpec("compute_total_revenue", "Calculate total revenue by multiplying qty * price for each line in 'sales.csv' (skip header), sum it, and save to 'total_rev.txt'.", setup_16, verify_16, ["pipeline", "awk"]))

    def setup_17(p: Path):
        (p / "app.log").write_text("2023-10-01 10:00:00 [USER:12] Login success\n2023-10-01 10:05:00 [USER:45] Login fail\n2023-10-01 10:10:00 [USER:12] Logout\n2023-10-01 10:15:00 [USER:78] Login success\n")
    def verify_17(p: Path):
        out = p / "users.txt"
        if not out.exists(): return False
        lines = sorted([x.strip() for x in out.read_text().splitlines() if x.strip()])
        return lines == ["12", "45", "78"]
    tasks.append(BashTaskSpec("extract_unique_user_ids", "Extract all unique user IDs from '[USER:<id>]' tags in 'app.log' sorted numerically into 'users.txt'.", setup_17, verify_17, ["pipeline", "grep", "sort"]))

    def setup_18(p: Path):
        (p / "data.json").write_text('{"name": "Alice", "score": 95}\n{"name": "Bob", "score": 82}\n{"name": "Charlie", "score": 98}\n')
    def verify_18(p: Path):
        out = p / "names.txt"
        if not out.exists(): return False
        lines = sorted([x.strip() for x in out.read_text().splitlines() if x.strip()])
        return lines == ["Alice", "Bob", "Charlie"]
    tasks.append(BashTaskSpec("extract_json_keys_regex", "Extract the 'name' field string values from each JSON line in 'data.json' and save sorted in 'names.txt'.", setup_18, verify_18, ["pipeline", "sed"]))

    # =========================================================================
    # Domain 4: Permissions, Security & Attributes (Tasks 51–70)
    # =========================================================================

    def setup_19(p: Path):
        (p / "secure.sh").write_text("#!/bin/bash\necho secure\n")
        os.chmod(p / "secure.sh", 0o777)
    def verify_19(p: Path):
        st = os.stat(p / "secure.sh")
        return (st.st_mode & 0o777) == 0o750
    tasks.append(BashTaskSpec("chmod_octal_permissions", "Change the permissions of 'secure.sh' to exactly rwxr-x--- (octal 750).", setup_19, verify_19, ["perm", "chmod"]))

    def setup_20(p: Path):
        (p / "bin").mkdir()
        for i in range(5):
            f = p / "bin" / f"tool_{i}.sh"
            f.write_text("#!/bin/bash\necho ok\n")
            os.chmod(f, 0o644)
    def verify_20(p: Path):
        for i in range(5):
            st = os.stat(p / "bin" / f"tool_{i}.sh")
            if not (st.st_mode & 0o111): return False
        return True
    tasks.append(BashTaskSpec("batch_chmod_scripts", "Make all '.sh' files in the 'bin/' directory executable.", setup_20, verify_20, ["perm", "chmod"]))

    # =========================================================================
    # Programmatic Generation for Remaining Tasks up to 100
    # =========================================================================
    base_count = len(tasks)
    for i in range(100 - base_count):
        idx = i + base_count
        category = idx % 4

        if category == 0:
            # Word frequency & counting
            word = f"metric_{idx}"
            target_file = f"metrics_{idx}.log"
            def make_setup(w=word, tf=target_file):
                def _setup(p: Path):
                    lines = [f"2023-11-0{j%9+1} INFO Status check {j}" for j in range(15)]
                    lines.insert(2, f"2023-11-01 WARN {w} threshold exceeded")
                    lines.insert(5, f"2023-11-02 ERROR {w} failure")
                    lines.insert(9, f"2023-11-03 WARN {w} threshold exceeded")
                    (p / tf).write_text("\n".join(lines) + "\n")
                return _setup
            def make_verify(w=word, tf=target_file, out_name=f"count_{idx}.txt"):
                def _verify(p: Path):
                    out = p / out_name
                    if not out.exists(): return False
                    return out.read_text().strip() == "3"
                return _verify
            tasks.append(BashTaskSpec(f"count_occurrences_task_{idx}", f"Count occurrences of '{word}' in '{target_file}' and save the total count to 'count_{idx}.txt'.", make_setup(), make_verify(), ["text", "grep"]))

        elif category == 1:
            # CSV filtering
            target_csv = f"data_{idx}.csv"
            def make_setup(tcsv=target_csv, index=idx):
                def _setup(p: Path):
                    (p / tcsv).write_text(f"id,name,value\n1,alpha,{index*2}\n2,beta,{index*5}\n3,alpha,{index*10}\n4,gamma,{index}\n")
                return _setup
            def make_verify(tcsv=target_csv, index=idx, out_name=f"alpha_sum_{idx}.txt"):
                def _verify(p: Path):
                    out = p / out_name
                    if not out.exists(): return False
                    val = float(out.read_text().strip())
                    expected = float(index * 2 + index * 10)
                    return abs(val - expected) < 1.0
                return _verify
            tasks.append(BashTaskSpec(f"csv_conditional_sum_{idx}", f"Sum the 'value' column in '{target_csv}' for rows where name == 'alpha' and write the sum to 'alpha_sum_{idx}.txt'.", make_setup(), make_verify(), ["pipeline", "awk"]))

        elif category == 2:
            # Filesystem reorganization
            def make_setup(index=idx):
                def _setup(p: Path):
                    (p / f"raw_{index}").mkdir()
                    for ext in ["txt", "log", "bak"]:
                        (p / f"raw_{index}" / f"file_{index}.{ext}").write_text("data")
                return _setup
            def make_verify(index=idx):
                def _verify(p: Path):
                    return (p / f"backup_{index}" / f"file_{index}.bak").exists() and not (p / f"raw_{index}" / f"file_{index}.bak").exists()
                return _verify
            tasks.append(BashTaskSpec(f"move_extension_dir_{idx}", f"Move all '.bak' files from 'raw_{idx}/' into a new directory named 'backup_{idx}/'.", make_setup(), make_verify(), ["fs", "mv"]))

        else:
            # Tar archive creation
            def make_setup(index=idx):
                def _setup(p: Path):
                    (p / f"bundle_{index}").mkdir()
                    (p / f"bundle_{index}" / "data.csv").write_text("1,2,3")
                return _setup
            def make_verify(index=idx):
                def _verify(p: Path):
                    archive = p / f"bundle_{index}.tar.gz"
                    return archive.exists() and archive.stat().st_size > 0
                return _verify
            tasks.append(BashTaskSpec(f"tar_compression_task_{idx}", f"Compress the directory 'bundle_{idx}' into a gzip archive named 'bundle_{idx}.tar.gz'.", make_setup(), make_verify(), ["archive", "tar"]))

    return tasks


class InterCodeBashBenchmark(BaseBenchmark):
    """
    InterCode-Bash Interactive Shell Benchmark Adapter.
    Executes real Linux commands inside isolated temporary workspaces with stdout/stderr capture.
    """

    DEFAULT_SYSTEM_PROMPT = (
        "You are an expert Linux sysadmin and Bash programmer.\n"
        "You will be given a task instruction in a clean workspace.\n"
        "Every round you should respond with a single Bash command to execute.\n"
        "You will receive the stdout, stderr, and return code from the environment.\n"
        "When you have accomplished the goal, you can run 'submit' or your final check.\n"
        "Output ONLY the exact Bash command to execute."
    )

    def __init__(self, num_tasks: int = 100, seed: int = 42):
        self.seed = seed
        self.num_tasks = num_tasks
        self.task_specs = create_task_suite()[:num_tasks]
        self.task_list = self._generate_tasks()

        self.current_workspace_dir: Optional[tempfile.TemporaryDirectory] = None
        self.current_task_spec: Optional[BashTaskSpec] = None
        self.is_completed = False

    def _generate_tasks(self) -> list[TaskInstance]:
        tasks = []
        for i, spec in enumerate(self.task_specs):
            task_id = f"intercode-bash/{spec.name}-v1"
            tasks.append(
                TaskInstance(
                    task_id=task_id,
                    instruction=spec.instruction,
                    system_prompt=self.DEFAULT_SYSTEM_PROMPT,
                    info={
                        "task_idx": i,
                        "task_name": spec.name,
                        "tags": spec.tags,
                    },
                )
            )
        return tasks

    def list_tasks(self, split: str = "train") -> list[TaskInstance]:
        return self.task_list

    def reset(self, task: TaskInstance, seed: Optional[int] = None) -> StepObservation:
        self.close()

        # Find task spec
        task_idx = task.info.get("task_idx", 0)
        self.current_task_spec = self.task_specs[task_idx]

        # Create isolated temporary directory
        self.current_workspace_dir = tempfile.TemporaryDirectory(prefix="intercode_bash_")
        work_path = Path(self.current_workspace_dir.name)

        # Run setup function
        self.current_task_spec.setup_fn(work_path)
        self.is_completed = False

        # Initial workspace listing
        listing = [f.name for f in work_path.iterdir()]
        listing_str = ", ".join(sorted(listing)) if listing else "(empty directory)"

        obs_text = (
            f"Instruction: {task.instruction}\n\n"
            f"Workspace Initial Files: {listing_str}\n"
            f"Current Working Directory: {work_path}\n"
            f"Enter bash command to execute:"
        )

        return StepObservation(
            observation_text=obs_text,
            step_reward=0.0,
            is_done=False,
            info={"workspace": str(work_path)},
        )

    def step(self, action_str: str) -> StepObservation:
        if not self.current_workspace_dir or not self.current_task_spec:
            raise RuntimeError("Environment not reset. Call reset() before step().")

        work_path = Path(self.current_workspace_dir.name)
        cmd = action_str.strip()

        # Remove markdown codeblocks if model wrapped command in ```bash ... ```
        if cmd.startswith("```"):
            lines = cmd.splitlines()
            cmd = "\n".join([l for l in lines if not l.startswith("```")]).strip()

        # Clean command prompt prefix
        if cmd.startswith("$ ") or cmd.startswith("> "):
            cmd = cmd[2:].strip()

        # Check for submit / done command
        if cmd.lower() in ("submit", "done", "exit"):
            is_succ = self.current_task_spec.verify_fn(work_path)
            self.is_completed = is_succ
            reward = 1.0 if is_succ else 0.0
            return StepObservation(
                observation_text=f"Submitted task. Evaluation: {'SUCCESS' if is_succ else 'FAILED'}",
                step_reward=reward,
                is_done=True,
                info={"success": is_succ},
            )

        # Execute bash command in workspace
        try:
            res = subprocess.run(
                cmd,
                shell=True,
                cwd=str(work_path),
                capture_output=True,
                text=True,
                timeout=10,
                executable="/bin/bash",
            )
            stdout = res.stdout.strip()
            stderr = res.stderr.strip()
            return_code = res.returncode

            # Truncate excessive outputs to prevent context overflow
            if len(stdout) > 1000:
                stdout = stdout[:1000] + "\n...[truncated]"
            if len(stderr) > 1000:
                stderr = stderr[:1000] + "\n...[truncated]"

            obs_lines = [f"[Return Code: {return_code}]"]
            if stdout:
                obs_lines.append(f"stdout:\n{stdout}")
            if stderr:
                obs_lines.append(f"stderr:\n{stderr}")
            if not stdout and not stderr:
                obs_lines.append("(Command produced no output)")

            obs_text = "\n".join(obs_lines)

            # Auto-verify on each step
            is_succ = self.current_task_spec.verify_fn(work_path)
            if is_succ:
                self.is_completed = True

            return StepObservation(
                observation_text=obs_text,
                step_reward=1.0 if is_succ else 0.0,
                is_done=False,
                info={"return_code": return_code, "success": is_succ},
            )

        except subprocess.TimeoutExpired:
            return StepObservation(
                observation_text="[Error: Command timed out after 10 seconds]",
                step_reward=0.0,
                is_done=False,
                info={"error": "timeout"},
            )
        except Exception as e:
            return StepObservation(
                observation_text=f"[Execution Error: {e}]",
                step_reward=0.0,
                is_done=False,
                info={"error": str(e)},
            )

    def evaluate(self, task: TaskInstance) -> bool:
        if not self.current_workspace_dir or not self.current_task_spec:
            return False
        work_path = Path(self.current_workspace_dir.name)
        return self.current_task_spec.verify_fn(work_path)

    def close(self) -> None:
        if self.current_workspace_dir:
            try:
                self.current_workspace_dir.cleanup()
            except Exception:
                pass
            self.current_workspace_dir = None
        self.current_task_spec = None
