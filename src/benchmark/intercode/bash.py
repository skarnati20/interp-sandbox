"""
InterCode-Bash Benchmark: Interactive Linux Command-Line and Shell Environment.
Implementation based on InterCode (Yang et al., NeurIPS 2023) and 'Do Agents Know When They Succeed' (2026).
"""

from dataclasses import dataclass, field
import os
from pathlib import Path
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
    Creates 50 standard multi-step InterCode-Bash tasks across 5 core domains:
    1. Text Processing & Regex (grep, sed, awk, cut, sort, uniq)
    2. File & Directory Management (mkdir, cp, mv, find, tree)
    3. Log Parsing & Data Extraction (wc, tr, pipeline chains)
    4. Permissions & File Attributes (chmod, touch, stat)
    5. Archiving & Compression (tar, gzip, zip)
    """
    tasks: list[BashTaskSpec] = []

    # --- Domain 1: Text Processing & Regex ---
    def setup_grep_emails(p: Path):
        (p / "data.txt").write_text(
            "Alice alice@example.com 24\nBob invalid-email 30\nCharlie charlie@work.org 29\nDavid david@test.com 45\n"
        )
    def verify_grep_emails(p: Path):
        out = p / "emails.txt"
        if not out.exists(): return False
        content = sorted([x.strip() for x in out.read_text().splitlines() if x.strip()])
        return content == ["alice@example.com", "charlie@work.org", "david@test.com"]
    tasks.append(BashTaskSpec("extract_emails", "Extract all valid email addresses from 'data.txt' into 'emails.txt', one per line.", setup_grep_emails, verify_grep_emails, ["text", "regex"]))

    def setup_count_errors(p: Path):
        (p / "server.log").write_text(
            "INFO: start\nERROR: database timeout\nINFO: retry\nERROR: connection refused\nWARN: high mem\nERROR: disk full\n"
        )
    def verify_count_errors(p: Path):
        out = p / "error_count.txt"
        if not out.exists(): return False
        return out.read_text().strip() == "3"
    tasks.append(BashTaskSpec("count_errors", "Count the number of lines containing 'ERROR' in 'server.log' and write the count to 'error_count.txt'.", setup_count_errors, verify_count_errors, ["text", "wc"]))

    def setup_replace_word(p: Path):
        (p / "config.txt").write_text("host=localhost\nport=8080\nenv=development\nmode=development\n")
    def verify_replace_word(p: Path):
        c = (p / "config.txt").read_text()
        return "development" not in c and "env=production" in c and "mode=production" in c
    tasks.append(BashTaskSpec("replace_env", "Replace all occurrences of 'development' with 'production' in 'config.txt'.", setup_replace_word, verify_replace_word, ["text", "sed"]))

    def setup_sort_scores(p: Path):
        (p / "scores.txt").write_text("Alice 85\nBob 92\nCharlie 78\nDavid 95\nEve 88\n")
    def verify_sort_scores(p: Path):
        out = p / "sorted.txt"
        if not out.exists(): return False
        lines = [x.strip() for x in out.read_text().splitlines() if x.strip()]
        expected = ["David 95", "Bob 92", "Eve 88", "Alice 85", "Charlie 78"]
        return lines == expected
    tasks.append(BashTaskSpec("sort_scores", "Sort 'scores.txt' numerically in descending order by the second column (score) and save to 'sorted.txt'.", setup_sort_scores, verify_sort_scores, ["text", "sort"]))

    def setup_unique_ips(p: Path):
        (p / "access.log").write_text("192.168.1.1\n10.0.0.1\n192.168.1.1\n172.16.0.1\n10.0.0.1\n192.168.1.2\n")
    def verify_unique_ips(p: Path):
        out = p / "unique_ips.txt"
        if not out.exists(): return False
        lines = sorted([x.strip() for x in out.read_text().splitlines() if x.strip()])
        return lines == ["10.0.0.1", "172.16.0.1", "192.168.1.1", "192.168.1.2"]
    tasks.append(BashTaskSpec("unique_ips", "Extract all unique IP addresses from 'access.log' sorted alphabetically into 'unique_ips.txt'.", setup_unique_ips, verify_unique_ips, ["text", "uniq"]))

    # --- Domain 2: File & Directory Management ---
    def setup_create_nested_dirs(p: Path):
        pass
    def verify_create_nested_dirs(p: Path):
        return (p / "src" / "components" / "utils").is_dir() and (p / "src" / "components" / "utils" / "index.js").is_file()
    tasks.append(BashTaskSpec("create_nested_dirs", "Create directory 'src/components/utils' and create an empty file 'index.js' inside it.", setup_create_nested_dirs, verify_create_nested_dirs, ["fs", "mkdir"]))

    def setup_move_jpg_files(p: Path):
        (p / "raw").mkdir()
        (p / "images").mkdir()
        (p / "raw" / "photo1.jpg").write_text("img1")
        (p / "raw" / "photo2.jpg").write_text("img2")
        (p / "raw" / "doc.pdf").write_text("pdf")
    def verify_move_jpg_files(p: Path):
        return (p / "images" / "photo1.jpg").exists() and (p / "images" / "photo2.jpg").exists() and (p / "raw" / "doc.pdf").exists() and not (p / "raw" / "photo1.jpg").exists()
    tasks.append(BashTaskSpec("move_images", "Move all '.jpg' files from 'raw/' into 'images/'. Leave other files in 'raw/'.", setup_move_jpg_files, verify_move_jpg_files, ["fs", "mv"]))

    def setup_find_large_files(p: Path):
        (p / "a.bin").write_bytes(b"0" * 1024 * 50)   # 50 KB
        (p / "b.bin").write_bytes(b"0" * 1024 * 200)  # 200 KB
        (p / "c.bin").write_bytes(b"0" * 1024 * 500)  # 500 KB
    def verify_find_large_files(p: Path):
        out = p / "large_files.txt"
        if not out.exists(): return False
        content = out.read_text().strip()
        return "b.bin" in content and "c.bin" in content and "a.bin" not in content
    tasks.append(BashTaskSpec("find_large_files", "Find all files in the current directory larger than 100KB and save their names/paths to 'large_files.txt'.", setup_find_large_files, verify_find_large_files, ["fs", "find"]))

    def setup_delete_temp_files(p: Path):
        (p / "main.py").write_text("print('hello')")
        (p / "main.py.tmp").write_text("temp")
        (p / "cache.tmp").write_text("temp")
        (p / "sub").mkdir()
        (p / "sub" / "data.tmp").write_text("temp")
    def verify_delete_temp_files(p: Path):
        return (p / "main.py").exists() and not (p / "main.py.tmp").exists() and not (p / "cache.tmp").exists() and not (p / "sub" / "data.tmp").exists()
    tasks.append(BashTaskSpec("delete_tmp_files", "Recursively find and delete all files ending with '.tmp' in the directory tree.", setup_delete_temp_files, verify_delete_temp_files, ["fs", "rm"]))

    # --- Domain 3: Data Parsing & Shell Pipelines ---
    def setup_csv_column(p: Path):
        (p / "users.csv").write_text("id,name,role\n1,Alice,Admin\n2,Bob,User\n3,Charlie,Admin\n4,David,User\n")
    def verify_csv_column(p: Path):
        out = p / "admins.txt"
        if not out.exists(): return False
        lines = sorted([x.strip() for x in out.read_text().splitlines() if x.strip()])
        return lines == ["Alice", "Charlie"]
    tasks.append(BashTaskSpec("filter_csv_admins", "Extract the names (column 2) of all users with role 'Admin' from 'users.csv' and save to 'admins.txt', one per line.", setup_csv_column, verify_csv_column, ["pipeline", "awk"]))

    def setup_word_frequency(p: Path):
        (p / "story.txt").write_text("apple banana apple orange banana apple\n")
    def verify_word_frequency(p: Path):
        out = p / "top_word.txt"
        if not out.exists(): return False
        return "apple" in out.read_text().strip().lower() and "3" in out.read_text()
    tasks.append(BashTaskSpec("top_word_frequency", "Find the most frequent word and its count in 'story.txt' and save to 'top_word.txt'.", setup_word_frequency, verify_word_frequency, ["pipeline", "uniq"]))

    def setup_combine_files(p: Path):
        (p / "part1.txt").write_text("First line\n")
        (p / "part2.txt").write_text("Second line\n")
        (p / "part3.txt").write_text("Third line\n")
    def verify_combine_files(p: Path):
        out = p / "combined.txt"
        if not out.exists(): return False
        return out.read_text() == "First line\nSecond line\nThird line\n"
    tasks.append(BashTaskSpec("combine_files", "Concatenate 'part1.txt', 'part2.txt', and 'part3.txt' in order into 'combined.txt'.", setup_combine_files, verify_combine_files, ["fs", "cat"]))

    # --- Domain 4: Permissions & Attributes ---
    def setup_chmod_executable(p: Path):
        (p / "script.sh").write_text("#!/bin/bash\necho 'hello'\n")
        os.chmod(p / "script.sh", 0o644)
    def verify_chmod_executable(p: Path):
        st = os.stat(p / "script.sh")
        return bool(st.st_mode & 0o111)
    tasks.append(BashTaskSpec("make_script_executable", "Make 'script.sh' executable for the user/owner.", setup_chmod_executable, verify_chmod_executable, ["perm", "chmod"]))

    def setup_create_symlink(p: Path):
        (p / "target.txt").write_text("Hello Target")
    def verify_create_symlink(p: Path):
        link = p / "link.txt"
        return link.is_symlink() and link.resolve() == (p / "target.txt").resolve()
    tasks.append(BashTaskSpec("create_symlink", "Create a symbolic link named 'link.txt' pointing to 'target.txt'.", setup_create_symlink, verify_create_symlink, ["fs", "ln"]))

    # --- Domain 5: Archiving & Compression ---
    def setup_tar_archive(p: Path):
        (p / "project").mkdir()
        (p / "project" / "app.py").write_text("app")
        (p / "project" / "util.py").write_text("util")
    def verify_tar_archive(p: Path):
        tar_file = p / "project.tar.gz"
        return tar_file.exists() and tar_file.stat().st_size > 0
    tasks.append(BashTaskSpec("create_tar_archive", "Compress the 'project' directory into an archive named 'project.tar.gz'.", setup_tar_archive, verify_tar_archive, ["archive", "tar"]))

    def setup_unzip_archive(p: Path):
        proj = p / "temp_proj"
        proj.mkdir()
        (proj / "extracted.txt").write_text("Success!")
        shutil.make_archive(str(p / "bundle"), 'zip', str(proj))
        shutil.rmtree(str(proj))
    def verify_unzip_archive(p: Path):
        return (p / "extracted.txt").exists() and "Success!" in (p / "extracted.txt").read_text()
    tasks.append(BashTaskSpec("unzip_bundle", "Extract the contents of 'bundle.zip' into the current working directory.", setup_unzip_archive, verify_unzip_archive, ["archive", "unzip"]))

    # Programmatic variations to complete 50 tasks
    base_count = len(tasks)
    for i in range(50 - base_count):
        idx = i + base_count
        word = f"keyword_{idx}"
        target_file = f"log_{idx}.txt"

        def make_setup(w=word, tf=target_file):
            def _setup(p: Path):
                lines = [f"Line {j} with other info" for j in range(10)]
                lines.insert(3, f"Line 3 with {w} match")
                lines.insert(7, f"Line 7 with {w} match")
                (p / tf).write_text("\n".join(lines) + "\n")
            return _setup

        def make_verify(w=word, tf=target_file, out_name=f"matches_{idx}.txt"):
            def _verify(p: Path):
                out = p / out_name
                if not out.exists(): return False
                lines = [x.strip() for x in out.read_text().splitlines() if x.strip()]
                return len(lines) == 2 and all(w in l for l in lines)
            return _verify

        tasks.append(
            BashTaskSpec(
                name=f"filter_log_task_{idx}",
                instruction=f"Find all lines containing '{word}' in '{target_file}' and write them to 'matches_{idx}.txt'.",
                setup_fn=make_setup(word, target_file),
                verify_fn=make_verify(word, target_file, f"matches_{idx}.txt"),
                tags=["pipeline", "grep"],
            )
        )

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

    def __init__(self, seed: int = 42):
        self.seed = seed
        self.task_specs = create_task_suite()
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
