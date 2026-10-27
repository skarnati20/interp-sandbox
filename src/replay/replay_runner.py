from __future__ import annotations

from pathlib import Path
from typing import Optional
import uuid

from src.activations import ActivationExtractor, ExtractionResult
from src.store import ShardWriter, save_run_summary
from src.traces.base import BaseTraceAdapter, Trajectory, TurnStep


class ReplayRunner:
    """
    Replays agent trajectory traces through an ActivationExtractor without needing
    an active benchmark environment, records turn-level residual-stream hidden states
    via teacher-forced forward passes on recorded actions, and persists activation tensors
    with aligned metadata into consolidated shards ('feat.shard*.npz' and 'meta.shard*.jsonl').
    """

    def __init__(
        self,
        extractor: ActivationExtractor,
        trace_adapter: BaseTraceAdapter,
        output_dir: str | Path = "data/runs",
        layer_ids: Optional[list[int]] = None,
        anchors: Optional[list[str]] = None,
        shard_size: int = 2000,
        verbose: bool = True,
    ):
        self.extractor = extractor
        self.trace_adapter = trace_adapter
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.layer_ids = layer_ids
        self.anchors = anchors or ["post_gen", "pre_gen"]
        self.shard_size = shard_size
        self.verbose = verbose

        # Sharded storage writer for features
        self.features_dir = self.output_dir / "features"
        self.shard_writer = ShardWriter(
            output_dir=self.features_dir,
            shard_size=shard_size,
            layer_ids=layer_ids,
        )

    def get_trajectories(
        self,
        limit: Optional[int] = None,
    ) -> list[Trajectory]:
        return list(self.trace_adapter.iter_trajectories(max_trajectories=limit))

    def run_trajectory(
        self,
        trajectory: Trajectory,
        max_steps: Optional[int] = None,
        task_idx: Optional[int] = None,
        episode_id: Optional[str] = None,
    ) -> dict:
        t_idx = task_idx if task_idx is not None else trajectory.metadata.get("task_idx", 0)
        episode_id = episode_id or f"{trajectory.task_id}_{uuid.uuid4().hex[:6]}"

        turns: list[TurnStep] = trajectory.turns
        if max_steps is not None:
            turns = turns[:max_steps]

        n_turns = len(turns)
        is_success = bool(trajectory.is_success)
        success_int = trajectory.is_success

        for step_idx, turn in enumerate(turns):
            chat_messages = turn.to_chat_messages()

            if chat_messages and chat_messages[-1]["role"] == "assistant":
                prompt_messages = chat_messages[:-1]
                recorded_action = chat_messages[-1]["content"]
            else:
                prompt_messages = chat_messages
                recorded_action = turn.action_text or ""

            # Extract residual-stream activations via teacher-forcing
            result: ExtractionResult = self.extractor.step_forward_completion(
                prompt_messages, completion_text=recorded_action
            )

            if self.verbose:
                first_line = recorded_action.split("\n")[0] if "\n" in recorded_action else recorded_action
                print(f"    [Step {step_idx + 1}/{n_turns}] Action: '{first_line[:70]}'")

            step_meta = {
                "episode_id": episode_id,
                "task_id": trajectory.task_id,
                "task_idx": t_idx,
                "round": step_idx,
                "step_idx": step_idx,
                "action": recorded_action,
                "recorded_action": recorded_action,
                "reward": 0.0,
                "is_done": (step_idx == n_turns - 1),
                "prompt_tokens": result.prompt_tokens,
                "completion_tokens": result.completion_tokens,
                "success": success_int,
                "n_rounds": n_turns,
                "hit_max_rounds": (
                    max_steps is not None
                    and n_turns >= max_steps
                    and not is_success
                ),
            }
            if "model_name" in trajectory.metadata:
                step_meta["model_name"] = trajectory.metadata["model_name"]

            # Immediately stream anchor feature vectors to shard writer to prevent holding full hidden states in RAM
            for anchor in self.anchors:
                anchor_meta = dict(step_meta)
                anchor_meta["anchor"] = anchor
                feat_vec = result.get_anchor_state(anchor=anchor, layer_ids=self.layer_ids)
                self.shard_writer.add_step(feat_vec, anchor_meta)

            # Explicitly drop the result to free intermediate tensor memory immediately
            del result

        return {
            "episode_id": episode_id,
            "task_id": trajectory.task_id,
            "task_idx": t_idx,
            "is_success": is_success,
            "total_reward": 0.0,
            "num_steps": n_turns,
            "hit_max_rounds": (
                max_steps is not None
                and n_turns >= max_steps
                and not is_success
            ),
        }

    def run_replay(
        self,
        max_trajectories: Optional[int] = None,
        max_steps: Optional[int] = None,
    ) -> list[dict]:
        trajectories = self.get_trajectories(limit=max_trajectories)
        results = []
        summary_path = self.output_dir / "run_summary.parquet"

        if self.verbose:
            print(
                f"Starting trace replay for {len(trajectories)} trajectories..."
            )

        for i, traj in enumerate(trajectories):
            task_idx = traj.metadata.get("task_idx", i)
            if self.verbose:
                print(
                    f"[{i + 1}/{len(trajectories)}] Replaying trajectory {traj.task_id}..."
                )
            result = self.run_trajectory(
                trajectory=traj,
                max_steps=max_steps,
                task_idx=task_idx,
            )
            results.append(result)
            status = "SUCCESS" if result["is_success"] else "FAILED"
            if self.verbose:
                print(f"  Result: {status} (Steps: {result['num_steps']})")

            # Flush current shard buffer so completed trajectory activations are immediately written to disk
            self.shard_writer.flush()
            save_run_summary(summary_path, results)

        # Finalize shard writer
        self.shard_writer.close()

        success_count = sum(1 for r in results if r["is_success"])
        if self.verbose and results:
            print(
                f"\nReplay Complete! Success Rate: {(success_count / len(results)) * 100:.1f}% "
                f"({success_count}/{len(results)})"
            )
            print(f"Saved feature shards to {self.features_dir}")
            print(f"Saved run summary to {summary_path}")

        return results
