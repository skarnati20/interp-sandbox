# Interp Sandbox

A research sandbox for **Agentic Mechanistic Interpretability & Representation Engineering**.

## Temporal probing

`TemporalProbe` uses the same `fit`, `predict`, and `evaluate` interface as the
other probes. It fits standardized logistic regression on the concatenation of
the current activation, the running mean through the current turn, and the
change from the previous turn (zero on the first turn).

```python
from src.store import ShardReader
from src.probe.probes.temporal_probe import TemporalProbe
from src.probe.train import train_and_evaluate

data = ShardReader("artifacts/my_run/features").read_all(anchor="pre_gen", layer=20)
result = train_and_evaluate(
    TemporalProbe(c_reg=0.1), dataset=data,
    group_by="task_id", eval_size=0.2, seed=42,
)
print(result["eval_metrics"])
```

Supply a single model's compatible activation features with `episode_id` and
zero-based `round` (or `step_idx`) metadata. Each episode/anchor must contain a
complete prefix: missing or duplicate turns raise an error. Rows may be shuffled;
predictions are returned in input order. Features never use future turns or labels.
For online prediction, pass the full prefix so far and take its latest prediction;
the probe keeps no history between calls. Scores represent success, so use
`1 - probe.predict(data, return_proba=True)` for failure scores.

Use task/episode-grouped splits and leave `target_success_ratio=None`: the existing
row-subsampling balancer can remove required history. Early-turn cutoffs are
supported because they preserve prefixes. As with the other probes, evaluation
returns row-level metrics, not episode-level early-warning metrics. This adds an
architecture option; it does not establish improved performance or transfer.

## RunPod Instructions

Clone the repo:

```bash
cd /workspace
git clone https://github.com/skarnati20/interp-sandbox.git
cd interp-sandbox
```

Run desired setup script:

```bash
./scripts/setup_textcraft.sh
```

Start `tmux` sessions:

```bash
tmux new -s textcraft
```

Run benchmark script to generate activations:

```bash
python scripts/collect_textcraft.py \
  --model Qwen/Qwen2.5-Coder-7B-Instruct \
  --num_tasks 50 \
  --n_rollouts 5 \
  --max_steps 16 \
  --temperature 0.7 \
  --layers 20 \
  --anchors post_gen pre_gen \
  --output_dir artifacts/textcraft/qwen2.5-7b/qwen2.5-7b
```

Upload to HuggingFace:

```bash
python scripts/export.py \
  --data_dir artifacts/textcraft/qwen2.5-7b/qwen2.5-7b \
  --repo_id <YOUR_HF_USERNAME>/textcraft-qwen7b-activations \
  --token hf_YOUR_WRITE_TOKEN_HERE \
  --commit_message "Upload TextCraft Qwen2.5-Coder-7B residual activations"
```
