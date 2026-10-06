# Interp Sandbox

A research sandbox for **Agentic Mechanistic Interpretability & Representation Engineering**.

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
