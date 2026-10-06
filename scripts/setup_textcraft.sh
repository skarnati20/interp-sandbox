#!/usr/bin/env bash
set -e

echo "=== [TextCraft] Installing Core Sandbox Requirements on RunPod ==="
apt-get update && apt-get install -y \
    git \
    git-lfs

echo "=== Installing Python Requirements ==="
pip install --upgrade pip
pip install -r requirements.txt

echo "=== TextCraft Setup Complete! ==="
echo "To run TextCraft Qwen2.5-Coder-7B data collection:"
echo "  python scripts/collect_textcraft.py --model Qwen/Qwen2.5-Coder-7B-Instruct --num_tasks 50 --n_rollouts 5 --min_depth 2 --max_steps 12 --temperature 0.7 --layers 20 --anchors post_gen pre_gen --output_dir artifacts/textcraft/qwen2.5-7b/qwen2.5-7b"
