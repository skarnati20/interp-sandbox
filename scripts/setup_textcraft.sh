#!/usr/bin/env bash
set -e

echo "=== [TextCraft] Installing System Dependencies on RunPod ==="
apt-get update && apt-get install -y \
    git \
    git-lfs

echo "=== Installing Core Sandbox Requirements ==="
pip install --upgrade pip
pip install -r requirements.txt

echo "=== Installing TextCraft Benchmark Requirements ==="
pip install \
    gymnasium>=0.29.0

echo "=== TextCraft Setup Complete! ==="
echo "To run TextCraft Qwen2.5-Coder-7B data collection:"
echo "  python scripts/collect_textcraft.py --model Qwen/Qwen2.5-Coder-7B-Instruct --num_tasks 50 --n_rollouts 5 --min_depth 2 --max_steps 12 --temperature 0.7 --layers 20 --anchors post_gen pre_gen --output_dir artifacts/textcraft/qwen2.5-7b/qwen2.5-7b"
