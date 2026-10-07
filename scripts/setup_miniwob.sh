#!/usr/bin/env bash
set -e

echo "=== [MiniWoB++] Installing Headless Browser Dependencies on RunPod ==="
if command -v apt-get &> /dev/null; then
    apt-get update && apt-get install -y \
        chromium-browser || apt-get install -y chromium || true
    apt-get install -y \
        chromium-chromedriver || apt-get install -y chromedriver || true
    apt-get install -y \
        xvfb \
        git \
        git-lfs \
        tmux
fi

echo "=== Installing Core Sandbox Requirements ==="
pip install --upgrade pip
pip install -r requirements.txt

echo "=== Installing MiniWoB++ Benchmark Requirements ==="
pip install \
    gymnasium>=0.29.0 \
    miniwob>=1.0.0 \
    selenium>=4.15.0

echo "=== MiniWoB++ Setup Complete! ==="
echo "To run MiniWoB++ Qwen2.5-Coder-7B data collection:"
echo "  xvfb-run -a python scripts/collect_miniwob.py --model Qwen/Qwen2.5-Coder-7B-Instruct --num_tasks 50 --n_rollouts 5 --max_steps 8 --temperature 0.7 --layers 20 --anchors post_gen pre_gen --output_dir artifacts/miniwob/qwen2.5-7b/qwen2.5-7b"
