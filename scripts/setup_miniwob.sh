#!/usr/bin/env bash
set -e

echo "=== [MiniWoB++] Installing Google Chrome & Headless Browser on RunPod ==="

if command -v apt-get &> /dev/null; then
    apt-get update && apt-get install -y wget xvfb git git-lfs tmux ca-certificates
    
    # Download and install official Google Chrome (automatically pulls all shared libraries)
    wget -q https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
    apt-get install -y ./google-chrome-stable_current_amd64.deb
    rm -f google-chrome-stable_current_amd64.deb
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
echo "  xvfb-run -a python scripts/collect_miniwob.py --model Qwen/Qwen2.5-Coder-7B-Instruct --all_envs --n_rollouts 3 --max_steps 8 --temperature 0.7 --layers 4 10 16 18 20 22 24 27 --anchors post_gen pre_gen thought_end --output_dir artifacts/miniwob/qwen2.5-7b/qwen2.5-7b"
