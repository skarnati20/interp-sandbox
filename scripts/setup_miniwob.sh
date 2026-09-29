#!/usr/bin/env bash
set -e

echo "=== [MiniWoB++] Installing Headless Browser Dependencies on RunPod ==="
apt-get update && apt-get install -y \
    chromium-browser \
    chromium-chromedriver \
    xvfb \
    git \
    git-lfs

echo "=== Installing Core Sandbox Requirements ==="
pip install --upgrade pip
pip install -r requirements.txt

echo "=== Installing MiniWoB++ Benchmark Requirements ==="
pip install \
    gymnasium>=0.29.0 \
    miniwob>=1.0.0 \
    selenium>=4.15.0

echo "=== MiniWoB++ Setup Complete! ==="
echo "To run MiniWoB++ 7B data collection:"
echo "  xvfb-run -a python scripts/collect_miniwob.py --model Qwen/Qwen2.5-Coder-7B-Instruct --num_episodes 50"
