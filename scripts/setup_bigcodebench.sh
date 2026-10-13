#!/usr/bin/env bash
set -e

echo "=== [BigCodeBench] Installing System Utilities on RunPod ==="
if command -v apt-get &> /dev/null; then
    apt-get update && apt-get install -y \
        git \
        git-lfs \
        tmux \
        curl \
        ca-certificates \
        build-essential
fi

echo "=== Installing Python Core Sandbox Requirements ==="
pip install --upgrade pip
pip install -r requirements.txt

echo "=== Installing BigCodeBench Evaluation Requirements ==="
pip install \
    scipy \
    scikit-learn \
    matplotlib \
    requests \
    tqdm \
    rich

echo "=== BigCodeBench Setup Complete! ==="
echo "To run BigCodeBench Qwen2.5-Coder-7B data collection:"
echo "  python scripts/collect_bigcodebench.py --model Qwen/Qwen2.5-Coder-7B-Instruct --subset hard --num_tasks 50 --n_rollouts 5 --max_steps 5 --temperature 0.7 --layers 20 --anchors post_gen pre_gen --output_dir artifacts/bigcodebench/qwen2.5-7b/qwen2.5-7b"
