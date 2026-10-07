#!/usr/bin/env bash
set -e

echo "=== [InterCode-Bash] Installing Linux Sandbox & Utilities on RunPod ==="
if command -v apt-get &> /dev/null; then
    apt-get update && apt-get install -y \
        git \
        git-lfs \
        tmux \
        tree \
        zip \
        unzip \
        tar \
        gzip \
        gawk \
        sed \
        grep \
        coreutils \
        bsdmainutils \
        procps
fi

echo "=== Installing Python Requirements ==="
pip install --upgrade pip
pip install -r requirements.txt

echo "=== InterCode-Bash Setup Complete! ==="
echo "To run InterCode-Bash Qwen2.5-Coder-7B data collection:"
echo "  python scripts/collect_intercode_bash.py --model Qwen/Qwen2.5-Coder-7B-Instruct --num_tasks 50 --n_rollouts 5 --max_steps 8 --temperature 0.7 --layers 20 --anchors post_gen pre_gen --output_dir artifacts/intercode_bash/qwen2.5-7b/qwen2.5-7b"
