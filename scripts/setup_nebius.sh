#!/usr/bin/env bash
set -e

echo "=== [Nebius SWE-Agent] Installing System Utilities ==="
if command -v apt-get &> /dev/null; then
    apt-get update && apt-get install -y \
        git \
        git-lfs \
        tmux \
        curl \
        ca-certificates
fi

echo "=== Installing Python Core Requirements ==="
pip install --upgrade pip
pip install -r requirements.txt

echo "=== Verifying Hugging Face & PyArrow Tooling ==="
python -c "import huggingface_hub, pyarrow.parquet, torch, transformers; print('Dependencies verified successfully.')"

echo ""
echo "=== Nebius SWE-Agent Trajectory Setup Complete! ==="
echo "To run trace replay with Qwen2.5-Coder-7B-Instruct:"
echo "  python scripts/collect_nebius.py \\"
echo "      --model Qwen/Qwen2.5-Coder-7B-Instruct \\"
echo "      --max_trajectories 50 \\"
echo "      --anchors post_gen pre_gen \\"
echo "      --output_dir artifacts/nebius/qwen2.5-7b"
