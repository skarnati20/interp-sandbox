#!/usr/bin/env bash
set -e

echo "=== [TextCraft Steering] Installing System Dependencies on RunPod ==="
if command -v apt-get &> /dev/null; then
    apt-get update && apt-get install -y \
        git \
        git-lfs \
        tmux \
        curl
fi

echo "=== Installing Python Sandbox & Probing Requirements ==="
pip install --upgrade pip
pip install -r requirements.txt
pip install scikit-learn

echo "=== Verifying PyTorch CUDA Environment ==="
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()} | Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"CPU\"}')"

echo ""
echo "=== TextCraft Steering Setup Complete! ==="
echo ""
echo "1. Run Quick Smoke Test (2 tasks per fold, first turn only):"
echo "  python scripts/steer_textcraft_kfold.py --max_eval_tasks_per_fold 2 --eval_max_turns 1 --output_dir data/runs/test_smoke"
echo ""
echo "2. Train on First Turn Only, Evaluate Full Multi-Turn Tasks:"
echo "  python scripts/steer_textcraft_kfold.py --train_max_turns 1 --output_dir data/runs/tc_steer_train_t1"
echo ""
echo "3. Full 5-Fold Steering (All Turns, Full Dynamic Budget):"
echo "  python scripts/steer_textcraft_kfold.py --model Qwen/Qwen2.5-Coder-7B-Instruct --layer_id 20 --threshold 0.5 --alpha 1.0 --output_dir data/runs/textcraft_steered_cv"
