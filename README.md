# Interp Sandbox

A research sandbox for **Agentic Mechanistic Interpretability & Representation Engineering**.

## Repository Structure

```text
interp-sandbox/
├── src/                               # REUSABLE CORE LIBRARY
│   ├── activations.py                 # Forward pass & residual-stream activation extraction
│   ├── store.py                       # Safetensors & Parquet trajectory serialization
│   ├── runner.py                      # Multi-turn agent interaction loop
│   └── benchmark/
│       ├── __init__.py
│       ├── base.py                    # Universal BaseBenchmark interface
│       └── miniwob.py                 # MiniWoB++ adapter (DOM formatter & action parser)
├── scripts/                           # BENCHMARK-SPECIFIC EXECUTABLES
│   ├── setup_miniwob.sh               # RunPod / Linux setup script for MiniWoB++
│   └── collect_miniwob.py             # MiniWoB++ data collection runner
├── test_local.py                      # Fast local pipeline verification script
├── requirements.txt                   # Minimal Python dependencies
└── README.md
```

## Quick Start

You can verify the pipeline with the MiniWoB++ benchmark as so:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run mock dry-run (no GPU / model download required)
python test_local.py

# 3. (Optional) Run with local small model (Qwen-0.5B)
python test_local.py --real-model
```
