# Interp Sandbox

A research sandbox for **Agentic Mechanistic Interpretability & Representation Engineering**.

## Repo Structure

```text
interp-sandbox/
├── src/
│   ├── activations.py         # Forward pass & full-sequence hidden state extraction
│   ├── store.py               # Memory-mapped Safetensors & Parquet serialization
│   ├── runner.py              # Multi-turn agent interaction loop
│   └── benchmark/
│       ├── __init__.py
│       ├── base.py            # Universal BaseBenchmark interface
│       └── ...                # Other benchmarks which implement BaseBenchmark
├── test_local.py              # Fast local pipeline verification script
├── requirements.txt           # Python dependencies
└── README.md
```

## Quick Start

You can verify the pipeline with the MiniWob++ benchmark as so:

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run mock dry-run (no GPU / model download required)
python test_local.py

# 3. (Optional) Run with local small model (Qwen-0.5B)
python test_local.py --real-model
```