# Two models, one TextCraft manifest

These examples run from the repository root with Python 3.10+ and
`python -m pip install -r requirements.txt`. Model collection requires your
chosen compute environment; the local regression tests use fixture extractors.

## 1. Create the manifest once

```python
from src.benchmark.textcraft import TextCraftBenchmark
from src.splits import create_split_manifest, save_split_manifest

benchmark = TextCraftBenchmark(num_tasks=50, seed=42)
catalog = benchmark.task_catalog(grouping="crafting_goal")
manifest = create_split_manifest(
    catalog,
    policy="group_holdout",
    grouping="crafting_goal",
    test_size=0.2,
    seed=42,
)
save_split_manifest(manifest, "data/manifests/textcraft-goals.json")
```

These settings generate 50 distinct goals, with 40 assigned to probe training
and 10 to testing. TextCraft has no official source partitions, so
`source_split` stays `None`; `probe_split` holds the assignment. All instances,
rollouts, steps, and anchors for a goal stay together.

If the manifest file already exists, load it with `load_split_manifest` instead
of recreating it. Generation settings and source fingerprints must match it.

## 2. Collect into two fresh output directories

```bash
python scripts/collect_textcraft.py \
  --model Qwen/Qwen2.5-Coder-0.5B-Instruct \
  --num_tasks 50 --n_rollouts 3 --max_steps 16 \
  --temperature 0.7 --max_new_tokens 512 \
  --layers 20 --anchors pre_gen post_gen \
  --split_manifest data/manifests/textcraft-goals.json \
  --output_dir artifacts/textcraft/comparison-001/qwen05b

python scripts/collect_textcraft.py \
  --model Qwen/Qwen2.5-Coder-1.5B-Instruct \
  --num_tasks 50 --n_rollouts 3 --max_steps 16 \
  --temperature 0.7 --max_new_tokens 512 \
  --layers 20 --anchors pre_gen post_gen \
  --split_manifest data/manifests/textcraft-goals.json \
  --output_dir artifacts/textcraft/comparison-001/qwen15b
```

Each command plans 150 episodes. Both models use the same task-to-probe
assignments; actions, success labels, and trajectory lengths can differ.
Layer 20 is zero-based and excludes embeddings; it is valid for these models.
Their hidden dimensions differ, so prepare a separate probe for each model.

Every output must be new or empty. Rerunning either command against an existing
collection fails before writing collection files, even with identical settings.
Choose a fresh comparison/run directory for another collection. There is no
resume mode; a failed run's completed episodes can still be loaded, and its
manifest can be reused for a fresh run.

Each output contains `features/feat.shard*.npz`, matching JSONL metadata,
`run_summary.parquet`, `run_metadata.json`, and a copy of the split manifest.
Run metadata includes resolved model/tokenizer revisions, actual layer indices,
generation defaults and overrides, rollout count, step limit, exact selected
task identities, source filter/limit, and completion status. Use the recorded
model commit through `--model_revision` to repeat a model version.

## 3. Apply the saved manifest to each collection

```python
from src.data import load_activation_dataset, split_activation_dataset
from src.splits import load_split_manifest

manifest = load_split_manifest("data/manifests/textcraft-goals.json")
probe_data = {}
for model in ("qwen05b", "qwen15b"):
    rows = load_activation_dataset(f"artifacts/textcraft/comparison-001/{model}")
    probe_data[model] = split_activation_dataset(rows, manifest)
    print(model, {name: len(partition) for name, partition in probe_data[model].items()})
```

Applying the manifest never reassigns tasks. Missing tasks in an incomplete
collection simply yield fewer rows, possibly an empty probe partition.

For an uploaded NPZ/JSONL run, use the same manifest:

```python
from src.data import load_train_test_split

splits = load_train_test_split(
    "YOUR_HF_USERNAME/textcraft-comparison",
    path_in_repo="qwen05b",
    manifest=manifest,
)
```

This snippet assumes the run was previously uploaded to that subdirectory.
It does not create or publish a dataset.

## 4. Prepare one model's probe inputs

```python
from torch.utils.data import DataLoader

splits = probe_data["qwen05b"]
train = splits["train"].filter(lambda row: row["anchor"] == "pre_gen")
test = splits["test"].filter(lambda row: row["anchor"] == "pre_gen")
train_loader = DataLoader(
    train.with_format("torch", columns=["activations", "success"]),
    batch_size=64, shuffle=True,
)
test_loader = DataLoader(
    test.with_format("torch", columns=["activations", "success"]),
    batch_size=64, shuffle=False,
)
batch = next(iter(train_loader))
X = batch["activations"].flatten(start_dim=1)
y_failure = 1 - batch["success"]
```

Each row's label describes its episode's final outcome. This example uses
pre-generation activations at every decision. Filter `step_idx == 0` as well
if the experiment specifically predicts failure from the first decision.

## Collect first, then choose crafting-goal groups

Omit `--split_manifest` from collection commands and use fresh output paths, for
example `artifacts/textcraft/collect-first-001/qwen05b`. These rows have provisional
task groups but no probe assignments or manifest IDs. Then explicitly regroup:

```python
from src.benchmark.textcraft import TextCraftBenchmark
from src.data import create_manifest_from_activations, load_activation_dataset, split_activation_dataset
from src.splits import save_split_manifest

benchmark = TextCraftBenchmark(num_tasks=50, seed=42)
catalog = benchmark.task_catalog(grouping="crafting_goal")
rows = load_activation_dataset("artifacts/textcraft/collect-first-001/qwen05b")
manifest = create_manifest_from_activations(
    rows,
    task_catalog=catalog,
    grouping="crafting_goal",
    policy="group_holdout",
    test_size=0.2,
    seed=42,
    allow_regroup=True,
)
save_split_manifest(manifest, "data/manifests/collect-first-goals.json")
splits = split_activation_dataset(
    rows, manifest, task_catalog=catalog, allow_regroup=True,
)
```

Reuse this manifest with the second model's original unassigned rows, passing
the same catalog and `allow_regroup=True`. The full catalog covers tasks missing
from either collection. The original files are unchanged. Once annotations are
established, their grouping cannot be changed: even one non-null `probe_split`
or `manifest_id` blocks explicit regrouping. Applying the same manifest to an
already annotated dataset uses `allow_regroup=False`, the default.

## Switching this example to BigCodeBench

```python
from src.benchmark.bigcodebench.bigcodebench import BigCodeBenchmark
from src.splits import create_split_manifest

benchmark = BigCodeBenchmark(subset="hard", release="v0.1.4", split="instruct")
manifest = create_split_manifest(
    benchmark.task_catalog(), policy="group_holdout", test_size=0.2, seed=42,
)
```

BigCodeBench's five official source partitions are releases of the same task
set. The adapter loads one release, defaulting to `v0.1.4`, and preserves that
name as `source_split`. Pass the same `--release v0.1.4` to both
`scripts/collect_bigcodebench.py` runs. `--split instruct` selects prompt format;
`--dataset_revision` selects a Hub commit. These are separate settings.
Do not map different releases to probe train/test: the task IDs repeat across
them. Use a holdout within the selected release.

## Verification scope

Regression tests use synthetic data matching the five-release Hub layout,
mocked Hub/model loading, fixture tensors, and the real local TextCraft adapter.
They exercise serialization, regrouping, incomplete collections, and output
protection. Reading the public dataset cards checks release names and layout;
it is not an execution test against downloaded BigCodeBench rows. The two real
model collection commands above have not been executed as part of those tests.
