# Interp Sandbox

A research sandbox for **Agentic Mechanistic Interpretability & Representation Engineering**.

## RunPod Instructions

Clone the repo:

```bash
cd /workspace
git clone https://github.com/skarnati20/interp-sandbox.git
cd interp-sandbox
```

Run desired setup script:

```bash
./scripts/setup_textcraft.sh
```

Start `tmux` sessions:

```bash
tmux new -s textcraft
```

Run benchmark script to generate activations:

```bash
python scripts/collect_textcraft.py \
  --model Qwen/Qwen2.5-Coder-7B-Instruct \
  --num_tasks 50 \
  --n_rollouts 5 \
  --max_steps 16 \
  --temperature 0.7 \
  --layers 20 \
  --anchors post_gen pre_gen \
  --output_dir artifacts/textcraft/qwen2.5-7b/qwen2.5-7b
```

Upload to HuggingFace:

```bash
python scripts/export.py \
  --data_dir artifacts/textcraft/qwen2.5-7b/qwen2.5-7b \
  --repo_id <YOUR_HF_USERNAME>/textcraft-qwen7b-activations \
  --token hf_YOUR_WRITE_TOKEN_HERE \
  --commit_message "Upload TextCraft Qwen2.5-Coder-7B residual activations"
```

## Source partitions and probe assignments

Use Python 3.10+ and install `requirements.txt`. Official source partition names
(`source_split`) describe where tasks came from. Probe assignments (`probe_split`)
are separate and come from one saved manifest reused across models.

### Create a manifest before collection

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

Task identity is namespaced by benchmark/dataset configuration and retains the
original source task ID. Adapters support `grouping="task"` by default. TextCraft
also supports `"crafting_goal"` to keep all instances of one goal together.
Generated adapters record generator code/data fingerprints and generation
parameters; their `source_split` is `None`, with no invented official partitions.
TextCraft sorts recipe candidates before seeded sampling for reproducibility
across Python processes.

For BigCodeBench, release partitions are alternate versions of the same tasks,
not train/test sets. The documented default is **`release="v0.1.4"`** for both full
and hard variants. Pass `release="v0.1.2"` in Python or `--release v0.1.2` on the
CLI to select another release. Only that partition is loaded; releases are never
combined. Its exact name (for example `v0.1.4`) remains the `source_split`, and the
release plus resolved Hub commit are recorded as provenance. `task_catalog()`
includes all tasks in the selected release even when `num_tasks` limits collection.
`list_tasks()` selects that release; `list_tasks(split="v0.1.4")` is an optional
matching source filter. Requesting a different or missing partition raises an error.
The existing `instruct`/`complete` option controls prompt format independently.
`dataset_config` and `dataset_revision` select the dataset configuration and Hub
commit, independently of the release partition.

The five partition names are `v0.1.0_hf`, `v0.1.1`, `v0.1.2`, `v0.1.3`, and
`v0.1.4`, as declared in the [full dataset card](https://huggingface.co/datasets/bigcode/bigcodebench/blob/main/README.md)
and [hard dataset card](https://huggingface.co/datasets/bigcode/bigcodebench-hard/blob/main/README.md).
The shared loader for future adapters is
`src.benchmark.huggingface.load_source_dataset`; its optional `source_split`
selects one partition at the load boundary.

For a benchmark whose official partitions are named `train`, `validation`, and
`test`, use explicit source-to-probe mapping:

```python
manifest = create_split_manifest(
    benchmark.task_catalog(),
    policy="source_mapping",
    source_mapping={"train": "train", "validation": "validation", "test": "test"},
)
```

This requires those source partitions to actually exist. If a dataset provides
only official `test`, a `group_holdout` can divide that source pool for probe
training/testing while every row retains `source_split="test"`. Alternatively,
`source_mapping={"test": "test"}` creates only a probe test partition. No missing
source partitions are synthesized. `source_splits=["test"]` optionally restricts
a group holdout to that source pool; excluded tasks remain recorded but unassigned.

Conflicting probe assignments within a group are errors, including a task or goal
spanning source partitions mapped to different probe splits. Missing partitions,
unknown tasks, changed provenance, and incompatible grouping also raise errors.
Manifest saving refuses to overwrite an existing file. Load and reuse it instead
of recreating assignments from a seed or from each model's completed tasks.

### Collect with the same manifest across models

```bash
python scripts/collect_textcraft.py \
  --model Qwen/Qwen2.5-Coder-7B-Instruct \
  --num_tasks 50 --n_rollouts 3 --max_steps 16 --max_new_tokens 512 \
  --temperature 0.7 --layers 20 \
  --split_manifest data/manifests/textcraft-goals.json \
  --output_dir artifacts/textcraft/qwen7b
```

All collection scripts accept `--split_manifest`, `--model_revision`,
`--source_split`, and `--max_new_tokens` (default 512 tokens per decision).
An official source filter is invalid for generated tasks. Use a new or empty
output directory per model/run and matching benchmark generation settings.
Nonempty output directories are rejected even if their metadata matches, including
partial or failed runs, metadata-only directories, and orphan feature shards.
There is no resume or overwrite mode. Each Runner can start only one collection;
`run_task(...)` creates and finalizes a single-episode collection.

The model and tokenizer are pinned to the resolved model commit;
local model directories receive a content fingerprint. Layer indices are
zero-based transformer blocks, excluding embeddings. Invalid, negative, duplicate,
or empty layer selections fail rather than being clamped; omitting layers selects
all blocks. Actual revisions and selected layers are saved in `run_metadata.json`
and step metadata. A copy of the split manifest is saved beside the collection.
Before the first episode, run metadata also records model generation defaults,
requested and effective temperature, sampling mode, generated-token limit,
rollout count, step limit, adapter settings, source filter, task limit, and the
ordered exact task identities selected. It records completion status and completed
episode count. The existing multi-rollout fallback from temperature 0 to 0.7 is
explicitly recorded. Completed episodes are flushed on a later episode failure;
start a fresh output directory to collect again with the same manifest.

`Runner(..., split_manifest=manifest_or_path)` provides the same interface in
Python. Without a manifest, collection records provenance and leaves probe
assignments unset so a manifest can be created afterward.

### Load activations and apply the saved manifest

```python
from src.data import load_activation_dataset, load_train_test_split, split_activation_dataset
from src.splits import load_split_manifest

manifest = load_split_manifest("data/manifests/textcraft-goals.json")
dataset = load_activation_dataset("artifacts/textcraft/qwen7b")
splits = split_activation_dataset(dataset, manifest)

# Or load NPZ/JSONL artifacts uploaded with scripts/export.py:
splits = load_train_test_split(
    "YOUR_HF_USERNAME/textcraft-activations",
    manifest=manifest,
    path_in_repo="qwen7b",  # Empty string when the run is at the repository root.
)
train = splits["train"].with_format("torch", columns=["activations", "success"])
splits.save_to_disk("data/textcraft-probe-splits")
# Optional standard Hugging Face dataset publishing:
# splits.push_to_hub("YOUR_HF_USERNAME/textcraft-probe-splits", private=True)
```

Each row contains float16 `activations` shaped `[n_layers, hidden_dim]`, actual
`layer_ids`, and step metadata including `success`. Source provenance is JSON text
in the Arrow dataset so different source schemas can coexist without losing
fields. The NPZ/JSONL files retain their existing layout. The loader checks shard
pairing, row counts, and layer layouts across all shards. The Hub helper supports
`token`, `revision`, `cache_dir`, and `local_files_only=True`.

Applying a manifest to an incomplete collection preserves every task's saved
assignment and returns empty probe partitions when their tasks are absent. It
never moves tasks to compensate for missing data. Unassigned source rows remain
available in `apply_activation_manifest(...)`, but are excluded from the returned
probe partitions. Existing conflicting annotations are rejected. Published
`DatasetDict` splits can be reloaded with standard `datasets.load_dataset(...)`.

### Create manifests for existing collections

```python
from src.data import create_manifest_from_activations

manifest = create_manifest_from_activations(
    dataset,
    task_catalog=benchmark.task_catalog(grouping="crafting_goal"),
    grouping="crafting_goal",
    policy="group_holdout",
    test_size=0.2,
    seed=42,
    allow_regroup=True,
)
save_split_manifest(manifest, "data/manifests/existing-collection.json")
splits = split_activation_dataset(
    dataset,
    manifest,
    task_catalog=benchmark.task_catalog(grouping="crafting_goal"),
    allow_regroup=True,
)
```

Collections created without a manifest have provisional task-based group IDs.
`allow_regroup=True` explicitly replaces those groups with crafting-goal groups
from the catalog (or a `group_column` for legacy metadata). It is accepted only
when every input row has both `probe_split` and `manifest_id` unset. A single
established assignment or manifest ID blocks regrouping for the whole input.
Identity and provenance checks still apply; the original shards are unchanged.
After annotation, use the saved grouping and manifest without `allow_regroup`.

With an authoritative catalog, the manifest covers the full intended task universe
even if this collection is incomplete. Older rows missing provenance can be
resolved against that catalog; ambiguous task IDs are rejected. When applying the
manifest to those older collections, pass the same `task_catalog` to
`split_activation_dataset` or the Hub helper.

If no catalog is available, metadata from new collections supplies identity and
provenance. For older collections lacking those fields, provide explicit
`namespace` and `provenance`, for example
`provenance={"kind": "legacy", "collection": "my-original-run", "revision": "unknown"}`.
Unknown official partitions and model revisions remain unknown. Non-task grouping
also requires an explicit `group_column` containing the goal/group key. The
resulting manifest covers only observed tasks, so create it from the union of
available task metadata across models when that is your intended universe.

See the [two-model walkthrough](docs/two-model-walkthrough.md) for complete
manifest-first and collect-first examples, including fresh output paths.

Run the offline test suite with:

```bash
HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 python -m unittest discover -s tests -v
```

The tests use a synthetic five-release BigCodeBench fixture with repeated task IDs
and the Hub card's field types, a real local TextCraft environment with a fixture
extractor, temporary NPZ/JSONL/Parquet files, and mocked Hub/model loading. The
offline suite does not download official benchmark rows or execute real model
inference. Reading the public Hub cards verifies release metadata only.
