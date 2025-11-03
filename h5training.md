# H5/MM Training Quickstart (HPC)

This project can train on precomputed Discogs-VI CQT features stored in HDF5 chunks and/or memory-mapped .mm files.

## Data locations

- HDF5 chunks (example on HPC):
	- `../../discogs-vi-h5-rechunked/discogs_vi_chunk_00001.h5` ...
- JSONL: `metadata/Discogs-VI-20240701.jsonl`
- Splits in Format.json style: `metadata/train.json`, `metadata/valid.json`, `metadata/verify.json`

## 1) Build the index (once)

The index maps `youtube_id` to feature location in H5 (file/key) and/or MM.

```powershell
# Windows PowerShell example (adjust paths)
python scripts/build_discogs_h5_index.py --jsonl metadata/Discogs-VI-20240701.jsonl --h5-root ../../discogs-vi-h5-rechunked --out cache/discogs_h5_index.pt
```

On Slurm/Linux:
```bash
python scripts/build_discogs_h5_index.py --jsonl metadata/Discogs-VI-20240701.jsonl \
	--h5-root ../../discogs-vi-h5-rechunked \
	--out cache/discogs_h5_index.pt
```

## 2) Configure

Edit `config/dvi-h5.yaml`:

```yaml
path:
	h5_root: "../../discogs-vi-h5-rechunked"
	jsonl: "metadata/Discogs-VI-20240701.jsonl"
	index: "cache/discogs_h5_index.pt"
	splits:
		train: "metadata/train.json"
		valid: "metadata/valid.json"
		test:  "metadata/verify.json"
model:
	cqt:
		hoplen: 0.02   # Set to your precomputed feature hop
```

## 3) Train

Windows PowerShell:
```powershell
$env:OMP_NUM_THREADS="1"
python train_h5.py jobname=dvi-h5 conf=config/dvi-h5.yaml fabric.nnodes=1 fabric.ngpus=2
```

Slurm example:
```bash
#!/bin/bash
#SBATCH -J dvi-h5
#SBATCH -A <account>
#SBATCH -N 1
#SBATCH -t 48:00:00
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=16
export OMP_NUM_THREADS=1
export HDF5_USE_FILE_LOCKING=FALSE
srun python -u train_h5.py jobname=dvi-h5 conf=config/dvi-h5.yaml fabric.nnodes=1 fabric.ngpus=2
```

## Notes

- The loader opens a separate HDF5 handle per worker; increase `data.nworkers` and `data.prefetch_factor` to improve throughput.
- Only CQT-domain augmentations are applied in H5 mode.
- If some ids are missing in the H5 index, the dataset will try MM fallback when `path.mm_root` is set.
- If you know the exact dataset key inside H5 (e.g., `ytid/cqt`), ensure the JSONL provides `h5_key`; otherwise, add it later and rebuild the index.

---

## Detailed reference: how H5 training works

### Files and folders expected

- HDF5 chunks directory (`path.h5_root`):
	- Example: `../../discogs-vi-h5-rechunked`
  - Contains files named like `discogs_vi_chunk_00001.h5` ... `discogs_vi_chunk_00489.h5`
- JSONL metadata (`path.jsonl`):
  - `metadata/Discogs-VI-20240701.jsonl`
  - One JSON record per youtube_id; may include optional fields we use:
	 - `youtube_id`: string id (required)
	 - `h5_chunk`: filename within `h5_root` (e.g., `discogs_vi_chunk_00001.h5`)
	 - `h5_key`: dataset key in the H5 file (e.g., `VYb7c8x/cqt`)
	 - `mm_path`: optional path to `.mm` file (absolute or relative to `path.mm_root`)
	 - `hop_s`: seconds per CQT frame (e.g., `0.02`)
	 - `shape`: `[C, T]` if loading from `.mm`
- Splits in Format.json style (`path.splits.*`):
  - `metadata/train.json`, `metadata/valid.json`, `metadata/verify.json`
  - Each is a dict mapping `clique_id` to a list of items; each item at least:
	 - `version_id`: unique version string (e.g., `V-1609560`)
	 - `youtube_id`: string id used to look up features
- Index file (`path.index`):
  - `cache/discogs_h5_index.pt`
  - Built by `scripts/build_discogs_h5_index.py`; maps `youtube_id` -> pointers (h5_file, h5_key, mm_path, hop_s, shape)

### Scripts to run

1) Build index:
	- `scripts/build_discogs_h5_index.py` parses JSONL and records any known H5/MM pointers per `youtube_id`.
	- Run this after the data is staged and before training.

2) Optional quick verification:
	- `scripts/verify_h5_sample.py --conf config/dvi-h5.yaml --split train --index 0 --verbose`
	- Loads a single item, prints out shapes and minimal stats.

3) Training:
	- `train_h5.py` uses Fabric/DDP, the CLEWS model, and the H5 dataset.

### How the dataset resolves features

`lib/h5_dataset.py` does the following for each sample:

1) Split parsing:
	- Loads the split JSON (e.g., `metadata/train.json`).
	- Builds `clique` as `clique_id -> [version_id, ...]` and `info` mapping `version_id -> {clique, youtube_id}`.

2) Sampling:
	- For a given index, picks anchor `v1` and `n_per_class-1` other versions from the same clique.
	- Returns a list `[clique_label, i1, x1, i2, x2, ...]` where `ik` is a stable int id from `version_id` and `xk` is the feature tensor.

3) Feature lookup:
	- Uses the index (`cache/discogs_h5_index.pt`) to locate features for the `youtube_id`.
	- Priority: H5 (using `h5_file` + `h5_key`) > MM (using `mm_path`).
	- If neither is found, raises `FileNotFoundError`.

4) Reading features:
	- H5 is opened read-only with one handle per worker; dataset caches these handles.
	- `.mm` files are read via `numpy.memmap` and reshaped to `[C, T]` using `shape` if provided, or common CQT bin counts.

5) Shingling (forming (S, C, T)):
	- Determines `frame_hop_s` from `index.hop_s` or falls back to `model.cqt.hoplen`.
	- Computes window frames from `model.shingling.len` and hop frames from `model.shingling.hop` (seconds -> frames).
	- Pads along time if too short, using `data.pad_mode` (`repeat`).
	- Unfolds along time to produce `(S, C, T_window)`.

6) Training loop integration:
	- `train_h5.py` stacks the batch to `(B, S, C, T)` and calls `model.embed` directly (skipping waveform/CQT in `model.prepare`).
	- Loss, metrics, optimizer, scheduler remain the same as waveform training.

### HPC-specific considerations

- HDF5 file locking: if your shared filesystem lacks POSIX locks, set `HDF5_USE_FILE_LOCKING=FALSE`.
- Multiprocessing context: DataLoader uses `spawn` by default in `train_h5.py` (configurable via `data.mp_context`).
- Worker resources: adjust `data.nworkers` and `data.prefetch_factor` to avoid I/O bottlenecks.
- Memory: precomputed CQT can increase activation size—tune `training.batchsize` if you hit VRAM limits.

### Configuration knobs (dvi-h5.yaml)

- `path.h5_root`: root dir with H5 chunks
- `path.mm_root`: optional MM root if some IDs fall back to `.mm`
- `path.jsonl`: Discogs-VI JSONL
- `path.index`: autogenerated index file (PyTorch .pt)
- `path.splits.*`: Format.json style split files
- `model.cqt.hoplen`: seconds per frame for CQT; should match the precomputed features
- `model.shingling.len/hop`: seconds per shingle and hop
- `data.nworkers`, `data.prefetch_factor`, `data.pad_mode`, `data.n_per_class`

### Troubleshooting

- Missing features for a youtube_id:
  - Ensure JSONL includes `h5_chunk` and `h5_key`, or provide `mm_path`/`mm_root`.
  - Re-run the index builder after updating JSONL.
- Shape mismatch loading `.mm`:
  - Add `shape: [C, T]` to the JSONL record or supply H5 pointers instead.
- Runtime HDF5 errors under DDP:
  - Use `spawn` mp context (default), set `HDF5_USE_FILE_LOCKING=FALSE`, reduce workers.


Plan overview
Add an HPC-ready dataset that reads precomputed features from HDF5 files (and falls back to .mm files) while honoring Format.json-style splits.
Build a lightweight index (once) from the JSONL + split files so data loading is O(1) and safe for multi-worker HDF5.
Provide a training script that bypasses waveform CQT (model.prepare) and feeds precomputed CQT into model.embed directly.
Add a config for H5/MM training and HPC-friendly DataLoader settings.
Document how to run (local and Slurm), with environment flags for HDF5 concurrency.
Below I keep things minimal and robust so you can plug in your data paths and start.

What we’ll add
New dataset: lib/h5_dataset.py

Reads cliques from Format.json split files (train.json, valid.json, verify.json).
Maps youtube_id to feature locations by scanning the JSONL once and caching an index file.
Loads from HDF5 if found, otherwise tries .mm memory-mapped paths.
Returns per-sample tensors shaped for model.embed: (S, C, T), where S is shingle count, C is frequency bins, T is time frames.
Per-worker HDF5 handles for HPC safety; no sharing across processes.
One-time indexing script: scripts/build_discogs_h5_index.py

Input: JSONL (Discogs-VI-20240701.jsonl) + discovered H5/MM files under discogs-vi-h5/.
Output: cache/discogs_h5_index.pt with:
youtube_id -> {h5_file, dataset path or key, mm_path} and metadata like time resolution (hop) and feature shape.
Training script: train_h5.py

Same Fabric training loop you’re using, but:
Skips model.prepare (CQT) and feeds CQT directly into model.embed.
Dataloader tuned for HPC (pin_memory, persistent_workers, prefetch_factor).
Keeps your loss, metrics, checkpointing, logging unchanged.
Config: config/dvi-h5.yaml

Paths for h5_root, jsonl, splits (train/valid/verify in Format.json).
Flags for input format (“cqt”) and shingling parameters (len, hop) to match the model config.
Dataloader settings for HPC.
README updates: short “H5/MM training” section with Slurm snippet, environment settings, and common pitfalls.

Data assumptions and flexible handling
Because H5/MM layouts vary, the dataset will try in this order per youtube_id:

HDF5
Expected patterns:
One of:
A top-level group per youtube_id with dataset “cqt” (e.g., /<ytid>/cqt)
A flat dataset where youtube_id is stored as an attribute or sidecar index in JSONL
The index file encodes the exact h5 filepath and internal path.
MM
Expected .mm path: found in JSONL (common for Discogs CQT), or derivable as:
Discogs-VI-20240701/magnitude_cqt/cqt/<2-char>/<youtube_id>.mm
We’ll load with numpy.memmap (float32) and reshape as (C, T) from metadata in the JSONL or from a default you can override.
If both exist, we use HDF5 by default and can switch via a config flag.

How the dataset will work
Inputs:

conf.path.h5_root: folder with discogs_vi_chunk_00001.h5 … discogs_vi_chunk_00489.h5
conf.path.jsonl: metadata/Discogs-VI-20240701.jsonl
conf.data.splits: paths to Format.json-like train.json, valid.json, verify.json
Build-time (first run or when index missing):

Parse JSONL lines for:
youtube_id
any feature pointers (h5 chunk id, dataset key, or mm path)
optional metadata (hop time in seconds, frequency bins)
Scan H5 folder once to map chunk file names to accessible datasets (if JSONL doesn’t give enough).
Save to cache/discogs_h5_index.pt.
Runtime:

Parse the split file (Format.json-like). Each top-level key is a clique id (e.g., “C-...”), and each entry has a version_id and youtube_id. We use:
clique: top-level key
version: the version_id
youtube_id: to find the feature source from the index
As in your current dataset.py, build batches of n_per_class by sampling versions within the same clique.
For each selected version, load features and apply shingling to shape (S, C, T) with:
Slen = conf.model.shingling.len (seconds)
Shop = conf.model.shingling.hop (seconds)
Frame hop = from feature metadata (default to conf.model.cqt.hoplen seconds if not available)
Return a list like your current dataset: [clique_label, i1, x1, i2, x2, …] where each xk is a tensor (S, C, T). The training loop will detect “precomputed cqt” mode and call model.embed directly.
Error handling:

Missing entry: skip or sample replacement (log once per worker).
Too-short features: pad using your pad_mode (repeat) along the time axis.
Inconsistent shapes: attempt to coerce to (C, T) and log shape mismatches.
Training changes
We’ll keep the training loop structure but adjust the “prepare” step:

If conf.data.input == "cqt":
Do not call model.prepare
Feed precomputed features directly to model.embed
If conf.data.input == "waveform":
Same as current behavior
This is easiest to ship as a new script (train_h5.py) to avoid touching your stable train.py:

Batch unpacking remains the same.
Data augmentation: only CQ-domain augmentations apply (time-domain augmentations are skipped since we don’t have waveforms).
Loss and metrics unchanged.
Config: config/dvi-h5.yaml (new)
path:
h5_root: "data/discogs-vi-h5/"
jsonl: "metadata/Discogs-VI-20240701.jsonl"
index: "cache/discogs_h5_index.pt"
splits:
train: "metadata/train.json"
valid: "metadata/valid.json"
test: "metadata/verify.json"
logs: "logs/"
cache: "cache/"
data:
input: "cqt"
nworkers: 8-16 per node (tune)
audiolen: not used for cqt; we use shingle length from model
n_per_class: 4
p_samesong: 0
pad_mode: "repeat"
prefetch_factor: 4
persistent_workers: true
model: keep your clews settings; ensure cqt.hoplen matches the feature hop used by Discogs VI CQT (commonly 0.02s)
fabric: same as current
training: same optimizer/scheduler; batchsize might need adjustment depending on feature size in memory
I’ll leave waveform-related augmentations commented out; keep only CQ-domain ones (specaugment, timestretch, pitchtranspose) which still make sense on CQT.

HPC-specific safeguards
HDF5 settings:
Use per-worker file handles (open files in worker_init_fn or lazily on first access).
Consider setting env var: HDF5_USE_FILE_LOCKING=FALSE when using shared filesystems that don’t support locking.
If your H5 was written with SWMR, open read-only with swmr=True.
DataLoader:
pin_memory=True
persistent_workers=True
prefetch_factor=4 (tune)
Multiprocessing context:
On Linux HPC, default “fork” is okay, but if you see issues with HDF5, set mp_context="spawn" for DataLoader.
Contract (dataset -> model)
Input to model.embed: (B, S, C, T), float32
Slen (seconds) and Shop from conf.model.shingling
T axis consistent with conf.model.cqt.hoplen seconds per frame
C equals conf.cqt.noctaves * conf.cqt.nbinsoct (e.g., 7*12=84 or whatever you stored)
Labels and indices:
cc: (B,), clique ids (int64)
ii: (B,), version ids (int64)
Error modes:
Missing features: log and replace with zeros; don’t crash the epoch.
Short sequences: pad along T to satisfy shingle windows.
Likely edge cases
Versions present in split but missing in index: handle gracefully and report count.
Extremely long CQT: shingling creates large S; consider random cropping for training (respecting your augment flag).
Mismatch between stored hop and conf.model.cqt.hoplen: detect and warn, then resample (optional) or proceed with mismatched hop (safer to proceed with warning).
Worker OOM during prefetch: reduce batchsize/prefetch_factor; consider half precision in Fabric if memory is tight (for model, not the input features).