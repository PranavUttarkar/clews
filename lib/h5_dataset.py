import os
import json
import torch
import numpy as np
from torch.utils.data import Dataset

try:
    import h5py  # type: ignore
except Exception:  # pragma: no cover
    h5py = None

from lib import tensor_ops as tops


class H5CQTDataset(Dataset):
    """
    Dataset for precomputed CQT stored in HDF5 (.h5) chunks or memory-mapped (.mm) files.

    Split files are in Format.json style:
    {
        "C-xxxx": [ {"version_id": "V-...", "track_title": "...", "youtube_id": "..."}, ... ]
    }

    It builds batches similar to lib.dataset.Dataset: returning [clique_label, i1, x1, i2, x2, ...]
    where each xk is a tensor (S, C, T) to be fed into model.embed directly.
    """

    def __init__(
        self,
        conf,
        split: str,
        augment: bool = False,
        fullsongs: bool = False,
        checks: bool = True,
        verbose: bool = False,
    ) -> None:
        assert split in ("train", "valid", "test")
        self.verbose = verbose
        self.augment = augment
        self.fullsongs = fullsongs
        self.pad_mode = conf.pad_mode
        self.n_per_class = conf.n_per_class
        self.p_samesong = conf.p_samesong

        # Feature/frame timing
        # Seconds per CQT hop in the precomputed features (default to model.cqt.hoplen if provided via conf)
        self.frame_hop_s = getattr(conf, "feature_hoplen", None)
        # Shingling from model config replicated into data conf by caller
        self.shingle_len_s = getattr(conf, "shingle_len", 20.0)
        self.shingle_hop_s = getattr(conf, "shingle_hop", 20.0)

        # Paths
        self.h5_root = conf.path.get("h5_root", None)
        self.mm_root = conf.path.get("mm_root", None)  # optional
        self.jsonl = conf.path.get("jsonl", None)
        self.index_path = conf.path.get("index", os.path.join("cache", "discogs_h5_index.pt"))
        self.splits = {
            "train": conf.path.splits.train,
            "valid": conf.path.splits.valid,
            "test": conf.path.splits.test,
        }

        # Load or build index (youtube_id -> source pointers)
        self.index = None
        if os.path.exists(self.index_path):
            try:
                self.index = torch.load(self.index_path)
                if self.verbose:
                    print(f"[H5CQTDataset] Loaded index: {self.index_path} ({len(self.index)} entries)")
            except Exception as e:
                if self.verbose:
                    print(f"[H5CQTDataset] Failed to load index ({e}), will build lazily.")
        if self.index is None:
            self.index = {}
            if self.verbose:
                print("[H5CQTDataset] Empty index; dataset will rely only on paths discoverable from JSONL or defaults.")

        # Parse split file
        # clique: dict[str, list[str]] mapping clique_id -> list of version_id strings
        # info: dict[str, dict] mapping version_id -> {clique, youtube_id}
        self.clique, self.info = self._load_split(self.splits[split])

        # Build versions list and id maps
        self.clique2id = {}
        for i, cl in enumerate(self.clique.keys()):
            self.clique2id[cl] = i
        self.versions = []  # list of version_id strings
        for vers in self.clique.values():
            self.versions += vers

        # Optional checks
        if checks:
            self._perform_checks()

        if self.verbose:
            print(f"[H5CQTDataset] {split}: {len(self.clique)} cliques, {len(self.versions)} versions")

        # Per-worker handle registry to avoid sharing HDF5 handles across processes
        self._h5_handles = {}

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.versions)

    def __getitem__(self, idx):
        v1 = self.versions[idx]  # version_id string
        cl = self.info[v1]["clique"]
        icl = self.clique2id[cl]
        # Other versions from same clique (with optional same-song probability)
        othervers = []
        for v in self.clique[cl]:
            if v != v1 or torch.rand(1).item() < self.p_samesong:
                othervers.append(v)
        if self.augment and len(othervers) > 1:
            order = torch.randperm(len(othervers)).tolist()
            othervers = [othervers[k] for k in order]

        # Construct n_per_class
        v_n = [v1]
        if len(othervers) == 0:
            othervers = [v1]
        for k in range(self.n_per_class - 1):
            v_n.append(othervers[k % len(othervers)])

        # Compose output list [icl, i1, x1, i2, x2, ...]
        output = [torch.tensor([icl], dtype=torch.long)]
        for v in v_n:
            vid = self._version_to_int(v)  # stable int id from version string
            x = self._load_and_shingle(self.info[v]["youtube_id"])  # (S,C,T)
            output += [torch.tensor([vid], dtype=torch.long), x]
            if self.fullsongs:
                return output
        return output

    # ------------------------------------------------------------------
    def _load_split(self, fn):
        with open(fn, "r", encoding="utf-8") as f:
            data = json.load(f)
        clique: dict[str, list[str]] = {}
        info: dict[str, dict] = {}
        # data is dict: clique_id -> list of {version_id, track_title, youtube_id}
        for cl, items in data.items():
            vers_ids = []
            for it in items:
                if "youtube_id" in it and "version_id" in it:
                    vid = it["version_id"]
                    vers_ids.append(vid)
                    info[vid] = {"clique": cl, "youtube_id": it["youtube_id"]}
            if len(vers_ids) >= 2:  # only useful cliques
                clique[cl] = vers_ids
        return clique, info

    def _perform_checks(self):
        # basic checks: cliques size and coverage in index
        small = [cl for cl, arr in self.clique.items() if len(arr) < 2]
        if small and self.verbose:
            print(f"[H5CQTDataset] Warning: {len(small)} cliques with <2 versions will be ignored in sampling.")

    def _version_to_int(self, vstr: str) -> int:
        # Stable hash truncated to positive int32
        return (abs(hash(vstr)) % (2**31))

    # ---------------------------- Feature loading ---------------------
    def _load_and_shingle(self, youtube_id: str) -> torch.Tensor:
        # Get feature array (C,T)
        feat, frame_hop = self._load_feature_ct(youtube_id)
        # Shingle into (S,C,T_window)
        S, C, T = self._shingle(feat, frame_hop)
        return S

    def _load_feature_ct(self, youtube_id: str):
        """Return (feat[C,T], frame_hop_seconds)."""
        # Try index lookup first
        entry = self.index.get(youtube_id, None)
        hop = self.frame_hop_s

        if entry is not None:
            # H5 path
            if "h5_file" in entry and "h5_key" in entry and self.h5_root:
                h5_path = os.path.join(self.h5_root, entry["h5_file"]) if not os.path.isabs(entry["h5_file"]) else entry["h5_file"]
                key = entry["h5_key"]
                feat = self._read_h5(h5_path, key)
                hop = entry.get("hop_s", hop)
                if feat is not None:
                    return feat, hop
            # MM path
            if "mm_path" in entry:
                mm_path = entry["mm_path"]
                if self.mm_root and not os.path.isabs(mm_path):
                    mm_path = os.path.join(self.mm_root, mm_path)
                feat = self._read_mm(mm_path, entry.get("shape"))
                hop = entry.get("hop_s", hop)
                if feat is not None:
                    return feat, hop

        # Fallback heuristics (if no index info)
        # Try a conventional .mm layout based on youtube_id two-char shard
        if self.mm_root is not None:
            shard = youtube_id[:2]
            cand = os.path.join(self.mm_root, shard, f"{youtube_id}.mm")
            feat = self._read_mm(cand)
            if feat is not None:
                return feat, hop

        raise FileNotFoundError(f"Features for {youtube_id} not found in index or mm_root")

    def _read_h5(self, h5_path: str, key: str):
        if h5py is None:
            return None
        try:
            # Use per-process handle cache
            handle = self._h5_handles.get(h5_path)
            if handle is None:
                # swmr=True only if files were written that way; safer to open read-only
                handle = h5py.File(h5_path, "r")
                self._h5_handles[h5_path] = handle
            ds = handle[key]
            arr = np.array(ds, dtype=np.float32)
            # Expect (C,T) or (T,C); fix if needed
            if arr.ndim == 2 and arr.shape[0] < arr.shape[1]:
                # likely (C,T)
                pass
            elif arr.ndim == 2:
                arr = arr.T
            else:
                # try to squeeze to (C,T)
                arr = np.squeeze(arr)
                if arr.ndim == 2 and arr.shape[0] > arr.shape[1]:
                    arr = arr.T
            return torch.from_numpy(arr)
        except Exception as e:
            if self.verbose:
                print(f"[H5CQTDataset] H5 read error {h5_path}:{key} -> {e}")
            return None

    def __del__(self):
        # Close any open HDF5 handles on dataset destruction
        try:
            for h5_path, handle in list(self._h5_handles.items()):
                try:
                    handle.close()
                except Exception:
                    pass
            self._h5_handles.clear()
        except Exception:
            pass

    def _read_mm(self, mm_path: str, shape=None):
        try:
            if not os.path.exists(mm_path):
                return None
            mm = np.memmap(mm_path, mode="r", dtype=np.float32)
            if shape is not None:
                C, T = shape
                arr = np.reshape(mm, (C, T))
            else:
                # Heuristic: infer C from common CQT bins, else assume 84
                size = mm.size
                for C in (84, 96, 120, 128):
                    if size % C == 0:
                        T = size // C
                        arr = np.reshape(mm, (C, T))
                        break
                else:
                    return None
            return torch.from_numpy(np.array(arr))
        except Exception as e:
            if self.verbose:
                print(f"[H5CQTDataset] MM read error {mm_path} -> {e}")
            return None

    # ---------------------------- Shingling ---------------------------
    def _shingle(self, feat_ct: torch.Tensor, frame_hop_s: float | None):
        # feat_ct: (C,T)
        if frame_hop_s is None:
            # Default to 0.02s per frame if unknown
            frame_hop_s = 0.02
        C, T = feat_ct.shape
        win_frames = max(1, int(round(self.shingle_len_s / frame_hop_s)))
        hop_frames = max(1, int(round(self.shingle_hop_s / frame_hop_s)))
        # Pad along time to ensure at least one window
        if T < win_frames:
            pad_needed = win_frames - T
            pad_tile = torch.tile(feat_ct, (1, (pad_needed + T - 1) // T + 1))
            feat_ct = pad_tile[:, :win_frames]
            T = feat_ct.shape[1]
        # Unfold
        windows = feat_ct.unfold(dimension=1, size=win_frames, step=hop_frames)  # (C, S, win)
        windows = windows.permute(1, 0, 2).contiguous()  # (S, C, win)
        return windows, C, win_frames

