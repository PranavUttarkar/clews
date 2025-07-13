import sys
import os
import torch
import numpy as np
import json

from utils import file_utils
from lib import tensor_ops as tops

LIMIT_CLIQUES = None


class NPYDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        conf,
        split,
        augment=False,
        fullsongs=False,
        checks=True,
        verbose=False,
    ):
        assert split in ("train", "valid", "test")
        # Params
        self.augment = augment
        self.samplerate = conf.samplerate
        self.fullsongs = fullsongs
        self.audiolen = conf.audiolen if not self.fullsongs else None
        self.maxlen = conf.maxlen if not self.fullsongs else None
        self.pad_mode = conf.pad_mode
        self.n_per_class = conf.n_per_class
        self.p_samesong = conf.p_samesong
        self.verbose = verbose
        self.npy_path = conf.path.npy  # Path to .npy files
        
        # Load metadata
        self.info, splitdict = torch.load(conf.path.meta)
        if LIMIT_CLIQUES is None:
            self.clique = splitdict[split]
        else:
            if self.verbose:
                print(f"[Limiting cliques to {LIMIT_CLIQUES}]")
            self.clique = {}
            for key, item in splitdict[split].items():
                self.clique[key] = item
                if len(self.clique) == LIMIT_CLIQUES:
                    break
        
        # Update filename to point to .npy files instead of audio files
        for ver in self.info.keys():
            # Convert audio filename to .npy filename
            audio_filename = self.info[ver]["filename"]
            # Remove audio extension and add .npy
            base_name = os.path.splitext(audio_filename)[0]
            npy_filename = base_name + ".npy"
            self.info[ver]["npy_filename"] = os.path.join(self.npy_path, npy_filename)
        
        # Checks
        if checks:
            self.perform_checks(splitdict, split)
        
        # Get clique id
        self.clique2id = {}
        if split == "train":
            offset = 0
        elif split == "valid":
            offset = len(splitdict["train"])
        else:
            offset = len(splitdict["train"]) + len(splitdict["valid"])
        for i, cl in enumerate(self.clique.keys()):
            self.clique2id[cl] = offset + i
        
        # Get idx2version
        self.versions = []
        for vers in self.clique.values():
            self.versions += vers
        
        # Prints
        if self.verbose:
            print(
                f"  {split}: --- Found {len(self.clique)} cliques, {len(self.versions)} songs ---"
            )

    ###########################################################################

    def __len__(self):
        return len(self.versions)

    def __getitem__(self, idx):
        # Get v1 (anchor) and clique
        v1 = self.versions[idx]
        i1 = self.info[v1]["id"]
        cl = self.info[v1]["clique"]
        icl = self.clique2id[cl]
        
        # Get other versions from same clique
        otherversions = []
        for v in self.clique[cl]:
            if v != v1 or torch.rand(1).item() < self.p_samesong:
                otherversions.append(v)
        if self.augment:
            new_vers = []
            for k in torch.randperm(len(otherversions)).tolist():
                new_vers.append(otherversions[k])
            otherversions = new_vers
        
        # Construct v1..vn array (n_per_class)
        v_n = [v1]
        i_n = [i1]
        for k in range(self.n_per_class - 1):
            v = otherversions[k % len(otherversions)]
            i = self.info[v]["id"]
            v_n.append(v)
            i_n.append(i)
        
        # Time augment?
        s_n = []
        for v in v_n:
            if self.augment:
                # For .npy files, we assume the length is stored in metadata
                dur = self.info[v].get("length", 150.0)  # Default to 150 seconds
                if self.maxlen is not None:
                    dur = min(self.maxlen, dur)
                start = max(0, torch.rand(1).item() * (dur - self.audiolen))
            else:
                start = 0
            s_n.append(start)
        
        # Load .npy data and create output
        output = [icl]
        for i, v, s in zip(i_n, v_n, s_n):
            fn = self.info[v]["npy_filename"]
            x = self.get_npy_data(fn, start=s, length=self.audiolen)
            output += [i, x]
            if self.fullsongs:
                return output
        return output

    ###########################################################################

    def get_npy_data(self, fn, start=0, length=None):
        """
        Load preprocessed .npy data instead of raw audio
        Assumes .npy files contain preprocessed features (e.g., CQT features)
        """
        try:
            # Load the .npy file
            data = np.load(fn)
            
            # Convert to torch tensor
            if isinstance(data, np.ndarray):
                x = torch.from_numpy(data).float()
            else:
                x = torch.tensor(data, dtype=torch.float32)
            
            # Handle different data formats
            if x.ndim == 1:
                # If it's 1D (raw audio), handle like original
                start_sample = int(start * self.samplerate)
                if length is not None:
                    length_samples = int(length * self.samplerate)
                    if start_sample + length_samples <= x.size(-1):
                        x = x[start_sample:start_sample + length_samples]
                    else:
                        # Pad if needed
                        x = x[start_sample:]
                        x = tops.force_length(
                            x,
                            length_samples,
                            dim=-1,
                            pad_mode=self.pad_mode,
                            cut_mode="random" if self.augment else "start",
                        )
                else:
                    x = x[start_sample:]
            elif x.ndim == 2:
                # If it's 2D (e.g., CQT features), handle time dimension
                if length is not None:
                    # Assuming second dimension is time
                    time_steps = int(length / 0.02)  # Assuming 20ms hop size
                    start_step = int(start / 0.02)
                    if start_step + time_steps <= x.size(-1):
                        x = x[:, start_step:start_step + time_steps]
                    else:
                        x = x[:, start_step:]
                        x = tops.force_length(
                            x,
                            time_steps,
                            dim=-1,
                            pad_mode=self.pad_mode,
                            cut_mode="random" if self.augment else "start",
                        )
                else:
                    x = x[:, int(start / 0.02):]
            elif x.ndim == 3:
                # If it's 3D (e.g., shingled CQT features), handle appropriately
                if length is not None:
                    time_steps = int(length / 0.02)
                    start_step = int(start / 0.02)
                    if start_step + time_steps <= x.size(-1):
                        x = x[:, :, start_step:start_step + time_steps]
                    else:
                        x = x[:, :, start_step:]
                        x = tops.force_length(
                            x,
                            time_steps,
                            dim=-1,
                            pad_mode=self.pad_mode,
                            cut_mode="random" if self.augment else "start",
                        )
                else:
                    x = x[:, :, int(start / 0.02):]
            
            return x
            
        except Exception as e:
            print(f"Error loading {fn}: {e}")
            # Return zeros as fallback
            if length is not None:
                return torch.zeros(int(length * self.samplerate))
            else:
                return torch.zeros(16000)  # Default 1 second

    ###########################################################################

    def perform_checks(self, splitdict, split):
        msg = ""
        errors = False
        
        # Check if .npy files exist
        missing_files = 0
        for ver in self.info.keys():
            npy_file = self.info[ver]["npy_filename"]
            if not os.path.exists(npy_file):
                missing_files += 1
                if missing_files <= 5:  # Only show first 5 missing files
                    msg += f"\n  {split}: Missing .npy file {npy_file}"
        
        if missing_files > 0:
            msg += f"\n  {split}: Total missing .npy files: {missing_files}"
            errors = True
        
        # Cliques have at least 2 versions
        for cl in self.clique.keys():
            if len(self.clique[cl]) < 2:
                msg += f"\n  {split}: Clique {cl} has < 2 versions"
                errors = True
        
        # No overlap between partitions
        for cl in splitdict[split].keys():
            for partition in ["train", "valid", "test"]:
                if split == partition:
                    continue
                if cl in splitdict[partition]:
                    msg += (
                        f"\n  {split}: Clique {cl} is both in {split} and {partition}"
                    )
        
        if self.verbose and len(msg) > 1:
            print(msg[1:])
        if errors:
            sys.exit() 


def load_kaggle_discogs_split(json_path):
    """
    Loads a Kaggle Discogs split JSON file and returns a list of .npy file paths.
    Assumes JSON structure: {"root": [[file1, file2, ...], ...]}
    Replaces .mm with .npy in file paths.
    """
    with open(json_path, "r") as f:
        data = json.load(f)
    all_files = []
    for group in data["root"]:
        all_files.extend(group)
    npy_files = [x.replace(".mm", ".npy") for x in all_files]
    return npy_files

class KaggleDiscogsNPYDataset(torch.utils.data.Dataset):
    """
    Dataset for Kaggle Discogs CQT data with JSON split files.
    Groups files into cliques by their parent folder (2-char subfolder).
    """
    def __init__(self, split_json, npy_root, augment=False, fullsongs=False, checks=True, verbose=False):
        self.npy_root = npy_root
        self.augment = augment
        self.fullsongs = fullsongs
        self.verbose = verbose
        self.file_list = load_kaggle_discogs_split(split_json)
        # Group by parent folder (clique)
        self.clique_dict = {}
        for path in self.file_list:
            # e.g., Discogs-VI-20240701/magnitude_cqt/cqt/8z/8zlQbNWDOO0.npy
            parts = path.split('/')
            if len(parts) < 2:
                clique = 'unknown'
            else:
                clique = parts[-2]  # 2-char folder as clique
            if clique not in self.clique_dict:
                self.clique_dict[clique] = []
            self.clique_dict[clique].append(path)
        # Flatten for indexing
        self.versions = []
        self.clique2id = {}
        for i, (clique, files) in enumerate(self.clique_dict.items()):
            self.clique2id[clique] = i
            self.versions.extend(files)
        if self.verbose:
            print(f"Loaded {len(self.versions)} files in {len(self.clique_dict)} cliques from {split_json}")

    def __len__(self):
        return len(self.versions)

    def __getitem__(self, idx):
        # Get file path and clique
        npy_rel_path = self.versions[idx]
        clique = npy_rel_path.split('/')[-2] if '/' in npy_rel_path else 'unknown'
        icl = self.clique2id[clique]
        # Load .npy data
        npy_path = os.path.join(self.npy_root, os.path.relpath(npy_rel_path, start="Discogs-VI-20240701/magnitude_cqt/cqt"))
        x = np.load(npy_path)
        x = torch.from_numpy(x).float()
        return icl, idx, x 