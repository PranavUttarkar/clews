import os
import re
import json
import argparse
import torch

"""
Build an index mapping youtube_id -> source pointers for Discogs-VI features.

Inputs:
- --jsonl: path to Discogs-VI-20240701.jsonl
- --h5-root: directory containing discogs_vi_chunk_XXXXX.h5 files
- --mm-root: optional directory containing magnitude_cqt/cqt shards
- --out: output path (default cache/discogs_h5_index.pt)

The script is conservative: it records what it can find. Training will try H5 first, then MM.
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True)
    ap.add_argument("--h5-root", required=True)
    ap.add_argument("--mm-root", default=None)
    ap.add_argument("--out", default=os.path.join("cache", "discogs_h5_index.pt"))
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    # Discover H5 files
    h5_files = [f for f in os.listdir(args.h5_root) if f.endswith('.h5')]
    h5_set = set(h5_files)

    # Known patterns inside H5 may include dataset per-youtube-id or groups
    # We'll record the h5_file name only (not opening all files) and rely on JSONL hints

    index = {}

    # Parse JSONL lines for youtube_id and feature pointers
    with open(args.jsonl, 'r', encoding='utf-8') as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                rec = json.loads(ln)
            except Exception:
                continue
            ytid = rec.get('youtube_id') or rec.get('ytid') or rec.get('youtubeId')
            if not ytid:
                continue
            entry = index.get(ytid, {})

            # Possible fields that hint the h5 file and key
            # e.g., rec['h5_chunk'] = 'discogs_vi_chunk_00001.h5', rec['h5_key'] = f"{ytid}/cqt"
            h5_file = rec.get('h5_chunk') or rec.get('h5_file')
            h5_key = rec.get('h5_key')
            if h5_file and h5_file in h5_set:
                entry['h5_file'] = h5_file
                if h5_key:
                    entry['h5_key'] = h5_key

            # MM path if present
            mm_path = rec.get('mm_path') or rec.get('cqt_mm_path')
            if mm_path:
                entry['mm_path'] = mm_path

            # Hop (seconds per frame) and shape
            hop_s = rec.get('hop_s') or rec.get('hoplen_s') or rec.get('frame_hop_s')
            if isinstance(hop_s, (int, float)):
                entry['hop_s'] = float(hop_s)
            shape = rec.get('shape') or rec.get('cqt_shape')
            if isinstance(shape, (list, tuple)) and len(shape) == 2:
                entry['shape'] = [int(shape[0]), int(shape[1])]

            if entry:
                index[ytid] = entry

    # Optional: guess mm_path if mm-root exists and entry has no pointers
    if args.mm_root:
        for ytid, entry in list(index.items()):
            if 'h5_file' not in entry and 'mm_path' not in entry:
                shard = ytid[:2]
                cand = os.path.join(args.mm_root, shard, f"{ytid}.mm")
                if os.path.exists(cand):
                    entry['mm_path'] = cand
                    index[ytid] = entry

    torch.save(index, args.out)
    print(f"Wrote index with {len(index)} entries to {args.out}")


if __name__ == "__main__":
    main()
