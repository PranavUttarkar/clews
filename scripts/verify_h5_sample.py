import os
import json
import argparse
import torch
from omegaconf import OmegaConf
from lib.h5_dataset import H5CQTDataset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--conf', required=True, help='Path to config YAML (e.g., config/dvi-h5.yaml)')
    ap.add_argument('--split', default='train', choices=['train','valid','test'])
    ap.add_argument('--index', type=int, default=0, help='Sample index to inspect')
    ap.add_argument('--verbose', action='store_true')
    args = ap.parse_args()

    conf = OmegaConf.load(args.conf)
    conf.data.path = conf.path
    # Inject model shingle + hop
    conf.data.shingle_len = conf.model.shingling.len
    conf.data.shingle_hop = conf.model.shingling.hop
    conf.data.feature_hoplen = conf.model.cqt.hoplen

    ds = H5CQTDataset(conf.data, args.split, augment=False, verbose=args.verbose)
    print(f"Dataset {args.split}: {len(ds)} versions, {len(ds.clique)} cliques")
    sample = ds[args.index]
    cc = sample[0]
    ii = sample[1]
    xx = sample[2]
    print(f"clique_label: {cc.shape} -> {cc}")
    print(f"version_id(int): {ii.shape} -> {ii}")
    print(f"features (S,C,T): {xx.shape} dtype={xx.dtype} min={xx.min():.3f} max={xx.max():.3f}")


if __name__ == '__main__':
    main()
