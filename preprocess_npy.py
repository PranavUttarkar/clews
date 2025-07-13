#!/usr/bin/env python3
"""
Script to preprocess .npy files for CLEWS training.

This script helps convert existing .npy files to the format expected by the CLEWS system.
It assumes you have a folder of .npy files and creates the necessary metadata structure.
"""

import os
import sys
import argparse
import numpy as np
import torch
from tqdm import tqdm
import json
 
def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess .npy files for CLEWS")
    parser.add_argument("--input_dir", type=str, required=True,
                       help="Directory containing .npy files")
    parser.add_argument("--output_dir", type=str, required=True,
                       help="Output directory for processed files")
    parser.add_argument("--metadata_file", type=str, required=True,
                       help="Path to save metadata file")
    parser.add_argument("--dataset_name", type=str, default="custom",
                       help="Name of the dataset")
    parser.add_argument("--sample_rate", type=int, default=16000,
                       help="Audio sample rate (for metadata)")
    parser.add_argument("--audio_length", type=float, default=150.0,
                       help="Default audio length in seconds")
    parser.add_argument("--split_ratio", type=str, default="0.7,0.15,0.15",
                       help="Train, validation, test split ratios (comma-separated)")
    parser.add_argument("--clique_size", type=int, default=2,
                       help="Minimum number of versions per clique")
    return parser.parse_args()

def load_npy_file(filepath):
    """Load a .npy file and return its shape and data type."""
    try:
        data = np.load(filepath)
        return data.shape, data.dtype
    except Exception as e:
        print(f"Error loading {filepath}: {e}")
        return None, None

def analyze_npy_files(input_dir):
    """Analyze all .npy files in the input directory."""
    print(f"Analyzing .npy files in {input_dir}...")
    
    files = []
    for root, dirs, filenames in os.walk(input_dir):
        for filename in filenames:
            if filename.endswith('.npy'):
                filepath = os.path.join(root, filename)
                shape, dtype = load_npy_file(filepath)
                if shape is not None:
                    files.append({
                        'path': filepath,
                        'filename': filename,
                        'shape': shape,
                        'dtype': dtype,
                        'size': os.path.getsize(filepath)
                    })
    
    print(f"Found {len(files)} .npy files")
    return files

def create_cliques(files, clique_size=2):
    """Create cliques based on filename patterns or other criteria."""
    print("Creating cliques...")
    
    # This is a simple example - you may need to customize this based on your data
    # Here we assume files with similar names (before extension) belong to the same clique
    
    cliques = {}
    for file_info in files:
        # Extract base name (remove .npy extension)
        base_name = os.path.splitext(file_info['filename'])[0]
        
        # Simple clustering: group by first part of filename
        # You may need to adjust this based on your naming convention
        clique_key = base_name.split('_')[0] if '_' in base_name else base_name[:8]
        
        if clique_key not in cliques:
            cliques[clique_key] = []
        cliques[clique_key].append(file_info)
    
    # Filter cliques that are too small
    valid_cliques = {}
    for clique_key, clique_files in cliques.items():
        if len(clique_files) >= clique_size:
            valid_cliques[clique_key] = clique_files
    
    print(f"Created {len(valid_cliques)} valid cliques (min size: {clique_size})")
    return valid_cliques

def create_splits(cliques, split_ratio):
    """Create train/validation/test splits."""
    print("Creating splits...")
    
    # Parse split ratios
    ratios = [float(x) for x in split_ratio.split(',')]
    assert len(ratios) == 3, "Must provide exactly 3 split ratios"
    assert abs(sum(ratios) - 1.0) < 1e-6, "Split ratios must sum to 1.0"
    
    clique_keys = list(cliques.keys())
    np.random.shuffle(clique_keys)
    
    n_total = len(clique_keys)
    n_train = int(ratios[0] * n_total)
    n_val = int(ratios[1] * n_total)
    
    train_cliques = clique_keys[:n_train]
    val_cliques = clique_keys[n_train:n_train + n_val]
    test_cliques = clique_keys[n_train + n_val:]
    
    splits = {
        'train': {k: cliques[k] for k in train_cliques},
        'valid': {k: cliques[k] for k in val_cliques},
        'test': {k: cliques[k] for k in test_cliques}
    }
    
    print(f"Split sizes: Train={len(train_cliques)}, Val={len(val_cliques)}, Test={len(test_cliques)}")
    return splits

def create_metadata(files, cliques, splits, sample_rate, audio_length):
    """Create metadata structure expected by CLEWS."""
    print("Creating metadata...")
    
    # Create info dictionary
    info = {}
    version_id = 0
    
    for clique_key, clique_files in cliques.items():
        for i, file_info in enumerate(clique_files):
            # Create version ID
            version_name = f"v{i+1}"
            version_id_str = f"{clique_key}-{version_name}"
            
            # Create relative path (relative to output directory)
            rel_path = os.path.relpath(file_info['path'], args.output_dir)
            rel_path = os.path.splitext(rel_path)[0]  # Remove .npy extension
            
            info[version_id_str] = {
                'id': version_id,
                'clique': clique_key,
                'version': version_name,
                'artist': clique_key,  # You may want to extract this from filename
                'title': file_info['filename'],
                'filename': rel_path + '.mp3',  # Keep original extension for compatibility
                'npy_filename': rel_path + '.npy',  # Add .npy filename
                'samplerate': sample_rate,
                'length': audio_length,
                'channels': 1,
                'shape': file_info['shape'],
                'dtype': str(file_info['dtype'])
            }
            version_id += 1
    
    return info, splits

def copy_files_to_output(files, output_dir):
    """Copy .npy files to output directory maintaining structure."""
    print(f"Copying files to {output_dir}...")
    
    os.makedirs(output_dir, exist_ok=True)
    
    for file_info in tqdm(files, desc="Copying files"):
        # Create relative path
        rel_path = os.path.relpath(file_info['path'], args.input_dir)
        output_path = os.path.join(output_dir, rel_path)
        
        # Create output directory
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Copy file
        import shutil
        shutil.copy2(file_info['path'], output_path)

def main():
    global args
    args = parse_args()
    
    # Check input directory
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory {args.input_dir} does not exist")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Analyze .npy files
    files = analyze_npy_files(args.input_dir)
    if not files:
        print("No .npy files found!")
        sys.exit(1)
    
    # Create cliques
    cliques = create_cliques(files, args.clique_size)
    if not cliques:
        print("No valid cliques found!")
        sys.exit(1)
    
    # Create splits
    splits = create_splits(cliques, args.split_ratio)
    
    # Create metadata
    info, splits = create_metadata(files, cliques, splits, args.sample_rate, args.audio_length)
    
    # Copy files to output directory
    copy_files_to_output(files, args.output_dir)
    
    # Save metadata
    print(f"Saving metadata to {args.metadata_file}...")
    torch.save([info, splits], args.metadata_file)
    
    # Print summary
    print("\n" + "="*50)
    print("PREPROCESSING SUMMARY")
    print("="*50)
    print(f"Dataset: {args.dataset_name}")
    print(f"Total files: {len(files)}")
    print(f"Total cliques: {len(cliques)}")
    print(f"Total versions: {len(info)}")
    print(f"Train split: {len(splits['train'])} cliques")
    print(f"Validation split: {len(splits['valid'])} cliques")
    print(f"Test split: {len(splits['test'])} cliques")
    print(f"Output directory: {args.output_dir}")
    print(f"Metadata file: {args.metadata_file}")
    print("="*50)
    
    # Print example file shapes
    print("\nExample file shapes:")
    for i, (version_id, version_info) in enumerate(info.items()):
        if i < 5:  # Show first 5 examples
            print(f"  {version_id}: {version_info['shape']}")
        else:
            break
    
    print("\nPreprocessing completed successfully!")
    print("\nNext steps:")
    print("1. Update your config file to point to the output directory")
    print("2. Set the 'npy' path in your config to point to the output directory")
    print("3. Set the 'meta' path in your config to point to the metadata file")
    print("4. Run training with: python train_npy.py jobname=your-job conf=config/your-config.yaml")

if __name__ == "__main__":
    main() 