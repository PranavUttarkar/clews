# CLEWS with Preprocessed .npy Files

This guide explains how to use the CLEWS (Contrastive Learning from Weakly-labeled Audio Segments) system with preprocessed .npy files instead of raw audio files.

## Overview

The CLEWS system has been modified to work with preprocessed .npy files, which can significantly speed up training by avoiding real-time audio processing. This is especially useful when you have already extracted features from your audio data.

## System Requirements

- Python >= 3.10
- CUDA-compatible GPU (recommended)
- At least 16GB RAM
- Sufficient disk space for your .npy files

## Installation

1. **Clone the repository** (if you haven't already):
   ```bash
   git clone <repository-url>
   cd clews
   ```

2. **Create and activate virtual environment**:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install torch==2.3.1 torchaudio==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu121
   pip install lightning==2.3.0 tensorboard==2.17.0 einops==0.8.0 torchinfo==1.8.0
   pip install omegaconf==2.3.0 tqdm==4.66.4 joblib==1.4.2
   pip install soundfile==0.12.1 soxr==0.3.7 nnAudio==0.3.3
   pip install numpy==1.26.4 julius==0.2.7
   ```

## Data Format Requirements

### .npy File Format

Your .npy files should contain one of the following formats:

1. **Raw Audio (1D)**: `(samples,)` - Raw audio samples at 16kHz
2. **CQT Features (2D)**: `(frequency_bins, time_steps)` - Constant-Q Transform features
3. **Shingled CQT (3D)**: `(shingles, frequency_bins, time_steps)` - Pre-shingled CQT features

### File Organization

Organize your .npy files in a directory structure like this:
```
your_npy_data/
├── song1_version1.npy
├── song1_version2.npy
├── song2_version1.npy
├── song2_version2.npy
└── ...
```

### Naming Convention

The system expects files to be organized in cliques (groups of similar songs). You can customize the clique creation logic in `preprocess_npy.py` based on your naming convention.

## Step-by-Step Usage

### Step 1: Preprocess Your .npy Files

Use the provided preprocessing script to organize your .npy files and create the necessary metadata:

```bash
python preprocess_npy.py \
    --input_dir /path/to/your/npy/files \
    --output_dir data/processed_npy/ \
    --metadata_file cache/metadata-custom.pt \
    --dataset_name "my_dataset" \
    --sample_rate 16000 \
    --audio_length 150.0 \
    --split_ratio "0.7,0.15,0.15" \
    --clique_size 2
```

**Parameters:**
- `--input_dir`: Directory containing your .npy files
- `--output_dir`: Where to store the organized files
- `--metadata_file`: Path to save the metadata file
- `--dataset_name`: Name for your dataset
- `--sample_rate`: Audio sample rate (for metadata)
- `--audio_length`: Default audio length in seconds
- `--split_ratio`: Train/validation/test split ratios
- `--clique_size`: Minimum number of versions per clique

### Step 2: Create Configuration File

Create a configuration file for your dataset. You can use `config/shs-clews-npy.yaml` as a template:

```yaml
jobname: null
seed: 43
checkpoint: null
limit_batches: null

path:
  cache: "cache/"
  logs: "logs/"
  audio: "data/processed_npy/"  # Keep for compatibility
  npy: "data/processed_npy/"    # Path to your .npy files
  meta: "cache/metadata-custom.pt"  # Your metadata file

fabric:
  nnodes: 1
  ngpus: 1
  precision: "32"

data:
  nworkers: 16
  samplerate: 16000
  audiolen: 150
  maxlen: null
  pad_mode: "repeat"
  n_per_class: 4
  p_samesong: 0

# ... rest of configuration (same as original)
```

### Step 3: Train the Model

Run training with your configuration:

```bash
python train_npy.py \
    jobname=my-custom-model \
    conf=config/my-custom-config.yaml \
    fabric.nnodes=1 \
    fabric.ngpus=2
```

### Step 4: Evaluate the Model

Test your trained model:

```bash
python test.py \
    jobname=test-script \
    checkpoint=logs/my-custom-model/checkpoint_best.ckpt \
    nnodes=1 \
    ngpus=4 \
    redux=bpwr-10 \
    path_meta=cache/metadata-custom.pt
```

## Customization

### Modifying Clique Creation

If your files have a different naming convention, modify the `create_cliques` function in `preprocess_npy.py`:

```python
def create_cliques(files, clique_size=2):
    """Customize this function based on your naming convention."""
    cliques = {}
    for file_info in files:
        # Example: files named "artist_song_version.npy"
        parts = file_info['filename'].split('_')
        if len(parts) >= 2:
            clique_key = f"{parts[0]}_{parts[1]}"  # artist_song
        else:
            clique_key = file_info['filename'][:8]
        
        if clique_key not in cliques:
            cliques[clique_key] = []
        cliques[clique_key].append(file_info)
    
    # Filter small cliques
    return {k: v for k, v in cliques.items() if len(v) >= clique_size}
```

### Handling Different Data Formats

If your .npy files have a different format, modify the `get_npy_data` method in `lib/npy_dataset.py`:

```python
def get_npy_data(self, fn, start=0, length=None):
    """Customize this method based on your data format."""
    data = np.load(fn)
    x = torch.from_numpy(data).float()
    
    # Handle your specific format here
    if x.ndim == 2 and x.size(0) == 84:  # Example: 84 frequency bins
        # Your custom processing logic
        pass
    
    return x
```

## Troubleshooting

### Common Issues

1. **Memory Errors**: Reduce batch size or number of workers in your config
2. **Missing Files**: Check that all .npy files exist and are readable
3. **Shape Mismatches**: Ensure your .npy files have consistent shapes within cliques
4. **CUDA Out of Memory**: Reduce model size or batch size

### Debug Mode

To debug data loading issues, you can modify the dataset to print more information:

```python
# In lib/npy_dataset.py, add debug prints
def get_npy_data(self, fn, start=0, length=None):
    print(f"Loading {fn}, shape: {data.shape}")
    # ... rest of method
```

## Performance Tips

1. **Use SSD Storage**: .npy files load faster from SSDs
2. **Increase Workers**: Set `nworkers` to 2-4x your number of CPU cores
3. **Use Mixed Precision**: Set `precision: "16"` in config for faster training
4. **Pre-shingle Data**: If possible, pre-compute shingled features to avoid real-time processing

## Example Workflow

Here's a complete example workflow:

```bash
# 1. Preprocess your data
python preprocess_npy.py \
    --input_dir /path/to/my/npy/files \
    --output_dir data/my_dataset/ \
    --metadata_file cache/metadata-my_dataset.pt \
    --dataset_name "my_dataset"

# 2. Create logs directory
mkdir -p logs/my-custom-model/

# 3. Copy and modify config
cp config/shs-clews-npy.yaml config/my-custom-config.yaml
# Edit config/my-custom-config.yaml to point to your paths

# 4. Train
python train_npy.py \
    jobname=my-custom-model \
    conf=config/my-custom-config.yaml \
    fabric.ngpus=2

# 5. Test
python test.py \
    jobname=test-my-model \
    checkpoint=logs/my-custom-model/checkpoint_best.ckpt \
    fabric.ngpus=2 \
    path_meta=cache/metadata-my_dataset.pt
```

## Support

For issues specific to .npy file usage:
1. Check the preprocessing script output for errors
2. Verify your .npy file formats and shapes
3. Ensure your naming convention is compatible with clique creation
4. Check that all paths in your config file are correct

For general CLEWS issues, refer to the main README.md file. 