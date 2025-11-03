import sys
import os
import importlib
from omegaconf import OmegaConf
import torch
from lightning import Fabric
from lightning.fabric.strategies import DDPStrategy
from torch.utils.data import DataLoader
from lib.npy_dataset import DiscogsNPYDataset
from lib import augmentations, eval
from utils import print_utils, pytorch_utils

# --- Get arguments ---
args = OmegaConf.from_cli()
assert "jobname" in args
assert "conf" in args
conf = OmegaConf.merge(OmegaConf.load(args.conf), args)
conf.jobname = args.jobname
conf.data.path = conf.path

# Use regular logs directory
conf.path.logs = os.path.join(conf.path.logs, conf.jobname)

fn_ckpt_last = os.path.join(conf.path.logs, "checkpoint_last.ckpt")
fn_ckpt_best = os.path.join(conf.path.logs, "checkpoint_best.ckpt")
fn_ckpt_epoch = os.path.join(conf.path.logs, "checkpoint_$epoch$.ckpt")

# Create logs directory
os.makedirs(conf.path.logs, exist_ok=True)

# Copy configuration file to logs directory for later use
import shutil
config_save_path = os.path.join(conf.path.logs, "configuration.yaml")
shutil.copy2(args.conf, config_save_path)

# Init pytorch/Fabric
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.set_float32_matmul_precision("medium")
torch.autograd.set_detect_anomaly(False)
fabric = Fabric(
    accelerator="cuda",
    devices=conf.fabric.ngpus,
    num_nodes=conf.fabric.nnodes,
    strategy=DDPStrategy(broadcast_buffers=False),
    precision=conf.fabric.precision,
    loggers=pytorch_utils.get_logger(conf.path.logs),
)
fabric.launch()

# Seed
fabric.barrier()
fabric.seed_everything(conf.seed, workers=True)

# Print config
myprint = lambda s, end="\n": print_utils.myprint(s, end=end, doit=fabric.is_global_zero)
myprint("-" * 65)
myprint(OmegaConf.to_yaml(conf))
myprint("-" * 65)

# Datasets
train_json = getattr(args, "train_json", "metadata/train.json")
valid_json = getattr(args, "valid_json", "metadata/valid.json")
train_dataset = DiscogsNPYDataset(train_json, conf.path.npy, augment=True, verbose=fabric.is_global_zero)
valid_dataset = DiscogsNPYDataset(valid_json, conf.path.npy, augment=False, verbose=fabric.is_global_zero)

train_loader = DataLoader(
    train_dataset,
    batch_size=conf.training.batchsize,
    shuffle=True,
    num_workers=conf.data.nworkers,
    drop_last=True,
    persistent_workers=False,
    pin_memory=True,
)
valid_loader = DataLoader(
    valid_dataset,
    batch_size=conf.training.batchsize,
    shuffle=False,
    num_workers=conf.data.nworkers,
    drop_last=False,
    persistent_workers=False,
    pin_memory=True,
)
train_loader, valid_loader = fabric.setup_dataloaders(train_loader, valid_loader)

# Model
myprint("Init model...")
module = importlib.import_module("models." + conf.model.name)
with fabric.init_module():
    model = module.Model(conf.model, sr=conf.data.samplerate)
model = fabric.setup(model)

# Optimizer
optim = torch.optim.Adam(model.parameters(), lr=conf.training.optim.lr, weight_decay=conf.training.optim.wd)

# Augmentations
augment = augmentations.Augment(conf.augmentations, sr=conf.data.samplerate)

# Training loop (simplified)
myprint("Start training...")
for epoch in range(conf.training.numepochs):
    model.train()
    epoch_loss = 0.0
    nbatches = 0
    for batch in train_loader:
        optim.zero_grad(set_to_none=True)
        cc, ii, xx = batch[:3]
        
        # Debug: print shape of first few batches
        if nbatches < 3 and fabric.is_global_zero:
            myprint(f"Batch {nbatches}: Input tensor shape: {xx.shape}")
            myprint(f"Batch {nbatches}: Input tensor dtype: {xx.dtype}")
            myprint(f"Batch {nbatches}: Input tensor min/max: {xx.min():.4f}/{xx.max():.4f}")
        
        # Apply augmentations if needed
        if hasattr(augment, 'waveform'):
            xx = augment.waveform(xx)
        # Forward - handle different input formats
        if xx.ndim == 2:
            if xx.shape[1] == 2:
                # This looks like metadata - try to use it as features
                if fabric.is_global_zero and nbatches < 3:
                    myprint(f"Using metadata as features: {xx.shape}")
                # Convert metadata to a format the model can use
                # Pad or repeat to create a longer sequence
                target_length = 16000  # 1 second at 16kHz
                if xx.shape[1] < target_length:
                    # Repeat the metadata to reach target length
                    repeats = (target_length + xx.shape[1] - 1) // xx.shape[1]
                    xx = xx.repeat(1, repeats)
                    xx = xx[:, :target_length]
                zz = model.forward(xx)
                extra = None
        elif xx.ndim == 4:
            # Already preprocessed (B, S, C, T) - use embed directly
            zz, extra = model.embed(xx)
        else:
            # Try to reshape to expected format
            if xx.ndim == 3:
                # Assume (B, C, T) and add shingle dimension
                xx = xx.unsqueeze(1)  # (B, 1, C, T)
                zz, extra = model.embed(xx)
            else:
                raise ValueError(f"Unexpected input shape: {xx.shape}")
        # CLEWS loss function
        loss, logdict = model.loss(cc, ii, zz, extra=extra)
        fabric.backward(loss)
        optim.step()
        epoch_loss += loss.item()
        nbatches += 1
    if nbatches > 0:
        myprint(f"Epoch {epoch+1} done. Avg loss: {epoch_loss/nbatches:.4f}")
    else:
        myprint(f"Epoch {epoch+1} done. No valid batches processed - all batches contained metadata instead of audio data")
    # Save checkpoint
    if fabric.is_global_zero:
        torch.save({'model': model.state_dict()}, fn_ckpt_last)
    # Validation (optional, add your own logic)
    model.eval()
    with torch.no_grad():
        for batch in valid_loader:
            cc, ii, xx = batch[:3]
            # Forward - handle different input formats
            if xx.ndim == 2:
                if xx.shape[1] == 2:
                    # This looks like metadata - try to use it as features
                    # Convert metadata to a format the model can use
                    target_length = 16000  # 1 second at 16kHz
                    if xx.shape[1] < target_length:
                        # Repeat the metadata to reach target length
                        repeats = (target_length + xx.shape[1] - 1) // xx.shape[1]
                        xx = xx.repeat(1, repeats)
                        xx = xx[:, :target_length]
                # Raw audio (B, T) - use forward which handles preprocessing
                zz = model.forward(xx)
                extra = None
            elif xx.ndim == 4:
                # Already preprocessed (B, S, C, T) - use embed directly
                zz, extra = model.embed(xx)
            else:
                # Try to reshape to expected format
                if xx.ndim == 3:
                    # Assume (B, C, T) and add shingle dimension
                    xx = xx.unsqueeze(1)  # (B, 1, C, T)
                    zz, extra = model.embed(xx)
                else:
                    raise ValueError(f"Unexpected input shape: {xx.shape}")
            # Compute validation loss/metrics as needed
    # Save best checkpoint logic can be added here
myprint("Training complete.") 