import sys
import os
import importlib
from omegaconf import OmegaConf
import torch
from lightning import Fabric
from lightning.fabric.strategies import DDPStrategy
from torch.utils.data import DataLoader
from lib.npy_dataset import KaggleDiscogsNPYDataset
from lib import augmentations, eval
from utils import print_utils, pytorch_utils

# --- Get arguments ---
args = OmegaConf.from_cli()
assert "jobname" in args
assert "conf" in args
conf = OmegaConf.merge(OmegaConf.load(args.conf), args)
conf.jobname = args.jobname
conf.data.path = conf.path
conf.path.logs = os.path.join(conf.path.logs, conf.jobname)
fn_ckpt_last = os.path.join(conf.path.logs, "checkpoint_last.ckpt")
fn_ckpt_best = os.path.join(conf.path.logs, "checkpoint_best.ckpt")
fn_ckpt_epoch = os.path.join(conf.path.logs, "checkpoint_$epoch$.ckpt")

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
train_dataset = KaggleDiscogsNPYDataset(train_json, conf.path.npy, augment=True, verbose=fabric.is_global_zero)
valid_dataset = KaggleDiscogsNPYDataset(valid_json, conf.path.npy, augment=False, verbose=fabric.is_global_zero)

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
        # Apply augmentations if needed
        if hasattr(augment, 'waveform'):
            xx = augment.waveform(xx)
        # Forward
        zz, extra = model.embed(xx)
        # CLEWS loss function
        loss, logdict = model.loss(cc, ii, zz, extra=extra)
        fabric.backward(loss)
        optim.step()
        epoch_loss += loss.item()
        nbatches += 1
    myprint(f"Epoch {epoch+1} done. Avg loss: {epoch_loss/nbatches:.4f}")
    # Save checkpoint
    if fabric.is_global_zero:
        torch.save(model.state_dict(), fn_ckpt_last)
    # Validation (optional, add your own logic)
    model.eval()
    with torch.no_grad():
        for batch in valid_loader:
            cc, ii, xx = batch[:3]
            zz, extra = model.embed(xx)
            # Compute validation loss/metrics as needed
    # Save best checkpoint logic can be added here
myprint("Training complete.") 