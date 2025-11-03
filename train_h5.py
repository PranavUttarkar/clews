import os
import importlib
from omegaconf import OmegaConf
import torch
from lightning import Fabric
from lightning.fabric.strategies import DDPStrategy

from lib.h5_dataset import H5CQTDataset
import multiprocessing as mp
from lib import augmentations, eval
from utils import print_utils, pytorch_utils

# ---------------------------- Args & config ----------------------------
args = OmegaConf.from_cli()
assert "jobname" in args
assert "conf" in args
conf = OmegaConf.merge(OmegaConf.load(args.conf), args)
conf.jobname = args.jobname
conf.data.path = conf.path
conf.path.logs = os.path.join(conf.path.logs, conf.jobname)

os.makedirs(conf.path.logs, exist_ok=True)

# Init pytorch/Fabric
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.set_float32_matmul_precision("medium")

def get_logger(logdir):
    return pytorch_utils.get_logger(logdir)

fabric = Fabric(
    accelerator="cuda",
    devices=conf.fabric.ngpus,
    num_nodes=conf.fabric.nnodes,
    strategy=DDPStrategy(broadcast_buffers=False),
    precision=conf.fabric.precision,
    loggers=get_logger(conf.path.logs),
)
fabric.launch()

fabric.barrier()
fabric.seed_everything(conf.seed, workers=True)

myprint = lambda s, end="\n": print_utils.myprint(s, end=end, doit=fabric.is_global_zero)
myprogbar = lambda it, desc=None, leave=False: print_utils.myprogbar(it, desc=desc, leave=leave, doit=fabric.is_global_zero)
myprint("-" * 65)
myprint(OmegaConf.to_yaml(conf)[:-1])
myprint("-" * 65)

# ---------------------------- Model ----------------------------
myprint("Init model...")
module = importlib.import_module("models." + conf.model.name)
with fabric.init_module():
    model = module.Model(conf.model, sr=conf.data.samplerate)
model = fabric.setup(model)
model.mark_forward_method("embed")
model.mark_forward_method("loss")

# Optim & sched
optim = pytorch_utils.get_optimizer(conf.training.optim, model)
optim = fabric.setup_optimizers(optim)
sched, sched_on_epoch = pytorch_utils.get_scheduler(
    conf.training.optim,
    optim,
    epochs=conf.training.numepochs,
    mode=conf.training.monitor.mode,
)

# ---------------------------- Data ----------------------------
myprint("Load data...")
# Inject model shingle params and feature hop to data conf for shingling
conf.data.shingle_len = conf.model.shingling.len
conf.data.shingle_hop = conf.model.shingling.hop
conf.data.feature_hoplen = conf.model.cqt.hoplen

ds_train = H5CQTDataset(conf.data, "train", augment=True, verbose=fabric.is_global_zero)

# valid uses "test" split name mapping to verify.json per request; keep "valid" here to match config
split_valid = "valid"
ds_valid = H5CQTDataset(conf.data, split_valid, augment=False, verbose=fabric.is_global_zero)

assert conf.training.batchsize > 1

mp_ctx_name = getattr(conf.data, "mp_context", "spawn")
mp_ctx = mp.get_context(mp_ctx_name) if mp_ctx_name else None

loader_kwargs = dict(
    batch_size=conf.training.batchsize,
    num_workers=conf.data.nworkers,
    persistent_workers=True,
    prefetch_factor=getattr(conf.data, "prefetch_factor", 4),
    pin_memory=True,
)
if mp_ctx is not None:
    loader_kwargs["multiprocessing_context"] = mp_ctx

dl_train = torch.utils.data.DataLoader(ds_train, shuffle=True, drop_last=True, **loader_kwargs)

dl_valid = torch.utils.data.DataLoader(ds_valid, shuffle=False, drop_last=False, **loader_kwargs)

# setup dataloaders
dl_train, dl_valid = fabric.setup_dataloaders(dl_train, dl_valid)

# CQ augmentations only
augment = augmentations.Augment(conf.augmentations, sr=conf.data.samplerate)

# ---------------------------- Train/valid step ----------------------------

def main_loss_func(batch, logdict, training=False):
    with torch.inference_mode():
        n_per_class = (len(batch) - 1) // 2
        cc = [batch[0]] * n_per_class
        cc = torch.cat(cc, dim=0)
        ii = torch.cat(batch[1::2], dim=0)
        xx = batch[2::2]
        xx = torch.stack(xx, dim=0)  # (B, S, C, T)
        if training:
            xx = augment.cqgram(xx)
    cc, ii, xx = cc.clone(), ii.clone(), xx.clone()
    if training:
        optim.zero_grad(set_to_none=True)
    zz, extra = model.embed(xx)
    loss, logdct = model.loss(cc, ii, zz, extra=extra)
    if training:
        fabric.backward(loss)
        optim.step()
        if not sched_on_epoch:
            sched.step()
    with torch.inference_mode():
        clist = torch.chunk(cc, n_per_class, dim=0)
        ilist = torch.chunk(ii, n_per_class, dim=0)
        zlist = torch.chunk(zz, n_per_class, dim=0)
        outputs = [clist[0]] + [None] * (2 * n_per_class)
        outputs[1::2] = ilist
        outputs[2::2] = zlist
        logdict.append(logdct)
    return outputs, logdict

# ---------------------------- Loops ----------------------------

def run_epoch(dl, training: bool, desc: str):
    model.train(training)
    logdict = pytorch_utils.LogDict()
    fabric.barrier()
    for n, batch in enumerate(myprogbar(dl, desc=desc)):
        if conf.limit_batches is not None and n >= conf.limit_batches:
            break
        _, logdict = main_loss_func(batch, logdict, training=training)
        if training:
            losses = logdict.get("l_main")
            myprint(f" [L*={losses[-1]:.3f}, L={losses.mean():.3f}]", end="")
    return logdict

# ---------------------------- Training ----------------------------
fn_ckpt_last = os.path.join(conf.path.logs, "checkpoint_last.ckpt")
fn_ckpt_best = os.path.join(conf.path.logs, "checkpoint_best.ckpt")
fn_ckpt_epoch = os.path.join(conf.path.logs, "checkpoint_$epoch$.ckpt")

epoch = 0
cost_best = torch.inf if conf.training.monitor.mode == "min" else -torch.inf
if conf.training.optim.sched.startswith("plateau"):
    sched.step(cost_best)

myprint("Training...")
for epoch in range(conf.training.numepochs):
    desc = f"{epoch+1:{len(str(conf.training.numepochs))}d}/{conf.training.numepochs}"
    fabric.log("hpar/epoch", epoch + 1, step=epoch + 1)
    # Train
    log_t = run_epoch(dl_train, True, desc="Train " + desc)
    log_t.sync_and_mean(fabric)
    fabric.log_dict(log_t.get(prefix="train/"), step=epoch + 1)
    # Valid
    log_v = run_epoch(dl_valid, False, desc="Valid " + desc)
    log_v.sync_and_mean(fabric)
    fabric.log_dict(log_v.get(prefix="valid/"), step=epoch + 1)
    # Monitor
    cost_current = log_v.get(conf.training.monitor.quantity)
    fabric.log("hpar/lr", sched.get_last_lr()[0], step=epoch + 1)
    if conf.training.optim.sched.startswith("plateau"):
        sched.step(cost_current)
    else:
        sched.step()
    # Checkpoint
    state = pytorch_utils.get_state(model, optim, sched, conf, epoch + 1, sched.get_last_lr()[0], cost_best)
    fabric.save(fn_ckpt_last, state)
    if (conf.training.monitor.mode == "max" and cost_current > cost_best) or (
        conf.training.monitor.mode == "min" and cost_current < cost_best
    ):
        cost_best = cost_current
        fabric.save(fn_ckpt_best, state)

myprint("Done.")
