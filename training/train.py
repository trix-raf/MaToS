#!/usr/bin/env python
"""
train.py - train the sign encoder on the prepared arrays.

Single process:   python train.py --data processed --out runs/exp1
Cluster / multi-process (what run_train.sh does):
                  torchrun --nproc_per_node=4 train.py --data processed --out runs/exp1
Multi node:       torchrun --nnodes=N ... (see run_train.sh)

* CPU-first (gloo backend); uses CUDA + nccl automatically if GPUs are visible.
* Validation is signer-independent: whole signers are held out.
* Every epoch: last.pt (resume with --resume auto), best.pt (EMA weights), metrics.csv.
"""
import argparse
import copy
import csv
import json
import math
import os
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import asl_core as A
from data import ClipStore, WindowDataset, WindowSampler
from model import SignEncoder, n_params


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch-size", type=int, default=512, help="GLOBAL batch size (split across processes)")
    ap.add_argument("--lr", type=float, default=2e-3, help="peak LR for the global batch")
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--warmup-epochs", type=float, default=3)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--ema", type=float, default=0.999)
    ap.add_argument("--window", type=int, default=64)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--neg-ratio", type=float, default=0.35, help="'no-sign' windows per positive window")
    ap.add_argument("--val-signers", type=int, default=3, help="number of Kaggle signers held out for validation")
    ap.add_argument("--val-ids", default="", help="comma separated signer ids to hold out (overrides --val-signers)")
    ap.add_argument("--extra-repeat", type=int, default=3, help="repeat clips from --extra-dir this many times per epoch")
    ap.add_argument("--workers", type=int, default=2, help="data-loader workers per process")
    ap.add_argument("--threads", type=int, default=0, help="torch threads per process (0 = auto)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", default="", help="path to last.pt, or 'auto'")
    ap.add_argument("--init-from", default="", help="start from these weights (fine-tuning), fresh optimiser")
    ap.add_argument("--time-budget-min", type=float, default=0, help="stop cleanly (checkpoint saved) after this many minutes")
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--max-steps-per-epoch", type=int, default=0, help="debug")
    return ap.parse_args()


class Ema:
    def __init__(self, model, decay):
        self.model = copy.deepcopy(model).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.decay, self.n = decay, 0

    @torch.no_grad()
    def update(self, model):
        self.n += 1
        d = min(self.decay, (1 + self.n) / (10 + self.n))
        ep, mp = list(self.model.parameters()), [p.detach() for p in model.parameters()]
        torch._foreach_mul_(ep, d)
        torch._foreach_add_(ep, mp, alpha=1 - d)


def load_weights(model, sd):
    """Load weights; if the head has more outputs (signs were appended), keep old rows and the old no-sign row."""
    own = model.state_dict()
    sd = dict(sd)
    for k in ("head.weight", "head.bias"):
        if sd[k].shape != own[k].shape:
            old, new = sd[k], own[k].clone()
            n_old = old.shape[0] - 1
            if n_old > new.shape[0] - 1:
                raise SystemExit("checkpoint has more classes than the prepared data")
            new[:n_old], new[-1] = old[:n_old], old[-1]
            sd[k] = new
    model.load_state_dict(sd)


def log(rank, *a):
    if rank == 0:
        print(*a, flush=True)


def lr_at(step, total, warm, peak):
    if step < warm:
        return peak * (step + 1) / warm
    return peak * (0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm))))


@torch.no_grad()
def evaluate(model, loader, dev, n_signs, world):
    model.eval()
    s = torch.zeros(9, dtype=torch.float64, device=dev)  # loss, n, pos, pos_ok, pos_sign_ok, top3, top5, neg, neg_ok
    for x, y in loader:
        x, y = x.to(dev), y.to(dev)
        lo = model(x)
        s[0] += F.cross_entropy(lo, y, reduction="sum"); s[1] += len(y)
        pos, neg = y < n_signs, y >= n_signs
        s[2] += pos.sum(); s[3] += (lo.argmax(1)[pos] == y[pos]).sum()
        ls = lo[pos][:, :n_signs]
        s[4] += (ls.argmax(1) == y[pos]).sum()
        top = ls.topk(min(5, n_signs), 1).indices
        s[5] += (top[:, :min(3, n_signs)] == y[pos][:, None]).any(1).sum(); s[6] += (top == y[pos][:, None]).any(1).sum()
        s[7] += neg.sum(); s[8] += (lo.argmax(1)[neg] == n_signs).sum()
    if world > 1:
        dist.all_reduce(s)
    s = s.cpu().numpy()
    return dict(loss=s[0] / max(s[1], 1), pos_acc=s[3] / max(s[2], 1), sign_acc=s[4] / max(s[2], 1),
                top3=s[5] / max(s[2], 1), top5=s[6] / max(s[2], 1), neg_acc=s[8] / max(s[7], 1))


def main():
    args = parse()
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    local = int(os.environ.get("LOCAL_RANK", 0))
    use_cuda = torch.cuda.is_available()
    dev = torch.device("cuda", local) if use_cuda else torch.device("cpu")
    if use_cuda:
        torch.cuda.set_device(dev)
    if world > 1:
        dist.init_process_group("nccl" if use_cuda else "gloo")
    cpus = int(os.environ.get("SLURM_CPUS_ON_NODE", os.cpu_count() or 1))
    local_world = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
    torch.set_num_threads(args.threads or max(1, cpus // local_world - args.workers))
    np.random.seed(args.seed + rank); torch.manual_seed(args.seed)
    os.makedirs(args.out, exist_ok=True)

    store = ClipStore(args.data)
    N = store.n_classes
    if store.prep.get("feature_version") != A.FEATURE_VERSION:
        raise SystemExit("prepared data was built with a different feature version; rerun prepare_data.py")
    # ---- signer-independent split -------------------------------------------------------------
    kag = store.meta[store.meta.source == "kaggle"]
    signers = sorted(kag.signer.unique())
    if args.val_ids:
        val_s = set(args.val_ids.split(","))
    else:
        rs = np.random.default_rng(args.seed)
        n_val = min(args.val_signers, max(0, len(signers) - 1))
        val_s = {str(v) for v in rs.choice(signers, n_val, replace=False)} if n_val else set()
    is_val = store.meta.signer.isin(val_s).to_numpy()
    tr_idx, va_idx = np.flatnonzero(~is_val), np.flatnonzero(is_val)
    extra = np.flatnonzero((store.meta.source == "extra").to_numpy())
    tr_idx = np.concatenate([tr_idx] + [extra] * max(0, args.extra_repeat - 1)) if len(extra) else tr_idx
    log(rank, f"[data] {store.n_clips} clips | {N} signs | train {len(tr_idx)} ({len(extra)} extra) | val {len(va_idx)} "
              f"from signers {sorted(val_s)} | world {world} on {dev}")
    if not len(va_idx):
        log(rank, "[data] WARNING: no validation clips; metrics are on training signers")
        va_idx = tr_idx[:2048]

    train_ds = WindowDataset(store, tr_idx, WindowSampler(store, args.window, pool=tr_idx), int(args.neg_ratio * len(tr_idx)),
                             seed=args.seed, train=True)
    val_ds = WindowDataset(store, va_idx, WindowSampler(store, args.window, pool=va_idx), int(args.neg_ratio * len(va_idx)),
                           seed=1234, train=False)
    per_rank = max(1, args.batch_size // world)
    winit = lambda _: torch.set_num_threads(1)  # noqa: E731
    tr_samp = DistributedSampler(train_ds, num_replicas=world, rank=rank, shuffle=True, seed=args.seed, drop_last=True)
    va_samp = DistributedSampler(val_ds, num_replicas=world, rank=rank, shuffle=False)
    tr_dl = DataLoader(train_ds, per_rank, sampler=tr_samp, num_workers=args.workers, worker_init_fn=winit, drop_last=True,
                       pin_memory=use_cuda, prefetch_factor=4 if args.workers else None)
    va_dl = DataLoader(val_ds, 256, sampler=va_samp, num_workers=args.workers, worker_init_fn=winit)

    arch = dict(in_dim=A.N_FEATURES, n_out=N + 1, T=args.window, d=args.d_model, layers=args.layers, heads=args.heads,
                drop=args.dropout)
    model = SignEncoder(**arch).to(dev)
    log(rank, f"[model] {n_params(model) / 1e6:.2f}M parameters, window {args.window} frames @ {store.prep['fps']} fps")
    decay, no_decay = [], []
    for n_, p in model.named_parameters():
        (no_decay if p.ndim < 2 or n_ == "pos" else decay).append(p)
    opt = torch.optim.AdamW([dict(params=decay, weight_decay=args.wd), dict(params=no_decay, weight_decay=0.0)],
                            lr=args.lr, betas=(0.9, 0.98))
    ema = Ema(model, args.ema)
    start_epoch, step, best = 0, 0, -1.0
    resume = os.path.join(args.out, "last.pt") if args.resume == "auto" else args.resume
    if resume and os.path.exists(resume):
        ck = torch.load(resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"]); ema.model.load_state_dict(ck["ema"]); opt.load_state_dict(ck["opt"])
        start_epoch, step, best = ck["epoch"] + 1, ck["step"], ck["best"]
        ema.n = ck.get("ema_n", ema.n)
        log(rank, f"[resume] from {resume} at epoch {start_epoch}")
    elif args.init_from:
        ck = torch.load(args.init_from, map_location="cpu", weights_only=False)
        sd = ck.get("ema") or ck["model"]
        load_weights(model, sd); ema.model.load_state_dict(copy.deepcopy(model.state_dict()))
        log(rank, f"[init] weights from {args.init_from}")
    dmodel = DDP(model, device_ids=[local] if use_cuda else None) if world > 1 else model

    steps_per_epoch = args.max_steps_per_epoch or len(tr_dl)
    total, warm = args.epochs * steps_per_epoch, int(args.warmup_epochs * steps_per_epoch)
    csv_path = os.path.join(args.out, "metrics.csv")
    t_start = time.time()
    epoch, it = start_epoch - 1, 0
    for epoch in range(start_epoch, args.epochs):
        train_ds.set_epoch(epoch); tr_samp.set_epoch(epoch)
        dmodel.train()
        t0, seen, run_loss = time.time(), 0, 0.0
        for it, (x, y) in enumerate(tr_dl):
            if args.max_steps_per_epoch and it >= args.max_steps_per_epoch:
                break
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            for g in opt.param_groups:
                g["lr"] = lr_at(step, total, warm, args.lr)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_cuda and torch.cuda.is_bf16_supported()):
                loss = F.cross_entropy(dmodel(x), y, label_smoothing=args.label_smoothing)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); ema.update(model); step += 1
            seen += len(y) * world; run_loss += loss.item()
            if (it + 1) % args.log_every == 0:
                log(rank, f"  ep {epoch + 1} it {it + 1}/{steps_per_epoch} loss {run_loss / (it + 1):.3f} "
                          f"lr {opt.param_groups[0]['lr']:.2e} {seen / (time.time() - t0):.0f} samples/s")
        m = evaluate(ema.model, va_dl, dev, N, world)
        score = 0.8 * m["sign_acc"] + 0.2 * m["neg_acc"]
        log(rank, f"[epoch {epoch + 1}/{args.epochs}] train_loss {run_loss / max(1, it + 1):.3f} | val loss {m['loss']:.3f} "
                  f"sign_acc {m['sign_acc']:.4f} top3 {m['top3']:.4f} top5 {m['top5']:.4f} win_acc {m['pos_acc']:.4f} "
                  f"no-sign_acc {m['neg_acc']:.4f} | {time.time() - t0:.0f}s")
        stop = torch.tensor([float(args.time_budget_min and (time.time() - t_start) / 60 > args.time_budget_min)], device=dev)
        if world > 1:
            dist.all_reduce(stop)
        if rank == 0:
            row = dict(epoch=epoch + 1, train_loss=run_loss / max(1, it + 1), **{k: float(v) for k, v in m.items()}, score=float(score))
            new = not os.path.exists(csv_path)
            with open(csv_path, "a", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(row)); new and w.writeheader(); w.writerow(row)
            if score > best:
                best = float(score)
                torch.save(dict(ema=ema.model.state_dict(), arch=arch, classes=store.classes, window=args.window,
                                fps=store.prep["fps"], metrics=row, val_signers=sorted(val_s)), os.path.join(args.out, "best.pt"))
            torch.save(dict(model=model.state_dict(), ema=ema.model.state_dict(), opt=opt.state_dict(), epoch=epoch, step=step,
                            best=best, ema_n=ema.n, args=vars(args)), os.path.join(args.out, "last.pt.tmp"))
            os.replace(os.path.join(args.out, "last.pt.tmp"), os.path.join(args.out, "last.pt"))
        if stop.item() > 0:
            log(rank, f"[budget] {args.time_budget_min} min reached - stopped after epoch {epoch + 1}; rerun with --resume auto")
            break
    if rank == 0:
        with open(os.path.join(args.out, "train_summary.json"), "w") as f:
            json.dump(dict(best_score=best, epochs_done=epoch + 1, args=vars(args), val_signers=sorted(val_s),
                           minutes=(time.time() - t_start) / 60), f, indent=1)
        log(rank, f"[done] best score {best:.4f} -> {os.path.join(args.out, 'best.pt')}")
    if world > 1:
        dist.barrier(); dist.destroy_process_group()


if __name__ == "__main__":
    main()