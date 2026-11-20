"""Trening jednego wariantu: python train.py --run r18_full [--epochs N] [--subset N]."""
import argparse
import json
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import wandb

import config as C
from data import get_loaders, prepare
from metrics import TopKCounter, macro_f1
from models import build_model, count_params
from utils import get_device, set_seed, sync


@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    counter = TopKCounter(ks=(1, 3, 5))
    loss_sum, preds, targets, logits_all = 0.0, [], [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        loss_sum += criterion(out, y).item() * y.size(0)
        counter.update(out, y)
        preds.append(out.argmax(1).cpu())
        targets.append(y.cpu())
        logits_all.append(out.float().cpu())
    acc = counter.compute()
    y_true, y_pred = torch.cat(targets).numpy(), torch.cat(preds).numpy()
    return dict(loss=loss_sum / counter.total, acc=acc[1], top3=acc[3], top5=acc[5],
                macro_f1=macro_f1(y_true, y_pred)), torch.cat(logits_all), y_true


def set_train_mode(model, frozen_modules):
    model.train()
    for m in frozen_modules:  # zamrożone części: BatchNorm w trybie eval
        m.eval()


@torch.no_grad()
def inference_latency(model, device, n=20, bs=1):
    model.eval()
    x = torch.randn(bs, 3, C.IMG_SIZE, C.IMG_SIZE, device=device)
    for _ in range(5):
        model(x)
    sync(device)
    t = time.perf_counter()
    for _ in range(n):
        model(x)
    sync(device)
    return (time.perf_counter() - t) / n * 1000  # ms / obraz


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, choices=list(C.VARIANTS))
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--subset", type=int, help="tylko N przykładów na split (smoke test)")
    args = ap.parse_args()

    cfg = dict(C.VARIANTS[args.run])
    if args.epochs:
        cfg["epochs"] = args.epochs
    set_seed(C.SEED)
    device = get_device()
    df = prepare()
    loaders = get_loaders(df, subset=args.subset)

    model, head_params, body_params, frozen = build_model(cfg["arch"], cfg["finetune"])
    model.to(device)
    total_p, trainable_p = count_params(model)
    groups = [{"params": head_params, "lr": cfg["lr_head"]}]
    if body_params:
        groups.append({"params": body_params, "lr": cfg["lr_body"]})
    opt = torch.optim.AdamW(groups, weight_decay=1e-4)
    steps = cfg["epochs"] * len(loaders["train"])
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[g["lr"] for g in groups], total_steps=steps, pct_start=0.1)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    name = args.run + (f"_smoke{args.subset}" if args.subset else "")
    mode = os.environ.get("WANDB_MODE") or ("online" if wandb.api.api_key else "offline")
    run = wandb.init(project=C.WANDB_PROJECT, name=name, mode=mode,
                     config=dict(cfg, seed=C.SEED, batch_size=C.BATCH_SIZE, device=str(device),
                                 total_params=total_p, trainable_params=trainable_p))
    print(f"[{name}] device={device} params={total_p:,} trainable={trainable_p:,}")

    C.CKPT_DIR.mkdir(exist_ok=True)
    C.RESULTS_DIR.mkdir(exist_ok=True)
    ckpt = C.CKPT_DIR / f"{name}.pt"
    history, best_acc, train_time = [], -1.0, 0.0
    for epoch in range(1, cfg["epochs"] + 1):
        set_train_mode(model, frozen)
        counter, loss_sum = TopKCounter(ks=(1,)), 0.0
        sync(device)
        t0 = time.perf_counter()
        for x, y in loaders["train"]:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = criterion(out, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            loss_sum += loss.item() * y.size(0)
            counter.update(out, y)
        sync(device)
        ep_time = time.perf_counter() - t0
        train_time += ep_time
        val, _, _ = evaluate(model, loaders["val"], device, criterion)
        row = dict(epoch=epoch, train_loss=loss_sum / counter.total, train_acc=counter.compute()[1],
                   val_loss=val["loss"], val_acc=val["acc"], val_top3=val["top3"],
                   val_macro_f1=val["macro_f1"], epoch_time_s=ep_time,
                   cum_train_time_s=train_time, lr_head=opt.param_groups[0]["lr"])
        history.append(row)
        wandb.log(row, step=epoch)
        print(f"  ep {epoch:2d} | train loss {row['train_loss']:.3f} acc {row['train_acc']:.3f} | "
              f"val acc {val['acc']:.3f} f1 {val['macro_f1']:.3f} top3 {val['top3']:.3f} | {ep_time:.0f}s",
              flush=True)
        if val["acc"] > best_acc:
            best_acc = val["acc"]
            torch.save(model.state_dict(), ckpt)

    model.load_state_dict(torch.load(ckpt, map_location=device))
    test, logits, y_true = evaluate(model, loaders["test"], device, criterion)
    latency = inference_latency(model, device)
    summary = dict(run=name, **{k: cfg[k] for k in ("arch", "finetune", "epochs", "lr_head", "lr_body")},
                   total_params=total_p, trainable_params=trainable_p,
                   train_time_s=train_time, mean_epoch_time_s=train_time / cfg["epochs"],
                   best_val_acc=best_acc, latency_ms_bs1=latency, device=str(device),
                   **{f"test_{k}": v for k, v in test.items()})
    print(json.dumps(summary, indent=1))
    wandb.summary.update(summary)
    run.finish()

    if not args.subset:
        pd.DataFrame(history).to_csv(C.RESULTS_DIR / f"{name}_history.csv", index=False)
        (C.RESULTS_DIR / f"{name}.json").write_text(json.dumps(summary, indent=1))
        top5 = torch.topk(logits, 5, dim=1).indices.numpy()
        test_df = df[df.split == "test"].reset_index(drop=True)
        pd.DataFrame(dict(file=test_df.file, label=y_true, pred=top5[:, 0],
                          top5=[" ".join(map(str, r)) for r in top5],
                          conf=torch.softmax(logits, 1).max(1).values.numpy().round(4))
                     ).to_csv(C.RESULTS_DIR / f"{name}_test_preds.csv", index=False)


if __name__ == "__main__":
    main()
