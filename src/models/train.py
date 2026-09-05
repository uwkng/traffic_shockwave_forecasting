"""
Stage 10 - train one ablation rung on one split, for N seeds.

    python -m src.models.train --rung 0_speed --split single
    python -m src.models.train --rung 6_all   --split fold00 --seeds 3
    python -m src.models.train --rung 6_all   --all-folds

Writes, per seed, into `checkpoints/`:
    <rung>__<split>__seed<K>.pt        best weights by validation MAE
    <rung>__<split>__seed<K>.json      per-epoch history + test metrics

and the test predictions into `data/processed/predictions/` in the contract's
format, so `src/eval` treats every model identically. Nothing here reads
data/processed directly - `loader.build_loaders` owns that.

Two conventions worth knowing before reading the numbers:

  * The loss is L1 on DE-NORMALISED speed. Y comes out of the loader in mph, so
    the loss is already in the unit the paper reports and needs no rescaling.
    Predictions are produced in normalised space and converted with
    `scaler.to_mph` before the loss, which is why `to_mph` appears in the
    training loop and not only at evaluation.
  * Metrics mask the imputed values. 521 speed readings were interpolated so the
    model would see no NaN; letting them into a metric would score the model
    against our own interpolation.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np
import torch
import torch.nn as nn
import yaml

from src import contract
from src.models.loader import build_loaders
from src.models.stgcn import build_model

CKPT = pathlib.Path("checkpoints")
PRED = pathlib.Path("data/processed/predictions")


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pick_device(requested: str | None = None) -> str:
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def metrics(pred: torch.Tensor, true: torch.Tensor,
            mask: torch.Tensor | None = None) -> dict:
    """MAE / RMSE / masked MAPE on de-normalised speed, at every eval horizon.

    `mask` is True where the observation is real. MAPE additionally excludes
    speeds below 1 mph, where a percentage error is meaningless.
    """
    out = {}
    for name, step in [("all", None)] + list(contract.EVAL_HORIZON_STEPS.items()):
        p, t = (pred, true) if step is None else (pred[:, step - 1], true[:, step - 1])
        m = (mask if step is None else mask[:, step - 1]) if mask is not None else None
        if m is not None:
            p, t = p[m], t[m]
        d = p - t
        ad = d.abs()
        nz = t.abs() > 1.0
        out[name] = {
            "mae": ad.mean().item(),
            "rmse": torch.sqrt((d ** 2).mean()).item(),
            "mape": (ad[nz] / t[nz].abs()).mean().item() * 100 if nz.any() else float("nan"),
            "n": int(t.numel()),
        }
    return out


@torch.no_grad()
def evaluate(model, loader, scaler, device, eval_mask=None, timesteps=None,
             return_pred=False):
    model.eval()
    preds, trues = [], []
    for x, y in loader:
        preds.append(model(x.to(device)).cpu())
        trues.append(y)
    pred = scaler.to_mph(torch.cat(preds))
    true = torch.cat(trues)

    mask = None
    if eval_mask is not None and timesteps is not None:
        # eval_mask is [T, N] over timesteps; a sample's targets are the HORIZON
        # steps after its input window, so index it the same way.
        H = contract.HORIZON
        idx = timesteps[:, None] + np.arange(H)[None, :]
        mask = torch.from_numpy(eval_mask[idx])            # [n_samples, H, N]
    m = metrics(pred, true, mask)
    return (m, pred, true) if return_pred else (m, None, None)


# Congestion-weighted L1. OFF by default: rung 0 must stay comparable to the
# published PEMS-BAY numbers, which are all plain L1/L2 on speed.
#
# Why it exists. 90.4% of observations are above 55 mph and they carry 63.7% of
# the total absolute error, at MAE 2.18; below 45 mph is 6.0% of observations
# carrying 26.1%, at MAE 12-14. A plain mean absolute error therefore spends
# most of its gradient on turning 65 mph into 63 rather than on the breakdowns
# this project is named after. Note also that the original STGCN used L2
# (`tf.nn.l2_loss`), which leans harder on large errors than our L1 does - the
# switch to L1 moved us AWAY from shockwaves and this moves back.
#
# w = clamp((60 - y)/20, min=0) + 1, i.e. 1.0 at 60+ mph, 1.5 at 50, 2.5 at 30,
# 3.0 at 20. Same weighting for every rung, so cross-rung MAEs stay comparable.
# It changes what the optimiser cares about, never how the metric is computed:
# every reported number is still unweighted MAE/RMSE/MAPE on de-normalised mph.
WEIGHTED_LOSS_PIVOT_MPH = 60.0
WEIGHTED_LOSS_SCALE_MPH = 20.0


def congestion_weighted_l1(pred_mph, true_mph):
    w = torch.clamp((WEIGHTED_LOSS_PIVOT_MPH - true_mph) / WEIGHTED_LOSS_SCALE_MPH,
                    min=0.0) + 1.0
    return (w * (pred_mph - true_mph).abs()).mean()


def train_one(rung, split, seed, cfg, device, epochs=None, quiet=False,
              loss_name="l1"):
    # NOT called `loss`: the training loop below binds `loss` to a Tensor every
    # step, and a shadowed name here silently put that Tensor into the JSON.
    set_seed(seed)
    loaders, adj, scaler = build_loaders(rung=rung, split=split)
    model = build_model(adj, loaders["c_in"], cfg,
                        contract.INPUT_WINDOW, contract.HORIZON, device)
    n_par = sum(p.numel() for p in model.parameters())

    t_cfg = cfg["training"]
    epochs = epochs or t_cfg["epochs"]
    opt = torch.optim.Adam(model.parameters(), lr=t_cfg["lr"],
                           weight_decay=t_cfg["weight_decay"])
    sched = torch.optim.lr_scheduler.StepLR(
        opt, step_size=t_cfg["scheduler"]["step_size"],
        gamma=t_cfg["scheduler"]["gamma"])
    assert loss_name in ("l1", "weighted"), f"unknown --loss {loss_name!r}"
    crit = nn.L1Loss() if loss_name == "l1" else congestion_weighted_l1

    tag = (f"{rung}__{split}__seed{seed}"
           + ("" if loss_name == "l1" else "__weighted"))
    CKPT.mkdir(exist_ok=True)
    ckpt = CKPT / f"{tag}.pt"
    if not quiet:
        print(f"  {tag}: c_in={loaders['c_in']}, {n_par:,} params, "
              f"{loaders['n_samples']['train']:,} train samples, device={device}")

    best, patience, history = float("inf"), 0, []
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        total = n = 0
        for x, y in loaders["train"]:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(scaler.to_mph(model(x)), y)      # both in mph
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()
            total += loss.item(); n += 1
        sched.step()
        val, _, _ = evaluate(model, loaders["val"], scaler, device)
        val_mae = val["all"]["mae"]
        history.append({"epoch": epoch, "train_loss": total / n,
                        "val_mae": val_mae, "seconds": time.time() - t0})
        if not quiet and (epoch % 10 == 0 or epoch == 1):
            print(f"    epoch {epoch:3d} | train L1 {total / n:6.4f} mph | "
                  f"val MAE {val_mae:6.4f} mph | {time.time() - t0:5.1f}s")
        if val_mae < best:
            best, patience = val_mae, 0
            torch.save(model.state_dict(), ckpt)
        else:
            patience += 1
            if patience >= t_cfg["patience"]:
                if not quiet:
                    print(f"    early stop at epoch {epoch}")
                break

    model.load_state_dict(torch.load(ckpt, weights_only=True, map_location=device))
    test, pred, true = evaluate(model, loaders["test"], scaler, device,
                                loaders["eval_mask"],
                                loaders["datasets"]["test"].timesteps(),
                                return_pred=True)

    PRED.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        PRED / f"{tag}.npz",
        pred=pred.numpy().astype(contract.PREDICTION_DTYPE),
        true=true.numpy().astype(contract.PREDICTION_DTYPE),
        timesteps=loaders["datasets"]["test"].timesteps(),
        sample_ids=loaders["datasets"]["test"].ids)

    report = {"rung": rung, "split": split, "seed": seed, "loss": loss_name,
              "c_in": loaders["c_in"],
              "channels": loaders["channels"], "params": n_par,
              "future_covariates": loaders["future_covariates"],
              "epochs_run": len(history), "best_val_mae": best,
              "n_samples": loaders["n_samples"], "history": history,
              "test": test}
    (CKPT / f"{tag}.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rung", default="6_all", choices=list(contract.ABLATION_RUNGS))
    ap.add_argument("--split", default="single")
    ap.add_argument("--all-folds", action="store_true",
                    help="run fold00..fold05 instead of --split")
    ap.add_argument("--seeds", type=int, default=None, help="default: contract.N_SEEDS")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--loss", choices=["l1", "weighted"], default="l1",
                    help="weighted = congestion-weighted L1; see the module "
                         "docstring. Off by default so rung 0 stays comparable "
                         "to the published baselines.")
    ap.add_argument("--device", default=None, help="cuda | mps | cpu (auto by default)")
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()

    cfg = yaml.safe_load(pathlib.Path(args.config).read_text(encoding="utf-8"))
    device = pick_device(args.device)
    n_seeds = args.seeds or contract.N_SEEDS
    splits = ([f"fold{i:02d}" for i in range(cfg["split"]["rolling"]["n_folds"])]
              if args.all_folds else [args.split])

    if args.rung == "3_weather" and "single" in splits:
        print("  WARNING: the `single` test block contains ZERO adverse-weather\n"
              "  episodes, so a weather-conditioned model cannot differ from a\n"
              "  traffic-only one on it. Use --all-folds for any weather result.")

    print(f"=== stage 10: train {args.rung} on {', '.join(splits)} ===\n")
    results = []
    for split in splits:
        for i in range(n_seeds):
            results.append(train_one(args.rung, split, contract.SEED + i,
                                     cfg, device, args.epochs,
                                     loss_name=args.loss))

    print(f"\n=== {args.rung}: test MAE / RMSE / MAPE, "
          f"mean +- sd over {n_seeds} seeds ===")
    print(f"  {'split':9s}" + "".join(f"{h:>26s}" for h in contract.EVAL_HORIZON_STEPS))
    for split in splits:
        rs = [r for r in results if r["split"] == split]
        row = f"  {split:9s}"
        for h in contract.EVAL_HORIZON_STEPS:
            v = np.array([[r["test"][h][k] for k in ("mae", "rmse", "mape")] for r in rs])
            row += (f"{v[:, 0].mean():7.3f}+-{v[:, 0].std():.3f}"
                    f"{v[:, 1].mean():7.3f}{v[:, 2].mean():6.2f}%")
        print(row)
    print(f"\n  checkpoints -> {CKPT}/    predictions -> {PRED}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
