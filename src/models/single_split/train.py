"""Train STGCN on any ablation rung and split.

    python -m src.models.single_split.train --rung 0_speed --split single
    python -m src.models.single_split.train --rung 6_all  --split fold00 --config configs/default.yaml
"""

from __future__ import annotations

import argparse
import json
import pathlib
import time

import numpy as np
import torch
import torch.nn as nn

from src import contract
from src.models.loader import build_loaders, load_config
from src.models.single_split.stgcn import STGCN, scaled_laplacian, cheb_polynomials


# ---------------------------------------------------------------------------
# Loss / metrics (from notebooks/train_stgcn.ipynb)
# ---------------------------------------------------------------------------

def _build_mask(true: torch.Tensor, null_val: float = 0.0) -> torch.Tensor:
    mask = (true != null_val).float()
    mask = mask / torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    return mask


def masked_mae_loss(pred: torch.Tensor, true: torch.Tensor,
                    null_val: float = 0.0) -> torch.Tensor:
    mask = _build_mask(true, null_val)
    loss = torch.abs(pred - true) * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def compute_metrics(pred: torch.Tensor, true: torch.Tensor,
                    null_val: float = 0.0) -> dict[str, float]:
    mask = _build_mask(true, null_val)
    diff = pred - true
    abs_diff = torch.abs(diff)

    def _masked(raw):
        v = raw * mask
        v = torch.where(torch.isnan(v), torch.zeros_like(v), v)
        return torch.mean(v)

    return {
        "mae": _masked(abs_diff).item(),
        "rmse": torch.sqrt(_masked(diff ** 2)).item(),
        "mape": _masked(abs_diff / torch.abs(true)).item() * 100,
    }


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_epoch(model: nn.Module, loader, optimizer, scaler, device: str) -> float:
    model.train()
    total_loss, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        pred = model(x)
        pred_mph = scaler.to_mph(pred)
        loss = masked_mae_loss(pred_mph, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        total_loss += loss.item()
        n += 1
    return total_loss / n


@torch.no_grad()
def evaluate(model: nn.Module, loader, scaler, device: str) -> dict:
    model.eval()
    preds, trues = [], []
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        preds.append(model(x))
        trues.append(y)
    pred = scaler.to_mph(torch.cat(preds))
    true = torch.cat(trues)

    horizon_metrics = [
        compute_metrics(pred[:, i, :], true[:, i, :])
        for i in range(pred.shape[1])
    ]
    results = {
        "all": {
            "mae": float(np.mean([m["mae"] for m in horizon_metrics])),
            "rmse": float(np.mean([m["rmse"] for m in horizon_metrics])),
            "mape": float(np.mean([m["mape"] for m in horizon_metrics])),
        },
    }
    for name, step in contract.EVAL_HORIZON_STEPS.items():
        if step <= len(horizon_metrics):
            results[name] = horizon_metrics[step - 1]
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(rung: str, split: str, seed: int, config: str,
        device: str | None = None, future_covariates: bool = False):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = load_config(config)
    tcfg = cfg["training"]
    mcfg = cfg["model"]["stgcn"]

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    fc_tag = "_fc" if future_covariates else ""
    print(f"rung={rung}  split={split}  seed={seed}  device={device}"
          f"  future_covariates={future_covariates}")

    loaders, adj, scaler = build_loaders(
        rung=rung, split=split, batch_size=tcfg["batch_size"], config=config,
        future_covariates=future_covariates,
    )
    for p in ("train", "val", "test"):
        print(f"  {p}: {loaders['n_samples'][p]:,} samples")
    print(f"  c_in={loaders['c_in']}  channels={loaders['channels']}")
    print(f"  scaler: {scaler}")

    L = scaled_laplacian(adj)
    cheb_polys = cheb_polynomials(L, mcfg["Ks"])

    model = STGCN(
        cheb_polys=cheb_polys,
        c_in=loaders["c_in"],
        input_window=contract.INPUT_WINDOW,
        horizon=contract.HORIZON,
        Ks=mcfg["Ks"],
        Kt=mcfg["Kt"],
        blocks=mcfg["blocks"],
        dropout=mcfg["dropout"],
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  parameters: {n_params:,}")

    optimizer = torch.optim.Adam(
        model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"],
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=tcfg["scheduler"]["step_size"],
        gamma=tcfg["scheduler"]["gamma"],
    )

    ckpt_dir = pathlib.Path(tcfg["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_name = f"stgcn_{rung}_{split}{fc_tag}_seed{seed}.pt"
    ckpt_path = ckpt_dir / ckpt_name

    best_val_mae = float("inf")
    patience_counter = 0

    for epoch in range(1, tcfg["epochs"] + 1):
        t0 = time.time()
        train_loss = train_epoch(model, loaders["train"], optimizer, scaler, device)
        val_results = evaluate(model, loaders["val"], scaler, device)
        scheduler.step()

        val_mae = val_results["all"]["mae"]
        elapsed = time.time() - t0

        if epoch % 10 == 0 or epoch == 1:
            print(f"  epoch {epoch:3d} | loss {train_loss:.4f} | "
                  f"val MAE {val_mae:.4f} | {elapsed:.1f}s")

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            patience_counter = 0
            torch.save(model.state_dict(), ckpt_path)
        else:
            patience_counter += 1
            if patience_counter >= tcfg["patience"]:
                print(f"  early stopping at epoch {epoch}")
                break

    model.load_state_dict(torch.load(ckpt_path, weights_only=True, map_location=device))
    test_results = evaluate(model, loaders["test"], scaler, device)

    print(f"\n  test results ({ckpt_name}):")
    for horizon, metrics in test_results.items():
        print(f"    {horizon:>5s}: MAE={metrics['mae']:.4f}  "
              f"RMSE={metrics['rmse']:.4f}  MAPE={metrics['mape']:.2f}%")

    meta = {
        "rung": rung, "split": split, "seed": seed,
        "future_covariates": future_covariates,
        "uses_forecast_weather": loaders.get("uses_forecast_weather", False),
        "c_in": loaders["c_in"], "channels": loaders["channels"],
        "n_params": n_params, "checkpoint": str(ckpt_path),
        "best_val_mae": best_val_mae, "test": test_results,
    }
    meta_path = ckpt_dir / f"stgcn_{rung}_{split}{fc_tag}_seed{seed}_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"  saved: {ckpt_path}")
    print(f"  meta:  {meta_path}")
    return meta


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rung", default="0_speed",
                    choices=list(contract.ABLATION_RUNGS))
    ap.add_argument("--split", default="single")
    ap.add_argument("--seed", type=int, default=contract.SEED)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--device", default=None)
    ap.add_argument("--future-covariates", action="store_true",
                    help="Feed forecast weather for the target window")
    args = ap.parse_args()
    run(args.rung, args.split, args.seed, args.config, args.device,
        args.future_covariates)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
