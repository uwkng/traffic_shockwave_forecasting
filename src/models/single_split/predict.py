"""Run inference and save de-normalized predictions.

    python -m src.models.predict --rung 0_speed --split single --seed 42

Loads the checkpoint written by train.py, runs the test set, and saves
predictions as [n_test, HORIZON, N_NODES] float32 in mph.
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np
import torch

from src import contract
from src.models.loader import build_loaders, load_config
from src.models.stgcn import STGCN, scaled_laplacian, cheb_polynomials


def predict(rung: str, split: str, seed: int, config: str,
            device: str | None = None, future_covariates: bool = False) -> pathlib.Path:
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    cfg = load_config(config)
    mcfg = cfg["model"]["stgcn"]
    tcfg = cfg["training"]
    fc_tag = "_fc" if future_covariates else ""

    loaders, adj, scaler = build_loaders(
        rung=rung, split=split, batch_size=tcfg["batch_size"], config=config,
        future_covariates=future_covariates,
    )

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

    ckpt_path = (pathlib.Path(tcfg["checkpoint_dir"])
                 / f"stgcn_{rung}_{split}{fc_tag}_seed{seed}.pt")
    model.load_state_dict(torch.load(ckpt_path, weights_only=True, map_location=device))
    model.eval()

    preds = []
    with torch.no_grad():
        for x, _ in loaders["test"]:
            pred = model(x.to(device))
            pred_mph = scaler.to_mph(pred)
            preds.append(pred_mph.cpu().numpy())

    predictions = np.concatenate(preds, axis=0).astype(contract.PREDICTION_DTYPE)
    expected = (loaders["n_samples"]["test"], contract.HORIZON, contract.N_NODES)
    assert predictions.shape == expected, f"{predictions.shape} != {expected}"

    out_dir = pathlib.Path(tcfg["checkpoint_dir"])
    out_path = out_dir / f"pred_{rung}_{split}{fc_tag}_seed{seed}.npy"
    np.save(out_path, predictions)
    print(f"saved {out_path}  shape={predictions.shape}  "
          f"range=[{predictions.min():.1f}, {predictions.max():.1f}] mph")
    return out_path


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
    predict(args.rung, args.split, args.seed, args.config, args.device,
            args.future_covariates)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
