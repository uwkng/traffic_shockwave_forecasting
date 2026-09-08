"""Evaluation metrics computed on de-normalized predictions (real mph).

All functions take numpy arrays, not torch tensors, because evaluation runs
after inference and does not need a GPU or autograd.
"""

from __future__ import annotations

import numpy as np


def mae(pred: np.ndarray, true: np.ndarray, mask: np.ndarray | None = None) -> float:
    diff = np.abs(pred - true)
    if mask is not None:
        diff = diff[mask]
    return float(np.mean(diff))


def rmse(pred: np.ndarray, true: np.ndarray, mask: np.ndarray | None = None) -> float:
    diff = (pred - true) ** 2
    if mask is not None:
        diff = diff[mask]
    return float(np.sqrt(np.mean(diff)))


def mape(pred: np.ndarray, true: np.ndarray, mask: np.ndarray | None = None) -> float:
    valid = np.abs(true) > 1e-5
    if mask is not None:
        valid = valid & mask
    return float(np.mean(np.abs(pred[valid] - true[valid]) / np.abs(true[valid])) * 100)


def all_metrics(pred: np.ndarray, true: np.ndarray,
                mask: np.ndarray | None = None) -> dict[str, float]:
    return {
        "mae": mae(pred, true, mask),
        "rmse": rmse(pred, true, mask),
        "mape": mape(pred, true, mask),
    }
