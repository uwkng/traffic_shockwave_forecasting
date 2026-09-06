"""
The interface between the data pipeline (stages 1-7) and the models (9-10).

    from src.models.loader import build_loaders

    loaders, adj, scaler = build_loaders(rung="6_all", split="fold00")
    model = STGCN(c_in=loaders["c_in"], ...)

Everything a model needs to consume `data/processed/` correctly is here, so it
is decided once rather than re-derived in each training script. There are seven
such decisions and none of them fails loudly when you get it wrong - they just
move the MAE:

  1. WHICH COLUMNS. Channel order is `contract.CHANNELS`; a rung is a subset.
     `contract.rung_channels()` returns the indices so nobody counts by hand.
  2. WHICH SCALER. Each split has its own, fit on THAT split's train span. Using
     fold00's scaler on fold03 normalises with statistics from the future.
  3. PER CHANNEL. One global mean/std over the tensor averages mph together with
     a 0/1 holiday flag, kilometres and millimetres.
  4. PASSTHROUGH. time_of_day / day_of_week / is_holiday are stored as (0, 1) in
     scalers.json, i.e. identity. Z-scoring an ordinal or a flag is meaningless
     and makes "not a holiday" a non-zero value.
  5. THE TARGET IS RAW. Y is speed in mph, un-normalised. Train on it directly
     with an L1 loss, or normalise it yourself - but then remember (6).
  6. DE-NORMALISE BEFORE METRICS. `scaler.to_mph()` exists for this. The
     proposal requires every metric in real mph.
  7. MASK IMPUTED VALUES. 521 speed readings were interpolated so the model sees
     no NaN. They must not enter a metric. `loaders["eval_mask"]` marks the
     positions that were actually observed.

Nothing here reshapes or copies the master tensor: it is opened with
mmap_mode="r" and windows are read per sample. Measured, that costs about 0.24 s
per epoch against ~15 s of compute, so there is no reason to materialise
[n_samples, 12, 325, C] (which would be 8.9 GB for one channel and 98 GB for
eleven).
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import yaml

from src import contract

DEFAULT_CONFIG = "configs/default.yaml"


# ---------------------------------------------------------------------------

class Scaler:
    """Per-channel Z-score for one split, plus the inverse for the target.

    `mean` and `std` are already restricted to the rung's columns and are in
    CHANNELS order, so `(x - mean) / std` broadcasts straight onto
    [T, N, len(cols)]. Passthrough channels carry (0, 1) and pass through
    unchanged - that is not a special case here, it is arithmetic.
    """

    def __init__(self, raw: dict, cols: list[int], future_cols: list[int] | None = None):
        self.mean = np.asarray(raw["mean"], dtype=np.float32)[cols]
        self.std = np.asarray(raw["std"], dtype=np.float32)[cols]
        self.channels = [raw["channels"][i] for i in cols]
        # The future copy of a channel is the SAME quantity at a later timestep,
        # so it takes the same mean and std. Giving it its own statistics would
        # make "20 degrees now" and "20 degrees in three hours" different numbers
        # to the model.
        self.future_channels = ([raw["channels"][i] for i in future_cols]
                                if future_cols else [])
        if future_cols:
            self.future_mean = np.asarray(raw["mean"], dtype=np.float32)[future_cols]
            self.future_std = np.asarray(raw["std"], dtype=np.float32)[future_cols]
        self.speed_mean = float(raw["speed_mean_mph"])
        self.speed_std = float(raw["speed_std_mph"])
        self.train_time_slice = tuple(raw["train_time_slice"])

    def transform(self, x: np.ndarray) -> np.ndarray:
        """[..., len(cols)] -> normalised."""
        return (x - self.mean) / self.std

    def to_mph(self, y: np.ndarray) -> np.ndarray:
        """Inverse for the TARGET channel only. Use before computing any metric.

        Works on torch tensors as well as arrays: it is a scalar affine map, so
        `pred * std + mean` needs no numpy.
        """
        return y * self.speed_std + self.speed_mean

    def __repr__(self):
        return (f"Scaler(speed {self.speed_mean:.3f} +- {self.speed_std:.3f} mph, "
                f"{len(self.channels)} channels)")


class WindowDataset:
    """Sliding windows over the master tensor, addressed by sample id.

    A SAMPLE IS NOT A TIMESTEP. Sample `t` spans 24 steps - `t .. t+11` as input
    and `t+12 .. t+23` as target, two hours in total. Consecutive samples are
    five minutes apart and therefore overlap in 23 of their 24 steps. That is
    why the splits carry an embargo: cutting a list of samples in two cuts
    through overlapping time ranges, and without a gap the last training sample
    predicts hours the first validation sample reads.

    Implements `__len__` / `__getitem__`, so `torch.utils.data.DataLoader` takes
    it directly. torch is NOT imported here - this module stays usable for the
    non-neural baselines (historical average, persistence), which need the same
    splits and the same de-normalisation but no framework.
    """

    def __init__(self, master, ids, cols, scaler, input_window, horizon,
                 target_channel, future_cols=None,
                 forecast=None, forecast_positions=None):
        self.master = master
        self.ids = np.asarray(ids, dtype=np.int64)
        self.cols = cols
        self.scaler = scaler
        self.L = input_window
        self.H = horizon
        self.target = target_channel
        # Absolute column indices of the known-future channels, and the offsets
        # into the horizon at which they are sampled. `future_cols` is None when
        # the rung has no exogenous channels (rungs 0 and 1) or the caller turned
        # the feature off, and then this is exactly the old behaviour.
        self.future_cols = future_cols
        if future_cols is not None:
            # L points spanning the WHOLE horizon, not the first L steps of it.
            # With L=12 and H=36 that is every 15 minutes across three hours.
            # Weather and a schedule do not change meaningfully inside 15 min,
            # so this loses nothing and keeps the time axis at L - appending the
            # horizon instead would make it L+H and cost about 4x in the
            # temporal convolutions.
            self.future_offsets = np.linspace(
                input_window, input_window + horizon - 1, input_window
            ).round().astype(np.int64)
        # The future weather must be a FORECAST, not the observation. Leaving
        # the observed value in the future block would hand the model the answer
        # it is being asked to anticipate; that is leakage, and it is the single
        # easiest way to make these channels look valuable. `forecast` is
        # [T, 3] of ECMWF IFS short-range predictions and `forecast_positions`
        # says which columns of the future block they replace. The INPUT window
        # keeps the observed series, because at serving time the past really is
        # observed - the seam between the two is what deployment looks like.
        self.forecast = forecast
        self.forecast_positions = forecast_positions

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        t = int(self.ids[i])
        x = np.asarray(self.master[t:t + self.L, :, self.cols], dtype=np.float32)
        x = self.scaler.transform(x)
        if self.future_cols is not None:
            f = np.asarray(self.master[t + self.future_offsets][:, :, self.future_cols],
                           dtype=np.float32)
            if self.forecast is not None:
                fc = self.forecast[t + self.future_offsets]        # [L, 3]
                for j, pos in enumerate(self.forecast_positions):
                    f[:, :, pos] = fc[:, j, None]                  # broadcast over nodes
            f = (f - self.scaler.future_mean) / self.scaler.future_std
            x = np.concatenate([x, f], axis=-1)       # [L, N, C + C_future]
        y = np.asarray(self.master[t + self.L:t + self.L + self.H, :, self.target],
                       dtype=np.float32)
        return x, y                                   # X normalised, Y in mph

    def timesteps(self) -> np.ndarray:
        """The TARGET timestep of every sample: `ids + input_window`.

        Use this to line predictions up with the evaluation windows -
        `src.eval.windows` masks are indexed by timestep, not by sample.
        """
        return self.ids + self.L


# ---------------------------------------------------------------------------

def load_config(path: str = DEFAULT_CONFIG) -> dict:
    return yaml.safe_load(pathlib.Path(path).read_text(encoding="utf-8"))


def available_splits(config: str = DEFAULT_CONFIG) -> list[str]:
    P = pathlib.Path(load_config(config)["data"]["processed_dir"])
    with np.load(P / "splits.npz") as z:
        return sorted({k.split("__")[0] for k in z.files})


def build_loaders(rung: str = "6_all", split: str = "single", *,
                  future_covariates: bool = False,
                  batch_size: int | None = None, config: str = DEFAULT_CONFIG,
                  torch_loaders: bool = True, num_workers: int = 0):
    """Return (loaders, adjacency, scaler) for one ablation rung on one split.

    rung   one of contract.ABLATION_RUNGS: "0_speed" .. "6_all".
    split  "single", or "fold00".."fold05". Both schemes live in the same
           splits.npz; this picks which set of bookmarks to use.

    future_covariates
           OFF by default, so the setup stays like-for-like with every published
           PEMS-BAY result: history in, forecast out. Turning it on appends, for
           each input step, the known-future channels sampled across the target
           window - the weather forecast and the fixture list for the hour being
           predicted, which in reality are known days ahead.

           It is off by default rather than on because it changes the TASK, not
           just the model, and no published baseline has that information. Run
           it BOTH ways and report the difference: that difference is itself the
           result - what a forecast buys over an observation - and reporting
           both answers the reviewer who would otherwise say the task was made
           easier.

           Worth measuring because the blind spot is real. Of the samples whose
           TARGET window contains adverse weather, the fraction whose INPUT
           window is entirely dry - the rain starts inside the horizon, so the
           model has no way to know - is 8.2% at 15 min, 22.4% at 60 min and
           41.9% at 3 h. Those are blind, not hard.

           Rungs 0 and 1 ignore the flag: speed and occupancy are the two
           channels nobody knows in advance, so there is no future to add.

    `loaders` is a dict:
        train / val / test   DataLoader if torch is importable and
                             torch_loaders, else the WindowDataset itself
        c_in                 pass straight to the model's first block
        channels             the channel names, in order, for logging
        eval_mask            [T, N] bool, True where speed was OBSERVED
        sensor_ids           [N] str, the canonical node order
        n_samples            {"train": ..., "val": ..., "test": ...}

    WHICH SPLIT TO USE. Both are chronological and neither leaks, but they
    answer different questions:

      "single"          70/10/20. For reproducing published PEMS-BAY numbers,
                        and for a fast first pass. Its test block contains ZERO
                        adverse-weather episodes, so a weather-conditioned model
                        CANNOT differ from a traffic-only one on it - the
                        channels are constant across the whole block. Do not
                        report a weather result on this split.

      "fold00".."05"    Expanding-window rolling. Each fold trains strictly
                        before it tests. 51 rain spells, 50 egress hours and 2
                        holidays reach the test blocks, against 0 / 17 / 1 for
                        "single". Required for anything involving weather.

    Use ONE split for the whole paper: two configurations evaluated on different
    test sets cannot be subtracted.
    """
    if rung not in contract.ABLATION_RUNGS:
        raise KeyError(f"unknown rung {rung!r}; "
                       f"choose from {list(contract.ABLATION_RUNGS)}")
    cfg = load_config(config)
    P = pathlib.Path(cfg["data"]["processed_dir"])
    need = ["master.npy", "splits.npz", "scalers.json", "eval_mask.npy",
            "adj_mx.npy", "sensor_ids.npy"]
    missing = [f for f in need if not (P / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"{missing} not in {P}. Build the dataset first:\n"
            f"    python -m src.data.build_dataset --config {config}\n"
            "  (add --acquire on a fresh clone to download the raw sources)")

    cols = contract.rung_channels(rung)
    # Positions of the known-future channels WITHIN this rung, mapped back to
    # absolute column indices. Empty for rungs 0 and 1, whose only channels are
    # speed and occupancy - nothing there is known in advance, so those two
    # rungs are identical with and without the flag.
    fcols = ([cols[j] for j in contract.future_channels(rung)]
             if future_covariates else [])
    master = np.load(P / "master.npy", mmap_mode="r")
    if master.shape[2] != contract.N_CHANNELS:
        raise ValueError(
            f"master.npy has {master.shape[2]} channels, contract declares "
            f"{contract.N_CHANNELS}. It was built against a different contract "
            "- rebuild it.")

    scalers = json.loads((P / "scalers.json").read_text(encoding="utf-8"))["splits"]
    if split not in scalers:
        raise KeyError(f"unknown split {split!r}; available: {sorted(scalers)}")
    scaler = Scaler(scalers[split], cols, fcols or None)

    L, H = contract.INPUT_WINDOW, contract.HORIZON
    with np.load(P / "splits.npz") as z:
        parts = {p: z[f"{split}__{p}"] for p in ("train", "val", "test")}

    # The scaler must not have seen anything the training samples do not cover.
    # Cheap, and it catches a mismatched processed_dir before a run wastes GPU.
    lo, hi = scaler.train_time_slice
    tr = parts["train"]
    if len(tr) and not (lo <= tr.min() and tr.max() + L + H <= hi):
        raise ValueError(
            f"scalers.json[{split!r}] was fit over timesteps [{lo}, {hi}) but the "
            f"train samples span [{tr.min()}, {tr.max() + L + H}). The scaler and "
            "the splits come from different builds - rerun stages 6 and 7.")

    # Substitute the FORECAST for the observation in the future block, for the
    # weather channels only. Refuse rather than silently fall back: a run that
    # quietly used observed weather as its "forecast" would report leakage as a
    # result, and the filename would not say so.
    forecast, fpos = None, None
    wx = ["temperature", "precipitation", "wind_speed"]
    fnames = [contract.CHANNELS[c][0] for c in fcols]
    if any(n in wx for n in fnames):
        fpath = P / "weather_forecast.npy"
        if not fpath.exists():
            raise FileNotFoundError(
                f"{fpath} is missing, but rung {rung!r} has weather among its "
                "known-future channels. Using the OBSERVED weather as the future "
                "value would be leakage, so this refuses instead of falling "
                "back. Fix:\n"
                "    python -c 'from src.data.acquire import fetch_weather_forecast as f; f()'\n"
                f"    python -m src.data.build_dataset --config {config} --force")
        forecast = np.load(fpath)
        if forecast.shape != (master.shape[0], 3):
            raise ValueError(f"weather_forecast.npy is {forecast.shape}, expected "
                             f"({master.shape[0]}, 3) - rebuild stages 2-3.")
        fpos = [fnames.index(n) for n in wx if n in fnames]
        assert len(fpos) == 3, (
            "the weather channels must enter the future block together; "
            f"got {[fnames[i] for i in fpos]}")

    datasets = {p: WindowDataset(master, ids, cols, scaler, L, H,
                                 contract.TARGET_CHANNEL, fcols or None,
                                 forecast=forecast, forecast_positions=fpos)
                for p, ids in parts.items()}

    out = {
        "c_in": len(cols) + len(fcols),
        "channels": scaler.channels + [f"{n}@future" for n in scaler.future_channels],
        "future_covariates": bool(fcols),
        "eval_mask": np.load(P / "eval_mask.npy"),
        "sensor_ids": np.load(P / "sensor_ids.npy", allow_pickle=True),
        "n_samples": {p: len(ids) for p, ids in parts.items()},
        "rung": rung,
        "split": split,
        "datasets": datasets,
    }

    if torch_loaders:
        try:
            import torch
            from torch.utils.data import DataLoader
        except ImportError:
            torch_loaders = False
    if torch_loaders:
        bs = batch_size or cfg["training"]["batch_size"]

        def collate(batch):
            xs, ys = zip(*batch)
            return (torch.from_numpy(np.stack(xs)), torch.from_numpy(np.stack(ys)))

        for p, ds in datasets.items():
            # NEVER shuffle val or test: their order is the chronological order
            # predictions are written in, and src/eval reads them positionally.
            out[p] = DataLoader(ds, batch_size=bs, shuffle=(p == "train"),
                                num_workers=num_workers, collate_fn=collate,
                                drop_last=False)
    else:
        out.update(datasets)

    adj = np.load(P / "adj_mx.npy")
    return out, adj, scaler


# ---------------------------------------------------------------------------

def main() -> int:
    """`python -m src.models.loader` - print what every rung and split gives."""
    import argparse
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--split", default="single")
    args = ap.parse_args()

    print(f"=== rungs, on split {args.split!r} ===")
    print(f"    input {contract.INPUT_WINDOW} steps (60 min), horizon "
          f"{contract.HORIZON} steps ({contract.HORIZON * 5} min)")
    print(f"    reported at {', '.join(contract.EVAL_HORIZON_STEPS)}\n")
    print(f"  {'rung':18s}{'hist':>5s}{'fut':>5s}{'c_in':>6s}  channels")
    for r in contract.ABLATION_RUNGS:
        d, adj, sc = build_loaders(rung=r, split=args.split, config=args.config,
                                   torch_loaders=False)
        nh = len(contract.rung_channels(r))
        print(f"  {r:18s}{nh:5d}{d['c_in'] - nh:5d}{d['c_in']:6d}  "
              f"{', '.join(d['channels'])}")
    print(f"\n  samples: {d['n_samples']}")
    print(f"  adjacency {adj.shape}, mean degree {(adj > 0).sum(1).mean():.2f}")
    print(f"  {sc}")
    print(f"  eval_mask usable {100 * d['eval_mask'].mean():.4f}%")

    print(f"\n=== all splits ===\n")
    print(f"  {'split':10s}{'train':>9s}{'val':>8s}{'test':>8s}"
          f"{'speed mean':>12s}{'std':>8s}")
    for s in available_splits(args.config):
        d, _, sc = build_loaders(rung="0_speed", split=s, config=args.config,
                                 torch_loaders=False)
        n = d["n_samples"]
        print(f"  {s:10s}{n['train']:>9,}{n['val']:>8,}{n['test']:>8,}"
              f"{sc.speed_mean:>12.3f}{sc.speed_std:>8.3f}")

    d, _, sc = build_loaders(rung="6_all", split=args.split, config=args.config,
                             torch_loaders=False)
    x, y = d["datasets"]["train"][0]
    print(f"\n=== one sample, rung 6_all ===")
    print(f"  X {x.shape} normalised, mean {x.mean():+.4f} std {x.std():.4f}")
    print(f"  Y {y.shape} raw mph,   min {y.min():.1f} max {y.max():.1f}")
    print(f"  target timesteps of the first 3 train samples: "
          f"{d['datasets']['train'].timesteps()[:3]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
