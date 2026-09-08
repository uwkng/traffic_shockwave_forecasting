"""Pre-flight check. Everything that must hold before an unattended GPU run.

    python -m scripts.verify

Exits non-zero on the first hard failure. Warnings do not fail the run but are
printed so nothing is discovered at 3am.
"""
import json, pathlib, sys
import numpy as np, pandas as pd, yaml

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from src import contract                                    # noqa: E402
from src.eval import windows as W, metrics as M             # noqa: E402

FAIL, WARN = [], []
def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not cond: FAIL.append(name)
def warn(name, cond, detail=""):
    if not cond:
        print(f"  WARN  {name}   {detail}"); WARN.append(name)

cfg = yaml.safe_load(open("configs/default.yaml"))
proc = pathlib.Path(cfg["data"]["processed_dir"])

print("\n[1] contract self-consistency")
check("11 channels", contract.N_CHANNELS == 11)
check("target is speed", contract.CHANNELS[contract.TARGET_CHANNEL][0] == "speed")
allg = [c for g in contract.CHANNEL_GROUPS.values() for c in g]
check("groups partition the channels", sorted(allg) == sorted(n for n, _ in contract.CHANNELS),
      f"{len(allg)} grouped")
for r in contract.ABLATION_RUNGS:
    cols = contract.rung_channels(r)
    check(f"  rung {r}: indices valid, speed first",
          cols == sorted(set(cols)) and cols[0] == contract.TARGET_CHANNEL, f"c_in={len(cols)}")
check("eval horizons within HORIZON",
      max(contract.EVAL_HORIZON_STEPS.values()) <= contract.HORIZON,
      str(contract.EVAL_HORIZON_STEPS))

print("\n[2] artifacts on disk")
master = np.load(proc / "master.npy", mmap_mode="r")
ti = pd.DatetimeIndex(np.load(proc / "time_index.npy", allow_pickle=True))
ids = np.load(proc / "sensor_ids.npy", allow_pickle=True)
adj = np.load(proc / "adj_mx.npy")
dist = np.load(proc / "dist_to_venue.npy")
check("master shape", master.shape == (contract.N_TIMESTEPS, contract.N_NODES, 11), str(master.shape))
check("time axis length", len(ti) == contract.N_TIMESTEPS)
check("DST gap absent", not ((ti >= contract.DST_GAP[0]) & (ti <= contract.DST_GAP[1])).any())
check("node count", len(ids) == contract.N_NODES == len(dist))
check("master finite", bool(np.isfinite(np.asarray(master[::97])).all()))
check("dist_to_venue finite and >=0", bool(np.isfinite(dist).all() and (dist >= 0).all()),
      f"min {dist.min():.2f} max {dist.max():.2f} km")

print("\n[3] channel semantics match CHANNELS[].kind")
sub = np.asarray(master[::37])
for i, (name, kind) in enumerate(contract.CHANNELS):
    ch = sub[:, :, i]
    same_nodes = np.allclose(ch, ch[:, :1], atol=1e-5)
    same_time = np.allclose(ch, ch[:1, :], atol=1e-5)
    ok = (same_nodes if kind == "temporal" else
          same_time if kind == "spatial" else not same_nodes)
    check(f"  ch{i:2d} {name:<20s} {kind}", ok)

print("\n[4] freshness: nothing stale after the rebuild")
mt = {f.name: f.stat().st_mtime for f in proc.glob("*.np*")}
cfg_t = pathlib.Path("configs/default.yaml").stat().st_mtime
al_t = pathlib.Path("src/data/align.py").stat().st_mtime
for f in ("master.npy", "splits.npz", "dist_to_venue.npy", "event_load.npy", "scalers.json"):
    t = (proc / f).stat().st_mtime
    check(f"  {f} newer than config+align.py", t > max(cfg_t, al_t) - 1)

print("\n[5] splits: chronological, disjoint, scaler train-only")
sp = np.load(proc / "splits.npz")
meta = json.load(open(proc / "splits_meta.json"))
si = np.load(proc / "sample_index.npy")
for name in ["single"] + sorted(meta["rolling"]):
    tr, va, te = (sp[f"{name}__{k}"] for k in ("train", "val", "test"))
    check(f"  {name}: train/val/test disjoint",
          not (set(tr) & set(va)) and not (set(va) & set(te)) and not (set(tr) & set(te)))
    # splits.npz stores START TIMESTEPS (a subset of sample_index), not
    # positions into it - indexing sample_index with them is an off-by-domain bug.
    check(f"  {name}: chronological", tr.max() < va.min() < te.min(),
          f"n={len(tr)}/{len(va)}/{len(te)}")
    check(f"  {name}: all starts are valid samples",
          set(np.concatenate([tr, va, te])) <= set(si.tolist()))
    check(f"  {name}: last target within the time axis",
          int(te.max()) + contract.INPUT_WINDOW + contract.HORIZON - 1 < contract.N_TIMESTEPS,
          f"last target step {int(te.max()) + contract.INPUT_WINDOW + contract.HORIZON - 1}")
sc = json.load(open(proc / "scalers.json"))["splits"]
check("scalers exist per split", set(sc) >= set(["single"] + sorted(meta["rolling"])), str(sorted(sc)))
for name, one in sc.items():
    check(f"  {name}: per-channel scaler, 11 entries",
          len(one["mean"]) == 11 and len(one["std"]) == 11)
    check(f"  {name}: passthrough channels untouched (mean 0 std 1)",
          all(one["mean"][contract.CHANNEL_INDEX[c]] == 0.0
              and one["std"][contract.CHANNEL_INDEX[c]] == 1.0
              for c in contract.PASSTHROUGH_CHANNELS))
    check(f"  {name}: std strictly positive", min(one["std"]) > 0, f"min {min(one['std']):.4g}")
    tr = sp[f"{name}__train"]
    lo, hi = one["train_time_slice"]
    check(f"  {name}: scaler span covers train, ends before val",
          lo <= int(tr.min()) and hi >= int(tr.max()) + contract.INPUT_WINDOW + contract.HORIZON - 1
          and hi <= int(sp[f"{name}__val"].min()) + contract.INPUT_WINDOW + contract.HORIZON,
          f"slice {lo}..{hi}, train {int(tr.min())}..{int(tr.max())}, "
          f"val starts {int(sp[f'{name}__val'].min())}")

print("\n[6] eval windows")
rain = W.weather_windows(cfg["data"]["weather_csv"], ti)
egr, sub_ev = W.event_windows(cfg["data"]["events_csv"], ti)
hol = W.holiday_mask(ti)
check("event window uses high confidence only",
      set(sub_ev["confidence"].str.lower()) == {"high"}, f"{len(sub_ev)} fixtures")
check("rain window non-empty", rain.sum() > 0, f"{rain.sum()} steps, {M.episodes(rain)} episodes")
check("holiday window non-empty", hol.sum() > 0)
comp = W.weather_commute_windows(cfg["data"]["weather_csv"], ti)
check("compound window rain x commute is usable",
      M.episodes(comp) >= 20, f"{M.episodes(comp)} episodes, {comp.sum()} steps, "
      f"{len({str(t.date()) for t in ti[comp]})} days")
check("compound rain x EGRESS is NOT used anywhere (it has n=3)",
      M.episodes(rain & egr) < 5, f"{M.episodes(rain & egr)} episodes - documented as absent")
check("distance bins cover every node",
      int(((dist >= cfg["eval"]["distance_bins_km"][0])
           & (dist < cfg["eval"]["distance_bins_km"][-1])).sum()) == contract.N_NODES,
      f"bins {cfg['eval']['distance_bins_km']}, max dist {dist.max():.2f}")
check("config horizons match the contract",
      cfg["eval"]["horizons_min"] == [int(k.replace("min", ""))
                                      for k in contract.EVAL_HORIZON_STEPS],
      f"{cfg['eval']['horizons_min']} vs {list(contract.EVAL_HORIZON_STEPS)}")
check("config min_confidence matches windows.MIN_CONFIDENCE",
      cfg["features"]["event_load"]["min_confidence"] == W.MIN_CONFIDENCE,
      f"{cfg['features']['event_load']['min_confidence']} / {W.MIN_CONFIDENCE}")
check("align used the same confidence floor",
      json.load(open(proc / "align_meta.json"))["event_load"]["confidence_filter"]
      ["min_confidence"] == W.MIN_CONFIDENCE)
check("align used road distance",
      json.load(open(proc / "align_meta.json"))["venue_distance"]["mode"] == "road")
for f in ("fold00", "fold01", "fold02"):
    t = meta["rolling"][f]["test"]
    m = (ti >= pd.Timestamp(t["from"])) & (ti <= pd.Timestamp(t["to"]))
    check(f"  {f} test block has rain", (rain & m).any(),
          f"{M.episodes(rain & m)} rain episodes, {M.episodes(egr & m)} egress")

print("\n[7] model graph operator")
from src.models.stgcn import scaled_laplacian, cheb_polynomials
L = scaled_laplacian(adj)
ev = np.linalg.eigvalsh((L + L.T) / 2)
check("Laplacian symmetric", np.allclose(L, L.T, atol=1e-6))
check("rescaled spectrum in [-1,1]", ev.min() >= -1.001 and ev.max() <= 1.001,
      f"[{ev.min():.4f}, {ev.max():.4f}]")
check("raw adj_mx.npy still DIRECTED (propagation needs it)",
      not np.allclose(adj, adj.T), f"{int((np.abs(adj-adj.T)>1e-6).sum()//2)} asymmetric pairs")
check("cheb polynomials count", len(cheb_polynomials(L, cfg["model"]["stgcn"]["Ks"]))
      == cfg["model"]["stgcn"]["Ks"])

print("\n[8] shockwave label + propagation")
speed = np.nan_to_num(np.load(proc / "speed.npy").astype(np.float32), nan=65.0)
lab = M.shockwave_label(speed)
check("label warm-up rows are False", not lab[:M.SHOCKWAVE_DROP_STEPS].any())
check("label rate plausible", 0.001 < lab.mean() < 0.05, f"{100*lab.mean():.3f}% of cells")
prop = M.upstream_propagation(speed, adj)
check("upstream > downstream at 15 min", prop["15min"]["ratio"] > 1.2, str(prop["15min"]))

print("\n[9] loader round-trip, every rung")
from src.models.loader import build_loaders
for r in contract.ABLATION_RUNGS:
    ld, a, scaler = build_loaders(rung=r, split="fold00")
    x, y = next(iter(ld["train"]))
    check(f"  {r}: x{tuple(x.shape[1:])} y{tuple(y.shape[1:])} c_in={ld['c_in']}",
          x.shape[1] == contract.INPUT_WINDOW and x.shape[3] == ld["c_in"]
          and len(contract.rung_channels(r)) == ld["c_in"]
          and y.shape[1] == contract.HORIZON)
    check(f"  {r}: y is de-normalised mph", 0 < float(y.min()) and float(y.max()) < 100,
          f"[{float(y.min()):.1f}, {float(y.max()):.1f}]")

print("\n" + "=" * 62)
if FAIL:
    print(f"FAILED ({len(FAIL)}): " + "; ".join(FAIL)); sys.exit(1)
print(f"ALL CHECKS PASSED" + (f"  ({len(WARN)} warnings)" if WARN else ""))
