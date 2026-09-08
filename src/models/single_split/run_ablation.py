"""Run the full ablation sweep: all rungs x all rolling folds.

    python -m src.models.single_split.run_ablation
    python -m src.models.single_split.run_ablation --splits single         # vanilla reproduction only
    python -m src.models.single_split.run_ablation --rungs 0_speed 6_all   # subset of rungs
    python -m src.models.single_split.run_ablation --future-covariates     # with forecast weather

After this finishes, run predict + eval for each:
    python -m src.models.single_split.run_ablation --predict-and-eval
"""

from __future__ import annotations

import argparse
import time

from src import contract
from src.models.loader import available_splits, load_config
from src.models.single_split.train import run as train_run
from src.models.single_split.predict import predict as predict_run


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rungs", nargs="+", default=None,
                    help="Rungs to train (default: all)")
    ap.add_argument("--splits", nargs="+", default=None,
                    help="Splits to train on (default: all rolling folds)")
    ap.add_argument("--seed", type=int, default=contract.SEED)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--device", default=None)
    ap.add_argument("--future-covariates", action="store_true",
                    help="Feed forecast weather for the target window")
    ap.add_argument("--predict-and-eval", action="store_true",
                    help="Run predict + eval instead of training")
    args = ap.parse_args()

    rungs = args.rungs or list(contract.ABLATION_RUNGS)
    if args.splits:
        splits = args.splits
    else:
        splits = [s for s in available_splits(args.config) if s.startswith("fold")]

    fc = args.future_covariates
    total = len(rungs) * len(splits)
    print(f"Ablation sweep: {len(rungs)} rungs x {len(splits)} splits = {total} runs")
    print(f"  rungs:  {rungs}")
    print(f"  splits: {splits}")
    print(f"  seed:   {args.seed}")
    print(f"  future_covariates: {fc}\n")

    if args.predict_and_eval:
        from src.eval.single_split.run import evaluate_split, _print_results
        for i, (rung, split) in enumerate((r, s) for r in rungs for s in splits):
            print(f"\n[{i+1}/{total}] predict + eval: {rung} / {split}")
            predict_run(rung, split, args.seed, args.config, args.device, fc)
            results = evaluate_split(rung, split, args.seed, args.config, fc)
            _print_results(results)
        return 0

    t_start = time.time()
    for i, (rung, split) in enumerate((r, s) for r in rungs for s in splits):
        print(f"\n{'#' * 65}")
        print(f"  [{i+1}/{total}] {rung} / {split}")
        print(f"{'#' * 65}")
        train_run(rung, split, args.seed, args.config, args.device, fc)

    elapsed = time.time() - t_start
    print(f"\n{'=' * 65}")
    print(f"  Done: {total} runs in {elapsed/3600:.1f} h")
    print(f"  Next: python -m src.models.run_ablation --predict-and-eval"
          + (" --future-covariates" if fc else ""))
    print(f"{'=' * 65}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
