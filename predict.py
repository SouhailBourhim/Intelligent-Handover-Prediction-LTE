"""
predict.py — Champion-model inference for LTE handover risk.

Scores raw UE measurement rows with the promoted champion model and prints,
for each row, the probability of a successful handover within the next 3 s
and the decision at the champion's validation-tuned threshold.

Usage
-----
# Score a measurement CSV (raw signals; one or more UEs, any row order):
python predict.py --csv data/test_scenarios/default_ues5_steps300_seed42.csv

# Score rows given as JSON (a list of objects, oldest first):
python predict.py --json '[{"ue_id":0,"timestamp":0,"rsrp_serving":-95, ...}, ...]'

# Override the decision threshold:
python predict.py --csv measurements.csv --threshold 0.35

# Show which model is currently champion:
python predict.py --info

Required input columns
----------------------
  rsrp_serving, rsrq_serving, sinr, cqi, rsrp_neighbor, rsrq_neighbor,
  l3_rsrp_serving, l3_rsrp_neighbor, rsrp_diff, ue_speed, cell_load_pct
  plus ue_id and timestamp when the file holds several UEs or is not in
  time order.

Lags, rolling statistics and deltas are engineered from these columns per UE
(src/features.build_features).  The first 3 rows of every UE have no lag
history yet: they are listed as "warm-up" and not scored.

Exit code 0 = success, 1 = champion model or input missing.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd

os.environ.setdefault("OMP_NUM_THREADS", "1")
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.serving import champion_metadata, load_champion, score_measurements  # noqa: E402


def _print_info():
    try:
        m = champion_metadata()
    except FileNotFoundError as e:
        print(e)
        return
    print("\nChampion model")
    print(f"   Name        : {m['display_name']}  ({m['model_name']})")
    print(f"   Threshold   : {m['threshold']:.2f}  (tuned on validation)")
    print(f"   Validation  : F1={m['val_f1']:.4f}  AUC={m['val_roc_auc']:.4f}")
    if m.get("test_f1") is not None:
        print(f"   Test        : F1={m['test_f1']:.4f}  AUC={m['test_roc_auc']:.4f}  "
              f"PR-AUC={m['test_pr_auc']:.4f}")
    print(f"   Promoted    : {str(m.get('promoted_at', ''))[:19].replace('T', ' ')}\n")


def _print_results(df: pd.DataFrame, scores: pd.DataFrame, threshold: float):
    print(f"\n{'UE':>4} {'t':>6} {'Prob':>8}  Decision  (threshold={threshold:.2f})")
    print("-" * 44)
    for i in df.index:
        ue = df.at[i, "ue_id"] if "ue_id" in df else 0
        t  = df.at[i, "timestamp"] if "timestamp" in df else i
        if scores.at[i, "warmup"]:
            print(f"{ue:>4} {t:>6} {'-':>8}  warm-up (needs 3 s of history)")
        else:
            p = scores.at[i, "prob"]
            print(f"{ue:>4} {t:>6} {p:>8.4f}  {'HANDOVER' if p >= threshold else 'ok'}")
    scored = (~scores["warmup"]).sum()
    alarms = (scores["prob"] >= threshold).sum()
    print(f"\n{scored} of {len(df)} rows scored, {len(df) - scored} warm-up, "
          f"{alarms} handover alarms.\n")


def main():
    parser = argparse.ArgumentParser(
        description="Score UE measurements against the champion handover model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--json", metavar="JSON",
                       help="JSON object or list of objects (one per measurement row).")
    group.add_argument("--csv", metavar="FILE",
                       help="CSV file with one or more measurement rows.")
    group.add_argument("--info", action="store_true",
                       help="Print champion model info and exit.")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Decision threshold (default: champion's tuned threshold).")
    args = parser.parse_args()

    if args.info or (not args.json and not args.csv):
        _print_info()
        return

    try:
        model = load_champion()
    except FileNotFoundError as e:
        sys.exit(str(e))
    if args.threshold is not None:
        model.threshold = args.threshold

    if args.json:
        try:
            rows = json.loads(args.json)
        except json.JSONDecodeError as e:
            sys.exit(f"Invalid JSON: {e}")
        df = pd.DataFrame(rows if isinstance(rows, list) else [rows])
    else:
        path = Path(args.csv)
        if not path.exists():
            sys.exit(f"File not found: {path}")
        df = pd.read_csv(path)

    try:
        scores = score_measurements(df, model)
    except ValueError as e:
        sys.exit(str(e))

    print(f"\nModel: {model.name}")
    _print_results(df, scores, model.threshold)


if __name__ == "__main__":
    main()
