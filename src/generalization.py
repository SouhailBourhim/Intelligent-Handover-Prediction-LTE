"""
Generalization study: does the champion still work on networks it never saw?

Datasets (all simulated fresh, none used for training or model selection):
  * the 4 scenario presets of generate_test_dataset.py (default, vehicle,
    stable, cell_edge) — different UE populations and placements;
  * 3 unseen seeds of the training configuration (8 pedestrians + 7 vehicles).

For each dataset the champion and the rule baseline are scored through the
production inference path (src/serving.py) at their validation-tuned
thresholds.  Reported: ROC-AUC, PR-AUC, F1, event recall@3 s and
false-alarm time share.

Run:  python -m src.generalization      → reports/generalization.json
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score

os.environ.setdefault("OMP_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from generate_test_dataset import create_ues_for_scenario  # noqa: E402
from simulate import create_ues  # noqa: E402
from src.operational import event_metrics  # noqa: E402
from src.serving import load_champion, load_model, score_measurements  # noqa: E402
from src.simulator import add_handover_soon_label, make_simulator  # noqa: E402

SCENARIOS   = ["default", "vehicle", "stable", "cell_edge"]
SCEN_UES    = 8
UNSEEN      = [1001, 1002, 1003]
N_STEPS     = 1800
REPORT_PATH = ROOT / "reports" / "generalization.json"


def datasets():
    for sc in SCENARIOS:
        factory = lambda bss, rng, sc=sc: create_ues_for_scenario(sc, SCEN_UES, bss, rng)
        yield f"scenario: {sc}", add_handover_soon_label(make_simulator(factory, 500).run(N_STEPS))
    for seed in UNSEEN:
        yield f"training config, seed {seed}", add_handover_soon_label(
            make_simulator(create_ues, seed).run(N_STEPS))


def evaluate_on(df: pd.DataFrame, model) -> dict:
    scores = score_measurements(df, model)
    ok = ~scores["warmup"]
    y, p = df.loc[ok, "handover_soon"].values, scores.loc[ok, "prob"].values
    row = {"rows": int(ok.sum()), "pos_rate": round(float(y.mean()), 4)}
    if y.sum() == 0:                       # e.g. "stable": no handovers at all
        row.update({"roc_auc": None, "pr_auc": None, "f1": None,
                    "event_recall@3s": None,
                    "false_alarm_time_share": round(float((p >= model.threshold).mean()), 4)})
        return row
    ev = event_metrics(df.loc[ok], p, model.threshold)
    row.update({
        "roc_auc": round(roc_auc_score(y, p), 4),
        "pr_auc":  round(average_precision_score(y, p), 4),
        "f1":      round(f1_score(y, p >= model.threshold, zero_division=0), 4),
        "event_recall@3s":        ev["event_recall@3s"],
        "false_alarm_time_share": ev["false_alarm_time_share"],
    })
    return row


def main():
    print("Generalization study")
    champion = load_champion()
    rule     = load_model("Rule (A3 gap)")
    rows = []
    for name, df in datasets():
        for model in (champion, rule):
            r = {"dataset": name, "model": model.name, **evaluate_on(df, model)}
            rows.append(r)
            print(f"  {name:<28} {model.name:<15} AUC={r['roc_auc']}  PR-AUC={r['pr_auc']}  "
                  f"F1={r['f1']}  EvR@3s={r['event_recall@3s']}  FA={r['false_alarm_time_share']}")

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps({
        "champion": champion.name, "n_steps": N_STEPS, "rows": rows,
        "note": "Fresh simulations; thresholds tuned on the training run's validation split.",
    }, indent=2), encoding="utf-8")
    print(f"\n  Saved -> {REPORT_PATH}")


if __name__ == "__main__":
    main()
