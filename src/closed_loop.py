"""
Closed-loop proactive handover experiment.

Question: does acting on the predictor improve the network — or would simply
changing the time-to-trigger (TTT) do the same?

The training simulator is re-run on unseen seeds with a TTT policy inside the
handover loop.  Every second the policy may override each UE's TTT:

  baseline      velocity-aware TTT, as in the training data
  always_short  TTT = 1 s for every UE (react fast everywhere)
  always_long   TTT = 4 s for every UE (filter transients everywhere)
  predictive    the champion model scores every UE online:
                  p ≥ hi                      → TTT = 1 s (prepare and act early)
                  p < lo while A3 is met      → TTT = default + 2 s (likely a
                                                transient: hold off)
                  otherwise                   → default TTT
  random_gate   control: the same two overrides applied at the same rates as
                `predictive`, but at random UE-seconds

hi is the champion's validation-tuned threshold; lo is the 5th percentile of
the champion's validation probabilities on true positives (only 5 % of real
handovers would be held back).  Neither is tuned on the experiment seeds.

Mobility and fading are identical across policies for a given seed (separate
RNG streams in src/simulator.py), so differences come from HO decisions only.

Caveat: the model was trained on baseline-policy data; once a policy changes
handover timing, the situations the model sees shift slightly.

Run:  python -m src.closed_loop      → reports/closed_loop.json
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

os.environ.setdefault("OMP_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from simulate import create_ues  # noqa: E402
from src.features import OnlineFeatureBuffer  # noqa: E402
from src.handover_logic import HO, velocity_ttt  # noqa: E402
from src.serving import champion_metadata, feature_columns, load_model  # noqa: E402
from src.simulator import make_simulator  # noqa: E402

SEEDS        = [101, 202, 303]     # unseen: the training data uses seed 42
N_STEPS      = 1800
SHORT_TTT    = 1
LONG_TTT     = 4
HOLD_EXTRA   = 2
REPORT_PATH  = ROOT / "reports" / "closed_loop.json"


# ── Policies ──────────────────────────────────────────────────────────────────

class FixedTTT:
    def __init__(self, ttt: int):
        self.ttt = ttt

    def __call__(self, t, recs):
        return {r["ue_id"]: self.ttt for r in recs}


class PredictivePolicy:
    """Scores every UE online with the champion and adapts its TTT."""

    def __init__(self, model, scaler, feat_cols, hi: float, lo: float):
        self.model, self.scaler, self.feat_cols = model, scaler, feat_cols
        self.hi, self.lo = hi, lo
        self.buffer = OnlineFeatureBuffer()
        self.n_steps = self.n_short = self.n_a3 = self.n_hold = 0

    def __call__(self, t, recs):
        feats = [(r, self.buffer.push(r)) for r in recs]
        ready = [(r, f) for r, f in feats if f is not None]
        out = {}
        self.n_steps += len(recs)
        if not ready:
            return out
        X = pd.DataFrame([f for _, f in ready])[self.feat_cols]
        p = self.model.predict_proba(self.scaler.transform(X).astype(np.float32))[:, 1]
        for (r, _), prob in zip(ready, p):
            a3_met = r["_a3_gap"] > HO.ho_margin_db
            self.n_a3 += a3_met
            if prob >= self.hi:
                out[r["ue_id"]] = SHORT_TTT
                self.n_short += 1
            elif a3_met and prob < self.lo:
                out[r["ue_id"]] = velocity_ttt(r["ue_speed"]) + HOLD_EXTRA
                self.n_hold += 1
        return out

    @property
    def rates(self) -> dict:
        return {"short_rate": self.n_short / max(self.n_steps, 1),
                "hold_rate_when_a3": self.n_hold / max(self.n_a3, 1)}


class RandomGate:
    """Applies the predictive policy's overrides at the same rates, at random."""

    def __init__(self, short_rate: float, hold_rate: float, seed: int):
        self.short_rate, self.hold_rate = short_rate, hold_rate
        self.rng = np.random.default_rng([seed, 7])

    def __call__(self, t, recs):
        out = {}
        for r in recs:
            if self.rng.random() < self.short_rate:
                out[r["ue_id"]] = SHORT_TTT
            elif r["_a3_gap"] > HO.ho_margin_db and self.rng.random() < self.hold_rate:
                out[r["ue_id"]] = velocity_ttt(r["ue_speed"]) + HOLD_EXTRA
        return out


# ── Metrics ───────────────────────────────────────────────────────────────────

def network_kpis(df: pd.DataFrame) -> dict:
    ue_hours = df["ue_id"].nunique() * df["timestamp"].nunique() / 3600.0
    ho  = df[df["handover_event"] == 1]
    ok  = ho[ho["handover_failure"] == 0]
    return {
        "ho_attempts_per_ue_hour": len(ho) / ue_hours,
        "ho_failure_rate":         ho["handover_failure"].mean() if len(ho) else 0.0,
        "ping_pong_rate":          ok["ping_pong"].mean() if len(ok) else 0.0,
        "rlf_seconds_per_ue_hour": df["rlf_flag"].sum() / ue_hours,
        "mean_sinr_at_ho_db":      ho["sinr"].mean() if len(ho) else float("nan"),
        "outage_share":            (df["sinr"] < -5.0).mean(),
    }


def _thresholds(model_name: str, model, scaler, feat_cols) -> tuple[float, float]:
    """hi = validation-tuned threshold; lo = 5th pct of val probs on positives."""
    thr = json.loads((ROOT / "models" / "thresholds.json").read_text())[model_name]["threshold"]
    val = pd.read_csv(ROOT / "data" / "processed" / "val.csv")   # already scaled
    p   = model.predict_proba(val[feat_cols].fillna(0).values.astype(np.float32))[:, 1]
    lo  = float(np.quantile(p[val["handover_soon"].values == 1], 0.05))
    return float(thr), min(lo, float(thr))


def run_experiment(seeds=SEEDS, n_steps=N_STEPS) -> dict:
    meta = champion_metadata()
    name = meta["display_name"]
    loaded = load_model(name)
    if loaded.kind != "sklearn":
        # Online scoring of sequence / stacking models would need per-UE
        # windows inside the loop; the best tree model stands in for them.
        name, loaded = "XGBoost", load_model("XGBoost")
        print(f"  champion is not a per-row model; using {name} in the loop")
    scaler    = joblib.load(ROOT / "models" / "scaler.pkl")
    feat_cols = feature_columns()
    hi, lo    = _thresholds(name, loaded.model, scaler, feat_cols)
    print(f"  model in the loop: {name}  hi={hi:.2f}  lo={lo:.3f}")

    per_seed = []
    for seed in seeds:
        res = {}
        t0 = time.time()
        res["baseline"]     = network_kpis(make_simulator(create_ues, seed).run(n_steps))
        res["always_short"] = network_kpis(make_simulator(create_ues, seed, FixedTTT(SHORT_TTT)).run(n_steps))
        res["always_long"]  = network_kpis(make_simulator(create_ues, seed, FixedTTT(LONG_TTT)).run(n_steps))
        pred = PredictivePolicy(loaded.model, scaler, feat_cols, hi, lo)
        res["predictive"]   = network_kpis(make_simulator(create_ues, seed, pred).run(n_steps))
        rates = pred.rates
        gate = RandomGate(rates["short_rate"], rates["hold_rate_when_a3"], seed)
        res["random_gate"]  = network_kpis(make_simulator(create_ues, seed, gate).run(n_steps))
        per_seed.append({"seed": seed, "rates": rates, "kpis": res})
        print(f"  seed {seed}: done in {time.time() - t0:.0f}s  "
              f"(short {rates['short_rate']:.1%}, hold {rates['hold_rate_when_a3']:.1%} of A3 steps)")

    policies = {}
    for pol in per_seed[0]["kpis"]:
        policies[pol] = {}
        for k in per_seed[0]["kpis"][pol]:
            vals = np.array([s["kpis"][pol][k] for s in per_seed], dtype=float)
            policies[pol][k] = {"mean": round(float(np.nanmean(vals)), 4),
                                "std":  round(float(np.nanstd(vals)), 4)}

    return {
        "model": name, "hi": round(hi, 3), "lo": round(lo, 3),
        "seeds": list(seeds), "n_steps": n_steps,
        "policies": policies, "per_seed": per_seed,
        "note": ("Unseen seeds; mobility and fading identical across policies. "
                 "The model was trained on baseline-policy data, so policies that "
                 "change handover timing shift its input distribution slightly."),
    }


def main():
    print("Closed-loop proactive handover experiment")
    report = run_experiment()
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")

    keys = ["ho_attempts_per_ue_hour", "ho_failure_rate", "ping_pong_rate",
            "rlf_seconds_per_ue_hour", "mean_sinr_at_ho_db", "outage_share"]
    print(f"\n  {'policy':<14}" + "".join(f"{k[:18]:>20}" for k in keys))
    for pol, m in report["policies"].items():
        print(f"  {pol:<14}" + "".join(f"{m[k]['mean']:>14.4f} ±{m[k]['std']:<5.3f}" for k in keys))
    print(f"\n  Saved -> {REPORT_PATH}")


if __name__ == "__main__":
    main()
