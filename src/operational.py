"""
Operational (event-level) evaluation of a handover predictor.

Row-level precision / recall count every second separately.  An operator
cares about handover events and about the cost of acting on alarms:

  * event recall@L      — share of successful handovers for which the model
                          raised an alarm at least once in the L seconds before;
  * mean lead time      — how early the first alarm in that window came (s);
  * false-alarm time    — share of seconds with an alarm that is NOT followed by
                          a successful handover within 3 s, i.e. the share of
                          time the network would prepare a handover for nothing.

A per-episode false-alarm count would reward a model that alarms permanently
(one endless episode counts once), so the cost is measured in time instead.

Alarms are `prob >= threshold`, evaluated per UE in timestamp order.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

LEAD_WINDOWS = (3, 5)     # seconds before the handover that count as "in time"
FA_HORIZON   = 3          # an alarm is useful if a successful HO follows within 3 s


def event_metrics(df: pd.DataFrame, prob: np.ndarray, threshold: float) -> dict:
    """
    Event-level metrics for one model at one threshold.

    df   : rows with ue_id, timestamp, handover_event, handover_failure
    prob : predicted P(handover_soon) for those rows, same order as df
    """
    d = df[["ue_id", "timestamp", "handover_event", "handover_failure"]].copy()
    d["alarm"] = np.asarray(prob) >= threshold

    n_events, hit = 0, {L: 0 for L in LEAD_WINDOWS}
    leads, false_alarm_s, alarm_s, ue_seconds = [], 0, 0, 0

    for _, g in d.sort_values(["ue_id", "timestamp"]).groupby("ue_id"):
        t     = g["timestamp"].to_numpy()
        alarm = g["alarm"].to_numpy()
        succ  = set(t[(g["handover_event"].to_numpy() == 1) &
                      (g["handover_failure"].to_numpy() == 0)].tolist())
        alarm_at = dict(zip(t.tolist(), alarm.tolist()))
        ue_seconds += len(g)
        alarm_s    += int(alarm.sum())

        for ts in succ:
            n_events += 1
            lags = [k for k in range(1, max(LEAD_WINDOWS) + 1) if alarm_at.get(ts - k, False)]
            if lags:
                leads.append(max(lags))                       # earliest alarm
            for L in LEAD_WINDOWS:
                if any(k <= L for k in lags):
                    hit[L] += 1

        for ts, a in zip(t.tolist(), alarm.tolist()):
            if a and not any((ts + k) in succ for k in range(1, FA_HORIZON + 1)):
                false_alarm_s += 1

    return {
        "threshold":              round(float(threshold), 2),
        "events":                 int(n_events),
        **{f"event_recall@{L}s": round(hit[L] / max(n_events, 1), 4) for L in LEAD_WINDOWS},
        "mean_lead_s":            round(float(np.mean(leads)), 2) if leads else None,
        "alarm_time_share":       round(alarm_s / max(ue_seconds, 1), 4),
        "false_alarm_time_share": round(false_alarm_s / max(ue_seconds, 1), 4),
    }


def operating_curve(df: pd.DataFrame, prob: np.ndarray,
                    thresholds=np.round(np.arange(0.02, 0.99, 0.02), 2)) -> list[dict]:
    """
    Event recall@3s and false-alarm time share across thresholds, so models
    can be compared at the same alarm cost instead of at one threshold each.
    """
    keys = ("threshold", "event_recall@3s", "false_alarm_time_share")
    return [{k: m[k] for k in keys}
            for m in (event_metrics(df, prob, t) for t in thresholds)]


def recall_at_false_alarm_budget(curve: list[dict], budget: float) -> float:
    """Best event recall@3s among thresholds whose false-alarm time share ≤ budget."""
    ok = [c["event_recall@3s"] for c in curve if c["false_alarm_time_share"] <= budget]
    return max(ok) if ok else 0.0
