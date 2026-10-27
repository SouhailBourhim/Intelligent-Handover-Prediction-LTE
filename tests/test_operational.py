import numpy as np
import pandas as pd
import pytest

from src.operational import event_metrics, operating_curve, recall_at_false_alarm_budget


def frame(events, failures=None):
    n = len(events)
    return pd.DataFrame({
        "ue_id": [0] * n, "timestamp": range(n),
        "handover_event": events,
        "handover_failure": failures or [0] * n,
    })


def test_alarm_before_handover_is_detected_with_lead_time():
    df = frame([0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
    prob = np.array([0, 0, 0.9, 0.9, 0, 0, 0, 0, 0, 0])      # alarms at t=2,3; HO at t=5
    m = event_metrics(df, prob, 0.5)
    assert m["events"] == 1
    assert m["event_recall@3s"] == 1.0
    assert m["mean_lead_s"] == 3.0
    assert m["false_alarm_time_share"] == 0.0


def test_late_or_missing_alarm_is_a_miss_and_useless_alarms_cost_time():
    df = frame([0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
    prob = np.array([0.9, 0, 0, 0, 0, 0.9, 0, 0, 0, 0.9])    # t=0 too early, t=5 too late
    m = event_metrics(df, prob, 0.5)
    assert m["event_recall@3s"] == 0.0
    assert m["event_recall@5s"] == 1.0                        # t=0 is 5 s before
    assert m["false_alarm_time_share"] == pytest.approx(0.3)  # t=0, t=5, t=9


def test_failed_handovers_are_not_events():
    df = frame([0, 0, 0, 1], failures=[0, 0, 0, 1])
    assert event_metrics(df, np.ones(4), 0.5)["events"] == 0


def test_permanent_alarm_is_expensive_not_free():
    """A model alarming all the time must not look like it has few false alarms."""
    df = frame([0] * 50 + [1] + [0] * 49)
    m = event_metrics(df, np.ones(100), 0.5)
    assert m["event_recall@3s"] == 1.0
    assert m["false_alarm_time_share"] > 0.9


def test_recall_at_budget_respects_the_budget():
    df = frame([0, 0, 0, 1] * 25)
    prob = np.tile([0.2, 0.6, 0.9, 0.1], 25)
    curve = operating_curve(df, prob)
    assert recall_at_false_alarm_budget(curve, 0.0) == 1.0     # only useful alarms exist
