# Technical Report — Intelligent Handover Prediction in LTE Networks (v4)

**Author:** Souhail Bourhim  
**Repository:** [github.com/SouhailBourhim/Intelligent-Handover-Prediction-LTE](https://github.com/SouhailBourhim/Intelligent-Handover-Prediction-LTE)

This report documents the methodology and the design decisions. Headline results and figures are in the [README](../README.md); a full French course (telecom fundamentals → ML → experiments → interview Q&A) is in [`docs/cours/`](cours/README.md).

---

## 1. Problem statement

In LTE (RRC_CONNECTED), the serving eNodeB triggers a handover when the UE's L3-filtered RSRP measurements satisfy event A3 for the time-to-trigger. Handovers triggered too late fail (radio-link failure, dropped session); handovers triggered on transient fluctuations ping-pong.

**Task.** At every second *t*, predict for every UE whether a **successful** handover will happen within the next **3 s** (`handover_soon`), using only measurements a real network has at time *t*. This is binary time-series classification with 6.4 % positives.

**Second question.** Does acting on the prediction improve the network? Answered by putting the model inside the simulator's handover loop (§8).

## 2. Simulator (v4)

`src/simulator.py` runs one shared step loop (training data, scenario CSVs, generalization, closed loop). Each 1 s step:

1. **Measure.** Mobility (Random Waypoint, new cruise speed per leg) → per-eNB channel → RSRP → L3 filter → SINR, RSRQ, CQI.
2. **Decide.** An optional TTT policy, then the A3 state machine.

Mobility/channel and HO-failure draws use independent RNG streams.

| Component | Model | Reference |
|---|---|---|
| Path loss | UMa LOS / NLOS closed forms, 2 GHz | TR 36.873 |
| LOS probability | min(18/d,1)(1−e^(−d/63)) + e^(−d/63) | TR 36.873 |
| LOS state | spatially consistent latent AR(1) over distance, 50 m decorrelation | TR 38.901 §7.6.3 |
| Shadowing | log-normal, Gudmundson AR(1), σ 4/6 dB, 100 m | — |
| Fast fading | Rayleigh / Rician K = 5 dB, power-averaged over 1 + Δd/(λ/2) samples per 1 s measurement | — |
| RSRP | per resource element: 46 dBm − 10·log10(600) = 18.2 dBm; reported in [−140, −44] dBm | TS 36.214, 36.133 |
| SINR | load-weighted interference + kTB(9 MHz) + 9 dB NF = −95.5 dBm | — |
| CQI | 15-level table | TS 36.213 |
| L3 filter | F = (1−a)F + aM, a = 0.5 (k = 4) | TS 36.331 §5.5.3.2 |
| Trigger | A3 with 3 dB margin on L3 RSRP; TTT = max(1, 3 − ⌊v/7⌋) s (speed-dependent scaling analogue) | TS 36.331 §5.5.4 |
| Failure | p = 0.5·σ(SINR) + 0.15·σ(speed) + 0.15·ramp(target RSRP) + 0.2·(time at RSRP < −113 dBm), capped 0.75 | heuristic |
| RLF | exactly 3 s, then RRC re-establishment on the strongest cell | — |
| Ping-pong | successful HO reversing one within 10 s; +3 dB on the reverse pair for 20 s | — |

`event_type` (A3/A4/A5) is a descriptive tag of the dominant condition when the TTT expires. A3 is the only trigger.

### 2.1 Corrections from v3

| Issue in v3 | Correction | Effect |
|---|---|---|
| RSRP computed from total Tx power | per-RE power, thresholds −27.8 dB, −44 dBm cap | 0 % rows above −44 dBm (was 20 %); SINR unchanged by construction |
| LOS redrawn every 5 s independently of motion | spatially consistent LOS | pedestrian HO attempts 114 → 38 /h; pedestrian ping-pong 27 % → 11 % |
| single-sample fast fading reported as a 1 s average | measurement averaging | per-second jitter ±5 dB → ~1 dB (pedestrian) |
| constant per-UE speed, non-zero while paused | per-leg speed, actual speed | speed no longer a UE identifier |
| RLF of 2 steps, UE kept on the failed cell | ordering fix, re-establishment | as configured |
| failed attempts in the ping-pong history | executed HOs only | correct ping-pong detection |
| noise −107 dBm | −95.5 dBm (kTB + NF) | negligible (interference-limited) |

A soft-LOS transition variant (TR 38.901 soft LOS) was prototyped and increased pedestrian handovers (38 → 51 /h); it was not adopted.

### 2.2 Dataset (seed 42)

54,000 rows (15 UEs × 3,600 s), 27 columns. 1,434 HO attempts: 83.6 % successful, 16.4 % failed into RLF, 20.4 % of successes are ping-pongs. Event tags: A3 57.7 %, A4 24.8 %, A5 17.5 %. Positive rate 6.4 %.

## 3. Features

83 features per row, computed per UE in timestamp order (`src/features.py`):

- the current value plus lags (t−1..t−3), rolling mean and std over 3 s and 5 s (shifted by one step), and deltas over 1 s and 3 s, for each of 8 signals: serving/neighbour RSRP and RSRQ, serving/neighbour L3 RSRP, SINR, CQI;
- speed, `rsrp_diff`, serving-cell load.

`pos_x`, `pos_y` and `los_flag` are excluded: a network cannot observe them.

**Leakage controls.**

- Backward-only lags and shifted rolling windows, with a unit test that rewrites a UE's future and checks past features.
- A temporal 70/15/15 split by timestamp.
- The scaler is fit on train only.
- Thresholds and model selection use validation only.

`build_features()` accepts any row order and preserves the input index; the first 3 rows of each UE are flagged as warm-up and never zero-filled. `OnlineFeatureBuffer` reproduces the same features in streaming mode, with a parity test at 1e-8.

## 4. Models and training

| Model | Settings | Imbalance |
|---|---|---|
| Rule (A3 gap) | σ((ΔL3 − 3 dB)/2 dB), no training | — |
| Logistic Regression | L2, C = 0.5, saga | class_weight |
| Random Forest | 300 trees, depth 12, min leaf 5 | class_weight |
| XGBoost | depth 6, lr 0.05, 0.8/0.8 subsampling, ≤ 1,500 trees, early stopping (30) on validation **PR-AUC** | scale_pos_weight |
| LSTM / GRU | 2 × 64, dropout 0.3, FC 32, 10 s windows built per UE; Adam, plateau LR schedule; checkpoint on validation PR-AUC | WeightedRandomSampler only |
| Stacking | logistic meta-model over XGBoost/RF/LSTM probabilities, class-balanced; validation score out-of-fold (5 contiguous folds) | class_weight |

**Design decisions**

- **One imbalance correction per model.** v3 combined a balanced sampler *and* `pos_weight` in the LSTM/GRU loss, weighting positives by (n_neg/n_pos)². This produced precision ≈ 0.33.
- **PR-AUC for early stopping.** With log-loss, XGBoost ran to ~900 trees and its validation PR-AUC dropped (0.41 vs 0.47).
- **Thresholds tuned on validation.** Class-weighted models are not calibrated, so 0.5 has no special meaning. Each model gets its F1-optimal validation threshold, stored in `models/thresholds.json`.
- **Champion selection on validation:** maximise 0.6·F1_val + 0.4·AUC_val. The rule is not eligible. v3 selected on test.
- **Reproducibility:** all seeds fixed, deterministic PyTorch, seeded sampler. Two full `dvc repro` runs give identical metrics for all models.

## 5. Evaluation protocol

All models are scored on the same test rows: those with a full 10 s history, which the sequence models require (7,965 rows, 7.5 % positive). Metrics:

- precision, recall and F1 at the validation threshold and at 0.5;
- ROC-AUC and PR-AUC;
- confusion matrices;
- event-level metrics (§6).

| Model | P | R | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| **XGBoost** (champion) | 0.424 | 0.601 | 0.497 | 0.921 | 0.431 |
| LSTM | 0.413 | 0.627 | 0.498 | 0.916 | 0.423 |
| GRU | 0.405 | 0.628 | 0.492 | 0.916 | 0.414 |
| Stacking | 0.435 | 0.563 | 0.491 | 0.923 | 0.441 |
| Random Forest | 0.403 | 0.623 | 0.489 | 0.916 | 0.401 |
| Logistic Regression | 0.392 | 0.627 | 0.482 | 0.909 | 0.356 |
| Rule (A3 gap) | 0.296 | 0.603 | 0.397 | 0.853 | 0.262 |

The learned models are within noise of each other. Their gap to the rule (+0.10 F1, PR-AUC ×1.6) is the robust finding.

## 6. Operational metrics

`src/operational.py`, evaluated per UE in time order:

- **event recall@L:** the share of successful handovers with an alarm in the L seconds before (L = 3, 5);
- **mean lead time;**
- **false-alarm time share:** seconds with an alarm not followed by a successful handover within 3 s.

The cost is counted in time rather than in alarm episodes, because an episode count rewards a model that alarms permanently: one endless episode counts once. This was detected when the stacking model briefly appeared perfect under a tight episode budget, and a unit test now guards against it.

Operating curves sweep the threshold from 0.02 to 0.98. At ≤ 5 % false-alarm time:

| Stacking | XGBoost | GRU | RF | LR | LSTM | Rule |
|---|---|---|---|---|---|---|
| 85 % | 83 % | 81 % | 79 % | 78 % | 75 % | 68 % |

At its own threshold, the champion flags 89 % of handovers within 3 s, with a mean lead of 3.4 s and 6.1 % false-alarm time.

## 7. Explainability

The XGBoost champion's mean |SHAP| over the test set:

- speed 17.7 %, `rsrp_diff` 14.9 %, SINR 9.8 %, serving RSRQ 8.7 %;
- by family: current values and context 63 %, rolling statistics 20 %, lags 10 %, deltas 6 %.

`rsrp_diff` contributes strongly negatively while the serving cell dominates (corr(value, SHAP) = −0.84), which is A3's logic. Speed acts as a prior: vehicles hand over ~4× more often and use shorter TTTs.

## 8. Closed-loop experiment

`src/closed_loop.py` uses 3 unseen seeds × 1,800 s with identical mobility and channel across policies. The champion scores every UE online through `OnlineFeatureBuffer` → scaler → model.

**Policies**

- baseline (velocity TTT);
- always_short (TTT 1 s);
- always_long (TTT 4 s);
- predictive: p ≥ 0.80 → TTT 1 s; A3 met and p < 0.17 → TTT + 2 s;
- random_gate: the same overrides at the same rates, at random times.

Both thresholds come from validation: the champion's tuned threshold, and the 5th percentile of its validation scores on positives.

| Policy | HO failure | Ping-pong | RLF s/UE-h | SINR at HO | SINR < −5 dB |
|---|---|---|---|---|---|
| baseline | 14.9 % | 21.7 % | 62.1 | −6.2 dB | 3.9 % |
| random_gate | 14.2 % | 23.0 % | 60.6 | −5.9 dB | 3.7 % |
| predictive | 10.5 % | 29.4 % | 55.5 | −4.1 dB | 2.6 % |
| always_short | 11.0 % | 29.8 % | 59.6 | −4.0 dB | 2.5 % |
| always_long | 22.3 % | 7.5 % | 59.1 | −10.1 dB | 6.9 % |

**Conclusions**

1. At an equal intervention rate, the model's timing matters: −30 % failures, whereas the random control changes almost nothing.
2. The predictive policy is only marginally better than a plain short TTT (within one standard deviation over 3 seeds).
3. Both trade failures for ping-pongs, which is the classic MRO trade-off. Predicting *that* a handover is coming does not resolve it; predicting ping-pongs or handover quality is the natural next step.

**Caveat:** the model was trained on baseline-policy data (policy-induced distribution shift).

## 9. Generalization

Fresh simulations, never used for training or selection: four scenario presets (8 UEs × 1,800 s) and three unseen seeds of the training configuration.

- On unseen seeds the champion holds ROC-AUC 0.93 and PR-AUC 0.43–0.45, consistent with the test split.
- The vehicle-only scenario is hardest (ROC-AUC 0.87).
- The champion beats the rule on every dataset, in PR-AUC and in false-alarm time.

## 10. Engineering

- **DVC:** 8 stages. JSON reports and thresholds are git-tracked (`cache: false`).
- **MLflow** (SQLite): validation metrics at training; test metrics and plots appended to the same run.
- **GitHub Actions:** a pytest job, then the full `dvc repro`.
- **36 tests** covering radio physics, the HO state machine, determinism, policy invariance of trajectories, label correctness, leakage, online/offline parity, row-order invariance of serving, per-UE sequence windows and event metrics.
- **One inference path** (`src/serving.py`), shared by `predict.py`, the dashboard and the generalization study.

## 11. Limitations

- **Synthetic data:** validation on operator traces is the real test.
- **1 s time step:** TTT granularity is coarser than the standard's 40 ms – 5.12 s.
- **Hard LOS transitions** keep pedestrian handover rates above realistic values.
- **Small network:** four omni cells, no sectorisation or wrap-around.
- **Model simplifications:** approximate RSRQ, heuristic failure model, A4/A5 as tags rather than triggers.
- **Closed loop:** shows no gain over a tuned fixed TTT.

## 12. Reproduction

```bash
pip install -r requirements.txt -r requirements-dev.txt
dvc repro            # or: python simulate.py && python run_pipeline.py &&
                     #     python scripts/promote_best_model.py &&
                     #     python -m src.generalization && python -m src.closed_loop
pytest -q
python scripts/make_figures.py          # docs/img/
python scripts/make_course_figures.py   # docs/cours/img/
python scripts/build_course_pdf.py      # docs/cours/cours_projet_handover_lte.pdf
```
