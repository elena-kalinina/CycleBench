# BENCHMARK — CycleBench

## Task (locked)
**Masked multimodal signal reconstruction** on a held-out hormonal-state channel.

Given a trailing 7-day multimodal window (HR, HRV, steps, sleep, CGM, symptoms),
predict the held-out target at day *t*, evaluated on **held-out participants**.

Default synthetic target: `cycle_phase` (categorical).
Swap to a continuous channel (e.g. `cgm_mean`) without changing the evaluator — it is target-agnostic.

## Splits
**Participant-level** 60% / 20% / 20% with `assert_no_participant_leakage`.

Why participant-level (not a global calendar cut): people start on different dates; the
leakage risk is *identity* (the same person's physiology in train and test), not clock time.
Time-awareness within a person comes from **trailing 7-day windows** (no future features).
The val split is reserved for model selection; the GBM baseline trains on train and
reports on test (plus TSTR/TRTR protocols).

## Metrics
- Categorical: macro-F1, balanced accuracy, accuracy, ECE (calibration)
- Continuous: MAE, RMSE
- Always report **delta vs naive** (population mode / mean)
- Modality ablations: `scripts/run_ablations.py` → `outputs/ablation.json`

## Protocols
| Protocol | Train | Test | When |
|---|---|---|---|
| **TSTS** | synthetic | synthetic held-out | always (green path) |
| **TSTR** | synthetic | real held-out | when mcPHASES adapted |
| **TRTR** | real | real held-out | ceiling once real available |

## Headline
Prefer TSTR. If only TSTS is available, say so honestly in the demo:
*"synth-only validation; real TSTR is the reproducible next step."*

## Reproduce
```bash
python scripts/generate_synth.py
python scripts/run_benchmark.py
python scripts/run_ablations.py   # optional modality table
# → outputs/latest_metrics.json, outputs/ablation.json
```
