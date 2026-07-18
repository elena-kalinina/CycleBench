# BENCHMARK — CycleBench

## Task (locked)
**Masked multimodal signal reconstruction** on a held-out hormonal-state channel.

Given a trailing 7-day multimodal window (HR, HRV, steps, sleep, CGM, symptoms),
predict the held-out target at day *t*, evaluated on **held-out participants**.

Default synthetic target: `cycle_phase` (categorical).
Swap to a continuous channel (e.g. `cgm_mean`) without changing the evaluator — it is target-agnostic.
Target is finalized after mcPHASES density inspection (H0–1).

## Splits
Participant-level, time-aware:
- train 60% / val 20% / test 20% of participants
- Leakage check: `assert_no_participant_leakage` (must be empty)

## Metrics
- Categorical: macro-F1, balanced accuracy, accuracy, ECE
- Continuous: MAE, RMSE
- Always report **delta vs naive** (population mode / mean)
- Ablation: missing-modality (TODO H10–13)

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
# → outputs/latest_metrics.json
```
