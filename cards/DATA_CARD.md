# DATA_CARD — CycleBench / SynthCycle

> Fill remaining fields once mcPHASES is inspected. Synthetic release is PHI-free.

## Dataset summary
- **Name:** SynthCycle v1 (open) + optional local mcPHASES evaluation split (not redistributed unless license permits)
- **Grain:** one row per participant per calendar day
- **Schema:** `cyclebench/data_contract/` (`cyclebench_daily_v1`)
- **Synthetic n (default):** 80 participants × 60 days
- **License (synthetic):** MIT — no PHI
- **License (mcPHASES-derived):** TBD after PhysioNet / upstream DUA review — do **not** claim open until verified

## Sources
| Source | Role | Redistributable? |
|---|---|---|
| SynthCycle generator | open training cohort | **Yes** (MIT) |
| mcPHASES (PhysioNet) | real held-out TSTR anchor | Only if DUA allows |

## Missingness
Reported per column in `outputs/latest_metrics.json` → `missingness_synth`.
Synthetic default missing rate target ≈ 8%.

## Intended use
Research benchmark for masked multimodal hormonal-state reconstruction.
**Not** for diagnosis, fertility advice, or clinical decisions.

## Known limitations / bias
- Default synthetic priors are literature-informed, not patient-derived, until `fit_from_dataframe` is run on a real train split.
- Phase boundaries are approximate; cycle-day is excluded from features when the target is `cycle_phase` (leakage guard).
- Western wearable / CGM-centric feature set; representation gaps expected.

## Provenance rules
Every demo value shows `source` + `license_tag` + model version (`baseline.joblib`).
