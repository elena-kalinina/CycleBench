# DATA_CARD — CycleBench / SynthCycle

## Dataset summary
- **Name:** SynthCycle (open) + local mcPHASES evaluation (not redistributed)
- **Grain:** one row per participant per calendar day
- **Schema:** `cyclebench/data_contract/` (`cyclebench_daily_v1`)
- **Synthetic:** generator + fitted-from-train cohort (MIT, no PHI)
- **Real eval:** mcPHASES PhysioNet v1.0.0 — **local only**

## Sources
| Source | Role | Redistributable? |
|---|---|---|
| SynthCycle generator | open training cohort | **Yes** (MIT) |
| mcPHASES (PhysioNet) | real held-out TSTR / TRTR anchor | **No** — Restricted Health Data License |

## Licensing (critical)
mcPHASES is under the **PhysioNet Restricted Health Data License**. CycleBench:
- evaluates on a **local** adapted daily table (`data/processed/`, gitignored)
- does **not** publish raw or derived patient rows
- releases **SynthCycle + code + aggregate metrics** so others can reproduce the method without restricted data access

## Missingness
Reported in `outputs/latest_metrics.json`. On the adapted mcPHASES daily table (local): CGM ~45% missing; symptoms ~41%; HRV ~14%; HR largely complete.

## Intended use
Research benchmark for masked multimodal hormonal-state reconstruction.
**Not** for diagnosis, fertility advice, or clinical decisions.

## Known limitations / bias
- Cycle-phase prediction from wearables alone is a hard ceiling (see TRTR in metrics) — we report TSTR vs that ceiling honestly.
- Phase label “Fertility” in mcPHASES is mapped to `ovulatory`.
- Western wearable / CGM-centric feature set; representation gaps expected.
- SynthCycle fitted only on the **train** participant split (no test leakage into the generator).

## Provenance rules
Demo values show `source` + `license_tag` + model version (`baseline.joblib`). Public demo uses **synthetic** participants only.
