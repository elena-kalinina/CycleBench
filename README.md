# CycleBench

**An open benchmark for reconstructing women's hormonal-state trajectories from scattered multimodal signals — plus SynthCycle, a privacy-safe synthetic cohort you can train on.**

> Research-only. Not a diagnosis. Built solo at [Hack-Nation](https://hack-nation.ai) · Women's Hormonal Health challenge (MIT Clubs).

**One sentence:** Point CycleBench at fragmented wearable, glucose, sleep, and symptom signals and it reconstructs an auditable hormonal-state trajectory — then publishes the schema, splits, baseline, evaluator, and an open synthetic cohort so the next researcher can improve it fairly.

## Why this exists

Women's hormonal health lacks shared AI infrastructure (no ImageNet-style foundation). Care still sees **snapshots**; researchers cannot compare methods. CycleBench leaves behind a **reusable scientific asset**: a data contract, a leakage-safe benchmark, a simple baseline, and **SynthCycle** — a PHI-free cohort that is cleanly open-licensable even when PhysioNet-derived artifacts cannot be redistributed.

## The wow

1. **Mess → trajectory:** five noisy streams snap into one uncertainty-aware timeline (observed vs inferred, receipts).
2. **Proof:** the model can train **only on SynthCycle** and still be evaluated on held-out real data (**TSTR**). Headline metric in `outputs/latest_metrics.json`.

## Quick start (synthetic path — no PhysioNet required)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python scripts/smoke.py              # must stay green
python scripts/generate_synth.py     # open cohort → data/synthetic/
python scripts/run_benchmark.py      # metrics → outputs/latest_metrics.json

# timeline demo
open demo/index.html                 # or: python -m http.server 8000  (from repo root)
```

## Repo map

```
cyclebench/
  data_contract/   # canonical daily schema + dictionary
  adapters/        # mcPHASES adapter (wire when download lands)
  preprocess/      # participant-safe splits, missingness
  synthcycle/      # open synthetic generator  ← redistributable
  benchmark/       # windowed masked-reconstruction task
  baselines/       # GBM + naive floor
  evaluation/      # metrics, TSTR harness
  ingest/          # symptom text → structured features (OpenAI optional)
cards/             # DATA / BENCHMARK / MODEL cards
demo/              # timeline viewer
scripts/           # smoke · generate_synth · run_benchmark
```

## Task (locked)

**Masked multimodal reconstruction** of a held-out hormonal-state channel from a trailing 7-day window, evaluated on **held-out participants**.

Default synthetic target: `cycle_phase`. Target is finalized after mcPHASES density inspection — the evaluator is **target-agnostic**.

## Protocols

| Name | Train | Test |
|---|---|---|
| TSTS | SynthCycle | SynthCycle held-out |
| **TSTR** (headline) | SynthCycle | real held-out |
| TRTR (ceiling) | real | real held-out |

## mcPHASES

Place the extracted PhysioNet package at `data/raw/mcphases/`, then:

```bash
python -c "from cyclebench.adapters.mcphases import inspect; import json; print(json.dumps(inspect(), indent=2))"
```

Implement column mapping in `cyclebench/adapters/mcphases.py::adapt()`, then:

```bash
python scripts/run_benchmark.py --real data/processed/mcphases_daily.parquet
```

**License honesty:** claim an open license on real-derived artifacts only after upstream DUA review. SynthCycle is MIT either way.

## Responsible design

- Calibrated / uncertainty-aware research estimates — **not** diagnostic claims
- Provenance on every displayed value
- OpenAI (if used) is ingestion-only: never labels, never metrics

## Author

Architected and built **solo** for Hack-Nation 2026.

## License

MIT for code + SynthCycle. See `LICENSE` and `cards/DATA_CARD.md`.
