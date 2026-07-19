# CycleBench

**Open AI infrastructure for women's hormonal health.**

CycleBench turns fragmented longitudinal signals — wearables, continuous glucose, sleep, symptoms, and intermittent labs — into a **reproducible research benchmark** for reconstructing hormonal-state trajectories over time. Alongside the benchmark we release **SynthCycle**: a privacy-safe synthetic cohort under an open license, so researchers can train and compare models without waiting on restricted clinical data access.

> **Research only.** CycleBench produces auditable research estimates. It is **not** a medical device, diagnosis, or treatment recommendation.

Built solo at the Hack-Nation Global AI Hackathon · Challenge 05 — *Foundation Models for Women's Hormonal Health* (MIT Club of Northern California × MIT Club of Germany).

---

## The problem we address

Hormones shift continuously with sleep, stress, nutrition, and age — yet clinical care and most AI still see **occasional snapshots**. Women's hormonal physiology remains underrepresented in biomedical AI. Closing the women's health gap is estimated at more than **$1 trillion** in annual global economic impact; progress stalls without **shared datasets, benchmarks, and reproducible evaluation**.

CycleBench contributes one reusable layer: a **data contract + leakage-safe benchmark + baseline + open synthetic cohort + evaluation pipeline** that the next team can extend immediately.

---

## What we built

| Layer | Deliverable | Why it matters |
|---|---|---|
| **Data contract** | Canonical daily multimodal schema + data dictionary | Interoperable foundation others can map into |
| **SynthCycle** | Open, PHI-free synthetic longitudinal cohort + generator | Train and share without redistributing restricted patient data |
| **Benchmark** | Masked multimodal reconstruction task, participant-level splits | One clear prediction problem with transparent methodology |
| **Baseline** | Reproducible GBM + naive floor | A number others can beat — not a black box |
| **Evaluation** | Metrics, calibration, Train-on-Synthetic / Test-on-Real (TSTR) harness | Scientific validation, not a polished UI alone |
| **Demo** | Timeline: scattered streams → reconstructed trajectory with provenance | Makes the research asset *visible* |

### The scientific task

**Masked multimodal reconstruction:** from a trailing multimodal window (heart rate, HRV, activity, sleep, CGM, symptoms), reconstruct a held-out hormonal-state channel, evaluated on **held-out participants** (no person leaks across train/test).

### Headline evaluation protocols

| Protocol | Train | Test | Role |
|---|---|---|---|
| TSTS | SynthCycle | SynthCycle held-out | Always available, fully open |
| **TSTR** | SynthCycle (fitted on real train stats only) | Real held-out (mcPHASES, local) | Proves the open cohort transfers |
| TRTR | Real | Real held-out | Ceiling for comparison |

**Current headline (cycle-phase reconstruction, held-out participants):**  
`TSTR balanced_accuracy=0.295 (+0.045 vs naive, 90% of TRTR)`  
See [`outputs/latest_metrics.json`](outputs/latest_metrics.json). Modality ablations and an estrogen-reconstruction task live in [`outputs/ablation.json`](outputs/ablation.json) / [`outputs/metrics_estrogen.json`](outputs/metrics_estrogen.json) — reported honestly (estrogen remains an open problem: even TRTR does not beat a population-mean floor on this cohort). Real patient rows are **not** redistributed (PhysioNet Restricted); the open asset is SynthCycle + this evaluation recipe.

**Demo (no external UI tool required):** from the repo root run `python -m http.server 8765` then open [http://127.0.0.1:8765/demo/](http://127.0.0.1:8765/demo/) — press **Reconstruct trajectory**.

---

## How this matches a strong submission

Mapped to the challenge brief (*What Makes a Strong Submission* + success criteria):

| Strong submission asks for… | CycleBench delivers… |
|---|---|
| Publish reusable datasets, benchmarks, checkpoints, evaluation under an open license | **SynthCycle** (MIT, no PHI) + schema, splits, baseline artifact, evaluator, cards |
| Solve **one** clearly defined prediction/infrastructure problem with transparent methods | Single task: masked multimodal reconstruction; documented splits, metrics, leakage checks |
| Share reproducible code so others can extend the work | One-command generate + evaluate; DATA / BENCHMARK / MODEL cards |
| Avoid unsupported diagnostic claims / UI without validation | Explicit research-only framing; metrics vs naive baseline; provenance on outputs |
| **Women's Health Impact** | Infrastructure for continuous hormonal understanding — relevant to cycle, menopause, and underdiagnosed conditions affecting hundreds of millions |
| **Technical Excellence** | Participant-safe splits, missingness reporting, calibration, TSTR vs TRTR vs naive |
| **Foundation Value** | Leaves open assets that outlive the weekend — not an isolated app |

We deliberately did **not** ship a consumer wellness wrapper. The demo exists to *show* the benchmark working — the product is the scientific asset.

---

## Try it (reproduce in minutes)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python scripts/generate_synth.py     # open SynthCycle cohort
python scripts/run_benchmark.py      # metrics → outputs/latest_metrics.json
python scripts/smoke.py              # integrity checks

# Demo timeline
open demo/index.html
# or: python -m http.server 8000   then visit /demo/
```

No PhysioNet credentials required for the open path. Real-data TSTR uses publicly documented sources (e.g. [mcPHASES on PhysioNet](https://physionet.org/content/mcphases)) under their terms — derived patient rows are **not** redistributed unless upstream licensing permits.

---

## Repository layout

```
cyclebench/          Core library (schema, SynthCycle, benchmark, baselines, evaluation)
cards/               DATA_CARD · BENCHMARK · MODEL_CARD
demo/                Trajectory viewer (observed vs inferred + receipts)
scripts/             generate_synth · run_benchmark · smoke
outputs/             Latest metrics and demo artifacts
```

---

## Multimodal & responsible design

- **Modalities:** wearable physiology, CGM, sleep, structured symptoms (optional free-text → structured features via OpenAI or an offline heuristic — **ingestion only**, never labels or metrics).
- **Uncertainty & missingness** are first-class; observed vs model-inferred values are labeled in the demo.
- **No clinical claims.** Intended use, limitations, and failure modes: [`cards/MODEL_CARD.md`](cards/MODEL_CARD.md).

---

## Author & license

Architected and built **solo** for Hack-Nation 2026.

**MIT** for code and SynthCycle. See [`LICENSE`](LICENSE) and [`cards/DATA_CARD.md`](cards/DATA_CARD.md). Real clinical data remains under its upstream license and access rules.
