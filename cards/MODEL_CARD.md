# MODEL_CARD — CycleBench baseline

## Model
- **Name:** CycleBench GBM baseline (`baselines/models.py`)
- **Type:** GradientBoostingClassifier / Regressor + median imputer
- **Version artifact:** `outputs/baseline.joblib`

## Intended use
Research-only reconstruction of a held-out hormonal-state channel from multimodal
wearable / CGM / sleep / symptom windows.

## Out of scope
- Diagnosis or disease detection (PCOS, endometriosis, etc.)
- Fertility / contraception / treatment recommendations
- Clinical deployment without independent validation

## Training data
- Primary open release: **SynthCycle** (MIT, no PHI)
- Optional: real mcPHASES train split for TRTR ceiling / fitting synth stats
  (not redistributed unless license permits)

## Evaluation
See `cards/BENCHMARK.md` and `outputs/latest_metrics.json`.
Primary claim uses **TSTR** when real held-out data exists; otherwise **TSTS** with explicit labeling.

## Ethical considerations
- De-identified / synthetic only in the public release
- Uncertainty and missingness are first-class (not hidden)
- OpenAI (if used) is limited to **symptom text → structured features** at ingestion —
  never labels, never metrics, never health findings

## Failure modes
- Domain shift if synthetic priors ≠ real population
- Sparse labs → weak target; switch target channel via config
- Missing entire modalities → expect ablation drop (report it)
