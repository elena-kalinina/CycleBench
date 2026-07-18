# CycleBench Data Dictionary — daily timeline v1

Grain: one row per participant per calendar day.

| Column | Type | Description | Provenance |
|---|---|---|---|
| participant_id | str | De-identified participant key | source |
| date | date | Calendar day (ISO) | source |
| cycle_day | int | Day within current cycle (1 = menses start) | source / derived |
| cycle_phase | cat | menstrual / follicular / ovulatory / luteal | source / derived |
| hr_mean | float | Mean heart rate (bpm) | wearable |
| hrv_rmssd | float | HRV RMSSD (ms) | wearable |
| steps | float | Daily step count | wearable |
| sleep_hours | float | Total sleep (hours) | wearable |
| sleep_efficiency | float | Sleep efficiency 0–1 | wearable |
| cgm_mean | float | Mean interstitial glucose (mg/dL) | CGM |
| cgm_std | float | Glucose std within day | CGM |
| symptom_fatigue | float | Fatigue 0–3 | diary / OpenAI-structured |
| symptom_mood | float | Mood disturbance 0–3 | diary / OpenAI-structured |
| symptom_pain | float | Pain 0–3 | diary / OpenAI-structured |
| source | str | `mcphases` \| `synthetic` | adapter |
| license_tag | str | Upstream / release license tag | adapter |

## Missingness

Missing values are allowed and **must be preserved** (not silently imputed in the released timeline).
Imputation, if any, happens inside the baseline and is reported in the evaluation.

## Licensing

- Synthetic cohort (`source=synthetic`): released under MIT with this repo — no PHI.
- Real mcPHASES-derived artifacts: only released if PhysioNet / upstream DUA permits.
  Otherwise we ship the *recipe* (adapter + schema) and evaluate against a local
  held-out split that is **not** redistributed.
