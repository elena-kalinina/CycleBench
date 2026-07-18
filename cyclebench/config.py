"""CycleBench configuration — paths, task defaults, reproducibility."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
SYNTH_DIR = DATA_DIR / "synthetic"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = ROOT / "outputs"

# Reproducibility
SEED = 42

# Synthetic cohort defaults (PHI-free; runs without mcPHASES)
N_SYNTH_PARTICIPANTS = 80
DAYS_PER_PARTICIPANT = 60
CYCLE_LENGTH_MEAN = 28.0
CYCLE_LENGTH_STD = 2.5

# Benchmark
WINDOW_DAYS = 7
TRAIN_FRAC = 0.60
VAL_FRAC = 0.20
TEST_FRAC = 0.20

# Locked task: masked multimodal reconstruction.
# Target chosen by data density at H0–1. Default for synthetic path:
# cycle_phase (categorical). Swap to a continuous channel (e.g. cgm_mean)
# without changing the evaluator — it is target-agnostic.
DEFAULT_TARGET = "cycle_phase"
TARGET_KIND = "categorical"  # "categorical" | "continuous"

FEATURE_COLUMNS = [
    "hr_mean",
    "hrv_rmssd",
    "steps",
    "sleep_hours",
    "sleep_efficiency",
    "cgm_mean",
    "cgm_std",
    "symptom_fatigue",
    "symptom_mood",
    "symptom_pain",
    "cycle_day",
]

PHASE_LABELS = ["menstrual", "follicular", "ovulatory", "luteal"]

# Provenance tags for the timeline demo
STATUS_OBSERVED = "observed"
STATUS_INFERRED = "inferred"
STATUS_MISSING = "missing"
