"""SynthCycle — privacy-safe synthetic longitudinal cohort generator.

Fits simple per-phase marginals (+ mild temporal correlation) so the open
release contains no PHI. When real mcPHASES train stats are available, call
`fit_from_dataframe` first; otherwise `fit_default_physiology` uses
literature-informed priors sufficient for a working end-to-end demo.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

import numpy as np
import pandas as pd

from cyclebench.config import (
    CYCLE_LENGTH_MEAN,
    CYCLE_LENGTH_STD,
    DAYS_PER_PARTICIPANT,
    FEATURE_COLUMNS,
    N_SYNTH_PARTICIPANTS,
    PHASE_LABELS,
    SEED,
)
from cyclebench.data_contract.schema import validate_timeline


def _phase_for_day(cycle_day: int, cycle_len: int) -> str:
    """Rough phase boundaries scaled to cycle length."""
    menses = max(3, int(round(cycle_len * 0.18)))
    follicular_end = max(menses + 1, int(round(cycle_len * 0.45)))
    ovulatory_end = max(follicular_end + 1, int(round(cycle_len * 0.55)))
    if cycle_day <= menses:
        return "menstrual"
    if cycle_day <= follicular_end:
        return "follicular"
    if cycle_day <= ovulatory_end:
        return "ovulatory"
    return "luteal"


# Literature-informed priors (not patient data). Used when mcPHASES stats
# are not yet available so the pipeline runs during the download window.
DEFAULT_PHASE_STATS: dict[str, dict[str, tuple[float, float]]] = {
    # feature: (mean, std) per phase
    "menstrual": {
        "hr_mean": (72.0, 6.0),
        "hrv_rmssd": (38.0, 10.0),
        "steps": (6500, 2200),
        "sleep_hours": (6.8, 1.0),
        "sleep_efficiency": (0.84, 0.06),
        "cgm_mean": (98.0, 12.0),
        "cgm_std": (14.0, 5.0),
        "symptom_fatigue": (1.6, 0.7),
        "symptom_mood": (1.4, 0.7),
        "symptom_pain": (1.8, 0.8),
    },
    "follicular": {
        "hr_mean": (70.0, 5.5),
        "hrv_rmssd": (45.0, 11.0),
        "steps": (7800, 2400),
        "sleep_hours": (7.2, 0.9),
        "sleep_efficiency": (0.88, 0.05),
        "cgm_mean": (95.0, 11.0),
        "cgm_std": (12.0, 4.5),
        "symptom_fatigue": (0.8, 0.6),
        "symptom_mood": (0.7, 0.5),
        "symptom_pain": (0.5, 0.5),
    },
    "ovulatory": {
        "hr_mean": (73.0, 5.5),
        "hrv_rmssd": (42.0, 10.0),
        "steps": (8200, 2500),
        "sleep_hours": (7.0, 0.9),
        "sleep_efficiency": (0.87, 0.05),
        "cgm_mean": (96.0, 11.0),
        "cgm_std": (13.0, 4.5),
        "symptom_fatigue": (0.9, 0.6),
        "symptom_mood": (0.8, 0.6),
        "symptom_pain": (0.7, 0.6),
    },
    "luteal": {
        "hr_mean": (74.5, 6.0),
        "hrv_rmssd": (36.0, 10.0),
        "steps": (7000, 2300),
        "sleep_hours": (6.7, 1.0),
        "sleep_efficiency": (0.83, 0.06),
        "cgm_mean": (102.0, 13.0),
        "cgm_std": (16.0, 5.5),
        "symptom_fatigue": (1.5, 0.7),
        "symptom_mood": (1.5, 0.8),
        "symptom_pain": (1.2, 0.7),
    },
}

CLIP = {
    "hr_mean": (50, 110),
    "hrv_rmssd": (10, 120),
    "steps": (500, 20000),
    "sleep_hours": (3.0, 11.0),
    "sleep_efficiency": (0.55, 0.99),
    "cgm_mean": (70, 180),
    "cgm_std": (3, 40),
    "symptom_fatigue": (0, 3),
    "symptom_mood": (0, 3),
    "symptom_pain": (0, 3),
}


@dataclass
class SynthCycleGenerator:
    phase_stats: dict[str, dict[str, tuple[float, float]]] = field(
        default_factory=lambda: {p: dict(v) for p, v in DEFAULT_PHASE_STATS.items()}
    )
    missing_rate: float = 0.08
    seed: int = SEED
    fitted_from: str = "default_physiology_priors"

    def fit_from_dataframe(self, df: pd.DataFrame) -> "SynthCycleGenerator":
        """Estimate per-phase (mean, std) from a real train split."""
        stats: dict[str, dict[str, tuple[float, float]]] = {}
        feature_cols = [c for c in FEATURE_COLUMNS if c != "cycle_day"]
        for phase in PHASE_LABELS:
            sub = df[df["cycle_phase"] == phase]
            stats[phase] = {}
            for col in feature_cols:
                if sub.empty or sub[col].isna().all():
                    stats[phase][col] = DEFAULT_PHASE_STATS[phase][col]
                else:
                    mu = float(sub[col].mean())
                    sd = float(sub[col].std(ddof=1) or 1.0)
                    stats[phase][col] = (mu, max(sd, 1e-3))
        self.phase_stats = stats
        self.fitted_from = "mcphases_train_stats"
        return self

    def generate(
        self,
        n_participants: int = N_SYNTH_PARTICIPANTS,
        days: int = DAYS_PER_PARTICIPANT,
        start: date | None = None,
    ) -> pd.DataFrame:
        rng = np.random.default_rng(self.seed)
        start = start or date(2024, 1, 1)
        feature_cols = [c for c in FEATURE_COLUMNS if c != "cycle_day"]
        rows: list[dict[str, Any]] = []

        for i in range(n_participants):
            pid = f"synth_{i:04d}"
            cycle_len = int(round(rng.normal(CYCLE_LENGTH_MEAN, CYCLE_LENGTH_STD)))
            cycle_len = int(np.clip(cycle_len, 24, 35))
            cycle_day = int(rng.integers(1, cycle_len + 1))
            # mild participant-level offsets
            offsets = {c: float(rng.normal(0, 0.15)) for c in feature_cols}
            prev: dict[str, float] = {}

            for d in range(days):
                phase = _phase_for_day(cycle_day, cycle_len)
                day_date = start + timedelta(days=d)
                row: dict[str, Any] = {
                    "participant_id": pid,
                    "date": day_date.isoformat(),
                    "cycle_day": cycle_day,
                    "cycle_phase": phase,
                    "source": "synthetic",
                    "license_tag": "MIT-synthetic-no-PHI",
                }
                for col in feature_cols:
                    mu, sd = self.phase_stats[phase][col]
                    mu = mu * (1.0 + offsets[col])
                    val = float(rng.normal(mu, sd))
                    # AR(1) temporal smoothing
                    if col in prev:
                        val = 0.55 * prev[col] + 0.45 * val
                    lo, hi = CLIP[col]
                    val = float(np.clip(val, lo, hi))
                    # inject missingness
                    if rng.random() < self.missing_rate:
                        row[col] = np.nan
                    else:
                        row[col] = val
                        prev[col] = val
                rows.append(row)

                cycle_day += 1
                if cycle_day > cycle_len:
                    cycle_day = 1
                    cycle_len = int(round(rng.normal(CYCLE_LENGTH_MEAN, CYCLE_LENGTH_STD)))
                    cycle_len = int(np.clip(cycle_len, 24, 35))

        df = pd.DataFrame(rows)
        errors = validate_timeline(df)
        if errors:
            raise ValueError(f"synthetic timeline failed validation: {errors}")
        return df

    def manifest(self, df: pd.DataFrame) -> dict[str, Any]:
        return {
            "name": "SynthCycle",
            "n_participants": int(df["participant_id"].nunique()),
            "n_rows": int(len(df)),
            "days_span": int(df.groupby("participant_id").size().median()),
            "fitted_from": self.fitted_from,
            "missing_rate_target": self.missing_rate,
            "observed_missing_rate": float(df[[c for c in FEATURE_COLUMNS if c != "cycle_day"]].isna().mean().mean()),
            "phases": PHASE_LABELS,
            "license": "MIT — synthetic, no PHI",
            "seed": self.seed,
        }
