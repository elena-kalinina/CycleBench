"""SynthCycle — privacy-safe synthetic longitudinal cohort generator.

Modes:
- independent per-phase Gaussians + AR(1) (default / fast)
- **correlated** (fancy): per-phase empirical covariance over features, MVN
  draw + AR(1) — better cross-modal structure for TSTR transfer

Fits only on a provided train dataframe (no test leakage). Open release = no PHI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any, Literal

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


DEFAULT_PHASE_STATS: dict[str, dict[str, tuple[float, float]]] = {
    "menstrual": {
        "hr_mean": (72.0, 6.0), "hrv_rmssd": (38.0, 10.0), "steps": (6500, 2200),
        "sleep_hours": (6.8, 1.0), "sleep_efficiency": (0.84, 0.06),
        "cgm_mean": (98.0, 12.0), "cgm_std": (14.0, 5.0),
        "symptom_fatigue": (1.6, 0.7), "symptom_mood": (1.4, 0.7), "symptom_pain": (1.8, 0.8),
        "estrogen": (80.0, 35.0), "lh": (4.0, 2.5),
    },
    "follicular": {
        "hr_mean": (70.0, 5.5), "hrv_rmssd": (45.0, 11.0), "steps": (7800, 2400),
        "sleep_hours": (7.2, 0.9), "sleep_efficiency": (0.88, 0.05),
        "cgm_mean": (95.0, 11.0), "cgm_std": (12.0, 4.5),
        "symptom_fatigue": (0.8, 0.6), "symptom_mood": (0.7, 0.5), "symptom_pain": (0.5, 0.5),
        "estrogen": (180.0, 70.0), "lh": (6.0, 3.0),
    },
    "ovulatory": {
        "hr_mean": (73.0, 5.5), "hrv_rmssd": (42.0, 10.0), "steps": (8200, 2500),
        "sleep_hours": (7.0, 0.9), "sleep_efficiency": (0.87, 0.05),
        "cgm_mean": (96.0, 11.0), "cgm_std": (13.0, 4.5),
        "symptom_fatigue": (0.9, 0.6), "symptom_mood": (0.8, 0.6), "symptom_pain": (0.7, 0.6),
        "estrogen": (250.0, 90.0), "lh": (25.0, 15.0),
    },
    "luteal": {
        "hr_mean": (74.5, 6.0), "hrv_rmssd": (36.0, 10.0), "steps": (7000, 2300),
        "sleep_hours": (6.7, 1.0), "sleep_efficiency": (0.83, 0.06),
        "cgm_mean": (102.0, 13.0), "cgm_std": (16.0, 5.5),
        "symptom_fatigue": (1.5, 0.7), "symptom_mood": (1.5, 0.8), "symptom_pain": (1.2, 0.7),
        "estrogen": (140.0, 55.0), "lh": (5.0, 2.5),
    },
}

CLIP = {
    "hr_mean": (50, 110), "hrv_rmssd": (10, 120), "steps": (500, 20000),
    "sleep_hours": (3.0, 11.0), "sleep_efficiency": (0.55, 0.99),
    "cgm_mean": (70, 180), "cgm_std": (3, 40),
    "symptom_fatigue": (0, 3), "symptom_mood": (0, 3), "symptom_pain": (0, 3),
    "estrogen": (20, 500), "lh": (0.5, 80),
}

CORE_FEATURES = [c for c in FEATURE_COLUMNS if c != "cycle_day"]
HORMONE_FEATURES = ["estrogen", "lh"]


@dataclass
class SynthCycleGenerator:
    phase_stats: dict[str, dict[str, tuple[float, float]]] = field(
        default_factory=lambda: {p: dict(v) for p, v in DEFAULT_PHASE_STATS.items()}
    )
    phase_cov: dict[str, np.ndarray] = field(default_factory=dict)
    feature_order: list[str] = field(default_factory=lambda: list(CORE_FEATURES))
    missing_rate: float = 0.08
    seed: int = SEED
    fitted_from: str = "default_physiology_priors"
    mode: Literal["independent", "correlated"] = "independent"
    include_hormones: bool = False

    def fit_from_dataframe(
        self,
        df: pd.DataFrame,
        mode: Literal["independent", "correlated"] = "correlated",
        include_hormones: bool = False,
    ) -> "SynthCycleGenerator":
        """Fit per-phase means/stds (+ covariance if correlated) from a TRAIN split only."""
        self.mode = mode
        self.include_hormones = include_hormones and all(c in df.columns for c in HORMONE_FEATURES)
        feature_cols = list(CORE_FEATURES)
        if self.include_hormones:
            feature_cols = feature_cols + HORMONE_FEATURES
        self.feature_order = feature_cols

        stats: dict[str, dict[str, tuple[float, float]]] = {}
        covs: dict[str, np.ndarray] = {}
        for phase in PHASE_LABELS:
            sub = df[df["cycle_phase"] == phase]
            stats[phase] = {}
            for col in feature_cols:
                default = DEFAULT_PHASE_STATS[phase].get(col, (0.0, 1.0))
                if sub.empty or col not in sub.columns or sub[col].isna().all():
                    stats[phase][col] = default
                else:
                    mu = float(sub[col].mean())
                    sd = float(sub[col].std(ddof=1) or 1.0)
                    stats[phase][col] = (mu, max(sd, 1e-3))

            if mode == "correlated" and len(sub) >= 10:
                mat = sub[feature_cols].astype(float)
                # fill for cov estimation only
                filled = mat.fillna(mat.mean())
                cov = np.cov(filled.to_numpy().T)
                # ridge for PSD
                cov = cov + np.eye(len(feature_cols)) * 1e-3
                # if any nan, fall back to diagonal
                if not np.isfinite(cov).all():
                    cov = np.diag([stats[phase][c][1] ** 2 for c in feature_cols])
                covs[phase] = cov
            else:
                covs[phase] = np.diag([stats[phase][c][1] ** 2 for c in feature_cols])

        self.phase_stats = stats
        self.phase_cov = covs
        self.fitted_from = f"mcphases_train_stats_{mode}"
        return self

    def _draw_features(self, phase: str, rng: np.random.Generator) -> dict[str, float]:
        cols = self.feature_order
        if self.mode == "correlated" and phase in self.phase_cov:
            mean = np.array([self.phase_stats[phase][c][0] for c in cols])
            cov = self.phase_cov[phase]
            try:
                sample = rng.multivariate_normal(mean, cov, check_valid="ignore")
            except Exception:  # noqa: BLE001
                sample = np.array([rng.normal(*self.phase_stats[phase][c]) for c in cols])
        else:
            sample = np.array([rng.normal(*self.phase_stats[phase][c]) for c in cols])
        out = {}
        for c, v in zip(cols, sample):
            lo, hi = CLIP.get(c, (float(v) - 1, float(v) + 1))
            out[c] = float(np.clip(v, lo, hi))
        return out

    def generate(
        self,
        n_participants: int = N_SYNTH_PARTICIPANTS,
        days: int = DAYS_PER_PARTICIPANT,
        start: date | None = None,
    ) -> pd.DataFrame:
        rng = np.random.default_rng(self.seed)
        start = start or date(2024, 1, 1)
        feature_cols = list(self.feature_order) if self.feature_order else list(CORE_FEATURES)
        # Ensure core schema columns always present even if hormones off
        for c in CORE_FEATURES:
            if c not in feature_cols:
                feature_cols.append(c)

        rows: list[dict[str, Any]] = []
        for i in range(n_participants):
            pid = f"synth_{i:04d}"
            cycle_len = int(np.clip(round(rng.normal(CYCLE_LENGTH_MEAN, CYCLE_LENGTH_STD)), 24, 35))
            cycle_day = int(rng.integers(1, cycle_len + 1))
            offsets = {c: float(rng.normal(0, 0.12)) for c in feature_cols}
            prev: dict[str, float] = {}

            for d in range(days):
                phase = _phase_for_day(cycle_day, cycle_len)
                drawn = self._draw_features(phase, rng)
                row: dict[str, Any] = {
                    "participant_id": pid,
                    "date": (start + timedelta(days=d)).isoformat(),
                    "cycle_day": cycle_day,
                    "cycle_phase": phase,
                    "source": "synthetic",
                    "license_tag": "MIT-synthetic-no-PHI",
                }
                for col in feature_cols:
                    val = drawn.get(col)
                    if val is None:
                        mu, sd = self.phase_stats.get(phase, DEFAULT_PHASE_STATS[phase]).get(
                            col, DEFAULT_PHASE_STATS[phase].get(col, (0.0, 1.0))
                        )
                        val = float(rng.normal(mu, sd))
                    val = val * (1.0 + offsets.get(col, 0.0))
                    if col in prev:
                        val = 0.55 * prev[col] + 0.45 * val
                    lo, hi = CLIP.get(col, (val - 1, val + 1))
                    val = float(np.clip(val, lo, hi))
                    if rng.random() < self.missing_rate:
                        row[col] = np.nan
                    else:
                        row[col] = val
                        prev[col] = val

                # Core schema must not miss required feature keys
                for col in CORE_FEATURES:
                    if col not in row:
                        row[col] = np.nan
                rows.append(row)

                cycle_day += 1
                if cycle_day > cycle_len:
                    cycle_day = 1
                    cycle_len = int(np.clip(round(rng.normal(CYCLE_LENGTH_MEAN, CYCLE_LENGTH_STD)), 24, 35))

        df = pd.DataFrame(rows)
        # Validate only required columns (hormones optional)
        errors = validate_timeline(df)
        if errors:
            raise ValueError(f"synthetic timeline failed validation: {errors}")
        return df

    def manifest(self, df: pd.DataFrame) -> dict[str, Any]:
        return {
            "name": "SynthCycle",
            "mode": self.mode,
            "include_hormones": self.include_hormones,
            "n_participants": int(df["participant_id"].nunique()),
            "n_rows": int(len(df)),
            "days_span": int(df.groupby("participant_id").size().median()),
            "fitted_from": self.fitted_from,
            "missing_rate_target": self.missing_rate,
            "observed_missing_rate": float(
                df[[c for c in CORE_FEATURES if c in df.columns]].isna().mean().mean()
            ),
            "phases": PHASE_LABELS,
            "license": "MIT — synthetic, no PHI",
            "seed": self.seed,
        }
