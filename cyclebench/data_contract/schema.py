"""Canonical daily-timeline schema for CycleBench.

Every adapter (mcPHASES today, future sources later) must emit rows that
validate against this contract. The synthetic generator also emits this schema
so train/eval paths are identical.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

import pandas as pd

from cyclebench.config import FEATURE_COLUMNS, PHASE_LABELS


REQUIRED_COLUMNS = [
    "participant_id",
    "date",
    "cycle_day",
    "cycle_phase",
    *FEATURE_COLUMNS[:-1],  # cycle_day already listed
    "source",  # "mcphases" | "synthetic" | ...
    "license_tag",
]


@dataclass
class DailyRow:
    participant_id: str
    date: str  # ISO YYYY-MM-DD
    cycle_day: int
    cycle_phase: str
    hr_mean: float
    hrv_rmssd: float
    steps: float
    sleep_hours: float
    sleep_efficiency: float
    cgm_mean: float
    cgm_std: float
    symptom_fatigue: float
    symptom_mood: float
    symptom_pain: float
    source: str
    license_tag: str


def validate_timeline(df: pd.DataFrame) -> list[str]:
    """Return a list of validation errors (empty = ok)."""
    errors: list[str] = []
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        errors.append(f"missing columns: {missing}")
        return errors
    if df["participant_id"].isna().any():
        errors.append("null participant_id")
    bad_phase = ~df["cycle_phase"].isin(PHASE_LABELS) & df["cycle_phase"].notna()
    if bad_phase.any():
        errors.append(f"unknown cycle_phase values: {df.loc[bad_phase, 'cycle_phase'].unique().tolist()}")
    if df.duplicated(subset=["participant_id", "date"]).any():
        errors.append("duplicate (participant_id, date) rows")
    return errors


def empty_timeline() -> pd.DataFrame:
    return pd.DataFrame(columns=REQUIRED_COLUMNS)


def schema_dict() -> dict[str, Any]:
    return {
        "name": "cyclebench_daily_v1",
        "grain": "one row per participant per calendar day",
        "required_columns": REQUIRED_COLUMNS,
        "feature_columns": FEATURE_COLUMNS,
        "phase_labels": PHASE_LABELS,
        "fields": {f.name: str(f.type) for f in fields(DailyRow)},
    }
