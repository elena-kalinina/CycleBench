"""mcPHASES → CycleBench daily timeline adapter.

Maps PhysioNet mcPHASES CSVs into the canonical daily schema.
Raw / derived patient rows are LOCAL ONLY (PhysioNet Restricted License) —
never commit or redistribute. The open release is SynthCycle + code + metrics.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from cyclebench.config import PHASE_LABELS, PROCESSED_DIR, RAW_DIR
from cyclebench.data_contract.schema import validate_timeline

JOIN_KEYS = ["id", "study_interval", "day_in_study"]

PHASE_MAP = {
    "menstrual": "menstrual",
    "follicular": "follicular",
    "fertility": "ovulatory",  # mcPHASES "Fertility" ≈ peri-ovulatory window
    "luteal": "luteal",
}

LIKERT_MAP = {
    "not at all": 0.0,
    "very low/little": 0.5,
    "low": 1.0,
    "moderate": 1.5,
    "high": 2.5,
    "very high": 3.0,
}

# mmol/L → mg/dL (schema / SynthCycle priors use mg/dL)
MMOL_TO_MGDL = 18.0182


def default_mcphases_root() -> Path:
    return RAW_DIR / "mcphases"


def available(root: Path | None = None) -> bool:
    root = root or default_mcphases_root()
    if not root.exists():
        return False
    return (root / "hormones_and_selfreport.csv").exists()


def inspect(root: Path | None = None) -> dict[str, Any]:
    root = root or default_mcphases_root()
    files = []
    if root.exists():
        for p in sorted(root.rglob("*")):
            if p.is_file() and p.suffix.lower() in {".csv", ".parquet", ".tsv", ".txt"}:
                files.append(str(p.relative_to(root)))
    return {
        "root": str(root),
        "available": available(root),
        "files": files[:50],
        "n_files": len(files),
        "license": "PhysioNet Restricted — do not redistribute derived patient rows",
        "note": "Run scripts/adapt_mcphases.py → data/processed/mcphases_daily.parquet (gitignored).",
    }


def _likert(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().map(LIKERT_MAP)


def _map_phase(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().map(PHASE_MAP)


def _daily_mean(path: Path, value_col: str, out_name: str, chunksize: int | None = None) -> pd.DataFrame:
    usecols = JOIN_KEYS + [value_col]
    if chunksize:
        parts = []
        for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize):
            parts.append(chunk.groupby(JOIN_KEYS, as_index=False)[value_col].mean())
        g = pd.concat(parts, ignore_index=True)
        g = g.groupby(JOIN_KEYS, as_index=False)[value_col].mean()
    else:
        df = pd.read_csv(path, usecols=usecols)
        g = df.groupby(JOIN_KEYS, as_index=False)[value_col].mean()
    return g.rename(columns={value_col: out_name})


def _daily_sum(path: Path, value_col: str, out_name: str, chunksize: int = 500_000) -> pd.DataFrame:
    usecols = JOIN_KEYS + [value_col]
    parts = []
    for chunk in pd.read_csv(path, usecols=usecols, chunksize=chunksize):
        parts.append(chunk.groupby(JOIN_KEYS, as_index=False)[value_col].sum())
    g = pd.concat(parts, ignore_index=True)
    g = g.groupby(JOIN_KEYS, as_index=False)[value_col].sum()
    return g.rename(columns={value_col: out_name})


def _daily_glucose(path: Path) -> pd.DataFrame:
    usecols = JOIN_KEYS + ["glucose_value"]
    df = pd.read_csv(path, usecols=usecols)
    df["glucose_mgdl"] = df["glucose_value"] * MMOL_TO_MGDL
    g = df.groupby(JOIN_KEYS, as_index=False).agg(
        cgm_mean=("glucose_mgdl", "mean"),
        cgm_std=("glucose_mgdl", "std"),
    )
    return g


def _daily_sleep(path: Path) -> pd.DataFrame:
    """Main-sleep only; hours + efficiency. Avoid loading giant `levels` blob."""
    usecols = [
        "id",
        "study_interval",
        "sleep_end_day_in_study",
        "minutesasleep",
        "efficiency",
        "mainsleep",
    ]
    df = pd.read_csv(path, usecols=usecols)
    if "mainsleep" in df.columns:
        df = df[df["mainsleep"] == True]  # noqa: E712
    df = df.rename(columns={"sleep_end_day_in_study": "day_in_study"})
    df["sleep_hours"] = df["minutesasleep"] / 60.0
    df["sleep_efficiency"] = df["efficiency"] / 100.0
    g = df.groupby(JOIN_KEYS, as_index=False).agg(
        sleep_hours=("sleep_hours", "sum"),
        sleep_efficiency=("sleep_efficiency", "mean"),
    )
    return g


def _cycle_day_from_phase(phases: pd.Series) -> pd.Series:
    """Ordinal day-within-phase-run — not calendar PHI. Used as a weak feature;
    excluded from features when target is cycle_phase (leakage guard in windows.py).
    """
    out = []
    prev = None
    day = 0
    for p in phases:
        if pd.isna(p) or p != prev:
            day = 1
            prev = p
        else:
            day += 1
        out.append(day)
    return pd.Series(out, index=phases.index)


def adapt(root: Path | None = None) -> pd.DataFrame:
    """Map mcPHASES → CycleBench daily timeline (local evaluation only)."""
    root = root or default_mcphases_root()
    if not available(root):
        raise FileNotFoundError(
            f"mcPHASES not found at {root}. Symlink PhysioNet extract to data/raw/mcphases."
        )

    hormones = pd.read_csv(root / "hormones_and_selfreport.csv")
    spine_cols = JOIN_KEYS + ["phase", "fatigue", "moodswing", "cramps", "estrogen", "lh"]
    spine = hormones[spine_cols].copy()
    spine["cycle_phase"] = _map_phase(spine["phase"])
    spine = spine.dropna(subset=["cycle_phase"])
    spine = spine[spine["cycle_phase"].isin(PHASE_LABELS)]

    spine["symptom_fatigue"] = _likert(spine["fatigue"])
    spine["symptom_mood"] = _likert(spine["moodswing"])
    spine["symptom_pain"] = _likert(spine["cramps"])
    # Optional research targets (not in required FEATURE_COLUMNS; continuous reconstruction)
    spine["estrogen"] = pd.to_numeric(spine["estrogen"], errors="coerce")
    spine["lh"] = pd.to_numeric(spine["lh"], errors="coerce")

    # Wearable / CGM daily aggs (lean set — skip 2GB heart_rate.csv)
    hr = _daily_mean(root / "resting_heart_rate.csv", "value", "hr_mean")
    hrv = _daily_mean(root / "heart_rate_variability_details.csv", "rmssd", "hrv_rmssd")
    steps = _daily_sum(root / "steps.csv", "steps", "steps")
    cgm = _daily_glucose(root / "glucose.csv")
    sleep = _daily_sleep(root / "sleep.csv")

    df = spine.merge(hr, on=JOIN_KEYS, how="left")
    df = df.merge(hrv, on=JOIN_KEYS, how="left")
    df = df.merge(steps, on=JOIN_KEYS, how="left")
    df = df.merge(cgm, on=JOIN_KEYS, how="left")
    df = df.merge(sleep, on=JOIN_KEYS, how="left")

    # Stable synthetic calendar per (id, interval) — ordinal study day only
    df = df.sort_values(JOIN_KEYS).reset_index(drop=True)
    # Offset intervals so dates don't collide across intervals for same id
    interval_offset = df["study_interval"].astype(str).map(
        {str(k): i * 400 for i, k in enumerate(sorted(df["study_interval"].astype(str).unique()))}
    )
    ordinal = interval_offset + df["day_in_study"].astype(int)
    # Anchor at 2022-01-01 (study year); not a claim of real calendar date
    df["date"] = (pd.Timestamp("2022-01-01") + pd.to_timedelta(ordinal - 1, unit="D")).dt.strftime(
        "%Y-%m-%d"
    )

    df["participant_id"] = "mcp_" + df["id"].astype(str)
    df["cycle_day"] = (
        df.groupby(["participant_id", "study_interval"], group_keys=False)["cycle_phase"]
        .apply(_cycle_day_from_phase)
        .astype(int)
    )
    df["source"] = "mcphases"
    df["license_tag"] = "PhysioNet-Restricted-local-eval-only"

    out = df[
        [
            "participant_id",
            "date",
            "cycle_day",
            "cycle_phase",
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
            "estrogen",
            "lh",
            "source",
            "license_tag",
        ]
    ].copy()

    # One row per participant-day if intervals somehow collide (shouldn't)
    out = out.drop_duplicates(subset=["participant_id", "date"], keep="first")

    errors = validate_timeline(out)
    if errors:
        raise ValueError(f"adapted timeline failed validation: {errors}")
    return out.reset_index(drop=True)


def adapt_and_save(
    root: Path | None = None,
    out_path: Path | None = None,
) -> tuple[pd.DataFrame, Path]:
    out_path = out_path or (PROCESSED_DIR / "mcphases_daily.parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = adapt(root=root)
    df.to_parquet(out_path, index=False)
    # CSV for local debugging only — still gitignored under data/processed/
    df.to_csv(out_path.with_suffix(".csv"), index=False)
    # Provenance manifest stays LOCAL with the restricted timeline (gitignored).
    from cyclebench.provenance import manifest_from_timeline

    info = inspect(root)
    manifest_from_timeline(
        df,
        source_name="mcPHASES",
        adapter="cyclebench.adapters.mcphases",
        path=out_path.with_name("mcphases_daily_provenance.json"),
        redistributable=False,
        extra={
            "upstream_license": info.get("license"),
            "source_root": info.get("root"),
            "n_source_files": info.get("n_files"),
        },
    )
    return df, out_path


def try_adapt() -> tuple[pd.DataFrame | None, dict[str, Any]]:
    info = inspect()
    if not info["available"]:
        return None, info
    try:
        df = adapt()
        info["validation_errors"] = []
        info["n_rows"] = int(len(df))
        info["n_participants"] = int(df["participant_id"].nunique())
        return df, info
    except Exception as e:  # noqa: BLE001
        info["status"] = str(e)
        return None, info
