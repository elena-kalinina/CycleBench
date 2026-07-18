"""Windowed masked-reconstruction task builder (target-agnostic)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cyclebench.config import FEATURE_COLUMNS, WINDOW_DAYS


def build_windows(
    df: pd.DataFrame,
    target: str,
    feature_cols: list[str] | None = None,
    window_days: int = WINDOW_DAYS,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Build trailing windows.

    For each day t (with enough history), features = flattened values of
    `feature_cols` over [t-window+1, t], with the *current-day target*
    masked out of the feature vector (set to NaN then filled with 0 after
    a missingness indicator is appended... simpler: exclude target from
    features entirely; predict target at day t from other channels + history.

    Returns X, y, meta (participant_id, date).
    """
    feature_cols = feature_cols or [c for c in FEATURE_COLUMNS if c != target]
    # ensure target not in features (leakage guard for phase/cycle_day cases)
    feature_cols = [c for c in feature_cols if c != target]
    if target == "cycle_phase":
        feature_cols = [c for c in feature_cols if c != "cycle_day"]

    df = df.sort_values(["participant_id", "date"]).reset_index(drop=True)
    Xs: list[np.ndarray] = []
    ys: list[Any] = []
    metas: list[dict[str, Any]] = []

    for pid, g in df.groupby("participant_id", sort=False):
        g = g.reset_index(drop=True)
        feats = g[feature_cols].to_numpy(dtype=float)
        # simple mean-impute within participant for window building
        col_means = np.nanmean(feats, axis=0)
        inds = np.where(np.isnan(feats))
        feats[inds] = np.take(col_means, inds[1])
        feats = np.nan_to_num(feats, nan=0.0)

        targets = g[target].to_numpy()
        dates = g["date"].to_numpy()

        for t in range(window_days - 1, len(g)):
            window = feats[t - window_days + 1 : t + 1].reshape(-1)
            y = targets[t]
            if pd.isna(y):
                continue
            Xs.append(window)
            ys.append(y)
            metas.append({"participant_id": pid, "date": dates[t], "target": target})

    if not Xs:
        raise ValueError("no windows built — check data density / window size")

    X = np.vstack(Xs)
    y = np.array(ys)
    meta = pd.DataFrame(metas)
    return X, y, meta


def feature_names(target: str, feature_cols: list[str] | None = None, window_days: int = WINDOW_DAYS) -> list[str]:
    feature_cols = feature_cols or [c for c in FEATURE_COLUMNS if c != target]
    feature_cols = [c for c in feature_cols if c != target]
    if target == "cycle_phase":
        feature_cols = [c for c in feature_cols if c != "cycle_day"]
    names: list[str] = []
    for d in range(window_days):
        for c in feature_cols:
            names.append(f"t-{window_days - 1 - d}:{c}")
    return names
