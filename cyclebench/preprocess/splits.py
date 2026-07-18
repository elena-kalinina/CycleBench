"""Participant-safe, time-aware splits + timeline helpers."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from cyclebench.config import SEED, TEST_FRAC, TRAIN_FRAC, VAL_FRAC


def participant_splits(
    df: pd.DataFrame,
    train_frac: float = TRAIN_FRAC,
    val_frac: float = VAL_FRAC,
    test_frac: float = TEST_FRAC,
    seed: int = SEED,
) -> dict[str, list[str]]:
    """Split by participant_id — never leak a person across folds."""
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6
    rng = np.random.default_rng(seed)
    pids = np.array(sorted(df["participant_id"].unique()))
    rng.shuffle(pids)
    n = len(pids)
    n_train = max(1, int(round(n * train_frac)))
    n_val = max(1, int(round(n * val_frac))) if n >= 5 else 0
    # remainder to test
    train = pids[:n_train].tolist()
    val = pids[n_train : n_train + n_val].tolist()
    test = pids[n_train + n_val :].tolist()
    if not test:
        # tiny-n fallback: peel one from train
        test = [train.pop()]
    return {"train": train, "val": val, "test": test}


def filter_split(df: pd.DataFrame, pids: list[str]) -> pd.DataFrame:
    return df[df["participant_id"].isin(pids)].copy()


def assert_no_participant_leakage(splits: dict[str, list[str]]) -> list[str]:
    errors: list[str] = []
    sets = {k: set(v) for k, v in splits.items()}
    for a in sets:
        for b in sets:
            if a >= b:
                continue
            overlap = sets[a] & sets[b]
            if overlap:
                errors.append(f"participant leakage {a}∩{b}: {sorted(overlap)[:5]}")
    return errors


def missingness_report(df: pd.DataFrame, cols: list[str]) -> dict[str, Any]:
    rates = {c: float(df[c].isna().mean()) for c in cols if c in df.columns}
    return {
        "n_rows": int(len(df)),
        "n_participants": int(df["participant_id"].nunique()),
        "per_column_missing_rate": rates,
        "mean_missing_rate": float(np.mean(list(rates.values()))) if rates else 0.0,
    }
