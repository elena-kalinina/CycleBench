"""Simple reproducible baselines — LOCF/mean floor + sklearn GBM/logistic."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder


TargetKind = Literal["categorical", "continuous"]


@dataclass
class FittedBaseline:
    kind: TargetKind
    pipeline: Any
    label_encoder: LabelEncoder | None = None
    name: str = "gbm"

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.kind == "categorical":
            pred = self.pipeline.predict(X)
            assert self.label_encoder is not None
            return self.label_encoder.inverse_transform(pred)
        return self.pipeline.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray | None:
        if self.kind != "categorical":
            return None
        if hasattr(self.pipeline, "predict_proba"):
            return self.pipeline.predict_proba(X)
        # Pipeline last step
        return self.pipeline.named_steps["model"].predict_proba(
            self.pipeline.named_steps["imputer"].transform(X)
        )


def fit_baseline(X: np.ndarray, y: np.ndarray, kind: TargetKind, seed: int = 42) -> FittedBaseline:
    imputer = SimpleImputer(strategy="median")
    if kind == "categorical":
        le = LabelEncoder()
        y_enc = le.fit_transform(y.astype(str))
        model = GradientBoostingClassifier(random_state=seed, max_depth=3, n_estimators=80)
        pipe = Pipeline([("imputer", imputer), ("model", model)])
        pipe.fit(X, y_enc)
        return FittedBaseline(kind=kind, pipeline=pipe, label_encoder=le, name="gbm_clf")
    model = GradientBoostingRegressor(random_state=seed, max_depth=3, n_estimators=80)
    pipe = Pipeline([("imputer", imputer), ("model", model)])
    pipe.fit(X, y.astype(float))
    return FittedBaseline(kind=kind, pipeline=pipe, label_encoder=None, name="gbm_reg")


def naive_predict(y_train: np.ndarray, n: int, kind: TargetKind) -> np.ndarray:
    """Population-mode / population-mean floor (must beat this)."""
    if kind == "categorical":
        # mode
        vals, counts = np.unique(y_train.astype(str), return_counts=True)
        mode = vals[int(np.argmax(counts))]
        return np.array([mode] * n)
    mu = float(np.nanmean(y_train.astype(float)))
    return np.full(n, mu)
