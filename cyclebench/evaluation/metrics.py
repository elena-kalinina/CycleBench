"""Target-agnostic metrics + TSTR harness."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
)

TargetKind = Literal["categorical", "continuous"]


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    kind: TargetKind,
    y_proba: np.ndarray | None = None,
    labels: list[str] | None = None,
) -> dict[str, Any]:
    if kind == "categorical":
        y_true_s = y_true.astype(str)
        y_pred_s = y_pred.astype(str)
        out: dict[str, Any] = {
            "kind": "categorical",
            "n": int(len(y_true_s)),
            "macro_f1": float(f1_score(y_true_s, y_pred_s, average="macro", zero_division=0)),
            "balanced_accuracy": float(balanced_accuracy_score(y_true_s, y_pred_s)),
            "accuracy": float(accuracy_score(y_true_s, y_pred_s)),
        }
        if y_proba is not None and labels is not None:
            # simple ECE over max-prob bins
            out["ece"] = float(_expected_calibration_error(y_true_s, y_pred_s, y_proba, labels))
        return out

    y_true_f = y_true.astype(float)
    y_pred_f = y_pred.astype(float)
    return {
        "kind": "continuous",
        "n": int(len(y_true_f)),
        "mae": float(mean_absolute_error(y_true_f, y_pred_f)),
        "rmse": float(np.sqrt(mean_squared_error(y_true_f, y_pred_f))),
    }


def _expected_calibration_error(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    labels: list[str],
    n_bins: int = 10,
) -> float:
    label_to_i = {l: i for i, l in enumerate(labels)}
    conf = y_proba.max(axis=1)
    correct = (y_true == y_pred).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        m = (conf > bins[i]) & (conf <= bins[i + 1])
        if not np.any(m):
            continue
        ece += float(np.abs(correct[m].mean() - conf[m].mean()) * m.mean())
    return ece


def delta_vs_naive(model_metrics: dict[str, Any], naive_metrics: dict[str, Any]) -> dict[str, Any]:
    """Positive = model better. For errors (mae/rmse), flip sign."""
    out: dict[str, Any] = {}
    if model_metrics["kind"] == "categorical":
        for k in ("macro_f1", "balanced_accuracy", "accuracy"):
            out[f"delta_{k}"] = float(model_metrics[k] - naive_metrics[k])
    else:
        for k in ("mae", "rmse"):
            out[f"delta_{k}"] = float(naive_metrics[k] - model_metrics[k])  # higher better
    return out


def tstr_summary(
    tstr: dict[str, Any],
    trtr: dict[str, Any] | None,
    naive: dict[str, Any],
    protocol: str = "TSTR",
) -> dict[str, Any]:
    """Headline numbers for the pitch.

    `protocol` should be ``TSTR`` (train synth → test real) or ``TSTS``
    (synth-only). Never label a synth-only result as TSTR.
    """
    primary = "balanced_accuracy" if tstr.get("kind") == "categorical" else "mae"
    tag = protocol if protocol in {"TSTR", "TSTS"} else "EVAL"
    summary: dict[str, Any] = {
        "primary_metric": primary,
        "protocol": tag,
        "tstr": tstr.get(primary),
        "naive": naive.get(primary),
        "trtr": trtr.get(primary) if trtr else None,
        "beats_naive": None,
        "pct_of_trtr": None,
        "headline": None,
    }
    if tstr.get(primary) is None or naive.get(primary) is None:
        return summary

    if primary == "balanced_accuracy":
        delta = float(tstr[primary] - naive[primary])
        summary["beats_naive"] = delta
        if trtr and trtr.get(primary) is not None and trtr[primary] > 0:
            summary["pct_of_trtr"] = float(tstr[primary] / trtr[primary])
        summary["headline"] = (
            f"{tag} balanced_accuracy={tstr[primary]:.3f} "
            f"(+{delta:.3f} vs naive"
            + (f", {summary['pct_of_trtr']*100:.0f}% of TRTR" if summary["pct_of_trtr"] else "")
            + ")"
        )
    else:
        # lower better
        delta = float(naive[primary] - tstr[primary])
        summary["beats_naive"] = delta
        if trtr and trtr.get(primary) is not None and trtr[primary] > 0:
            summary["pct_of_trtr"] = float(trtr[primary] / tstr[primary])  # >1 means TSTR worse
        summary["headline"] = (
            f"{tag} mae={tstr[primary]:.3f} "
            f"({delta:+.3f} vs naive; lower MAE better)"
        )
    return summary
