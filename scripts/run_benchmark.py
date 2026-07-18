#!/usr/bin/env python3
"""Run CycleBench: splits → windows → baselines → TSTR / TSTS metrics.

Synthetic-first: always runs end-to-end without mcPHASES.
When real data is adapted later, pass --real path to enable true TSTR.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cyclebench.baselines.models import fit_baseline, naive_predict
from cyclebench.benchmark.windows import build_windows
from cyclebench.config import (
    DEFAULT_TARGET,
    OUTPUTS_DIR,
    SEED,
    SYNTH_DIR,
    TARGET_KIND,
)
from cyclebench.evaluation.metrics import delta_vs_naive, evaluate_predictions, tstr_summary
from cyclebench.preprocess.splits import (
    assert_no_participant_leakage,
    filter_split,
    missingness_report,
    participant_splits,
)
from cyclebench.synthcycle.generator import SynthCycleGenerator


def _load_or_make_synth(path: Path) -> pd.DataFrame:
    if path.exists():
        if path.suffix == ".parquet":
            return pd.read_parquet(path)
        return pd.read_csv(path)
    print(f"no synth at {path} — generating…")
    gen = SynthCycleGenerator()
    df = gen.generate()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return df


def _pack(df: pd.DataFrame, pids: list[str], target: str):
    sub = filter_split(df, pids)
    return build_windows(sub, target=target)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--synth", type=Path, default=SYNTH_DIR / "synthcycle_v1.parquet")
    ap.add_argument("--real", type=Path, default=None, help="optional real timeline parquet for true TSTR")
    ap.add_argument("--target", default=DEFAULT_TARGET)
    ap.add_argument("--kind", default=TARGET_KIND, choices=["categorical", "continuous"])
    ap.add_argument("--out", type=Path, default=OUTPUTS_DIR / "latest_metrics.json")
    args = ap.parse_args()

    synth = _load_or_make_synth(args.synth)
    splits = participant_splits(synth, seed=SEED)
    leak = assert_no_participant_leakage(splits)
    if leak:
        raise RuntimeError(leak)

    X_tr, y_tr, _ = _pack(synth, splits["train"], args.target)
    X_te, y_te, meta_te = _pack(synth, splits["test"], args.target)

    # --- Train on synthetic ---
    model = fit_baseline(X_tr, y_tr, kind=args.kind, seed=SEED)
    pred_te = model.predict(X_te)
    proba = None
    labels = None
    if args.kind == "categorical":
        proba = model.predict_proba(X_te)
        labels = list(model.label_encoder.classes_)

    # TSTS = train-on-synth / test-on-synth (fallback when no real data)
    tsts = evaluate_predictions(y_te, pred_te, args.kind, proba, labels)
    naive_pred = naive_predict(y_tr, len(y_te), args.kind)
    naive_tsts = evaluate_predictions(y_te, naive_pred, args.kind)

    # --- Optional true TSTR (train synth → test real) ---
    tstr = None
    trtr = None
    naive_real = None
    if args.real and args.real.exists():
        real = pd.read_parquet(args.real) if args.real.suffix == ".parquet" else pd.read_csv(args.real)
        real_splits = participant_splits(real, seed=SEED)
        X_rtr, y_rtr, _ = _pack(real, real_splits["train"], args.target)
        X_rte, y_rte, _ = _pack(real, real_splits["test"], args.target)

        pred_tstr = model.predict(X_rte)
        proba_tstr = model.predict_proba(X_rte) if args.kind == "categorical" else None
        tstr = evaluate_predictions(y_rte, pred_tstr, args.kind, proba_tstr, labels)
        naive_real = evaluate_predictions(
            y_rte, naive_predict(y_tr, len(y_rte), args.kind), args.kind
        )
        # TRTR ceiling
        real_model = fit_baseline(X_rtr, y_rtr, kind=args.kind, seed=SEED)
        pred_trtr = real_model.predict(X_rte)
        proba_trtr = real_model.predict_proba(X_rte) if args.kind == "categorical" else None
        trtr_labels = list(real_model.label_encoder.classes_) if args.kind == "categorical" else None
        trtr = evaluate_predictions(y_rte, pred_trtr, args.kind, proba_trtr, trtr_labels)

    # Headline: prefer true TSTR; else TSTS framed honestly (never mislabel)
    primary_model = tstr or tsts
    primary_naive = naive_real or naive_tsts
    protocol = "TSTR" if tstr else "TSTS"
    headline = tstr_summary(primary_model, trtr, primary_naive, protocol=protocol)

    report = {
        "task": {
            "name": "masked_multimodal_reconstruction",
            "target": args.target,
            "kind": args.kind,
            "note": "Research-only. Not a diagnosis.",
        },
        "splits": {k: len(v) for k, v in splits.items()},
        "missingness_synth": missingness_report(
            synth, [c for c in synth.columns if c.startswith(("hr", "hrv", "steps", "sleep", "cgm", "symptom"))]
        ),
        "tsts": tsts,
        "naive_tsts": naive_tsts,
        "delta_tsts": delta_vs_naive(tsts, naive_tsts),
        "tstr": tstr,
        "trtr": trtr,
        "naive_real": naive_real,
        "headline": headline,
        "mode": "TSTR" if tstr else "TSTS_synth_only",
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))

    # Persist model + a demo trajectory slice
    joblib.dump(model, OUTPUTS_DIR / "baseline.joblib")
    demo_pid = splits["test"][0]
    demo_df = filter_split(synth, [demo_pid]).copy()
    # attach naive inferred phase for demo when target is cycle_phase
    if args.target == "cycle_phase":
        X_demo, y_demo, meta_demo = build_windows(demo_df, target=args.target)
        if len(X_demo):
            pred_demo = model.predict(X_demo)
            meta_demo = meta_demo.copy()
            meta_demo["y_true"] = y_demo
            meta_demo["y_pred"] = pred_demo
            meta_demo["status"] = "inferred"
            meta_demo.to_csv(OUTPUTS_DIR / "demo_predictions.csv", index=False)
    demo_df.to_csv(OUTPUTS_DIR / "demo_timeline.csv", index=False)

    print(json.dumps(headline, indent=2))
    print(f"wrote {args.out}")
    print(f"mode={report['mode']}")


if __name__ == "__main__":
    main()
