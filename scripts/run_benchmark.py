#!/usr/bin/env python3
"""Run CycleBench: splits → windows → baselines → TSTR / TSTS metrics.

Synthetic-first: always runs end-to-end without mcPHASES.
With --real: fits SynthCycle to the real TRAIN split, trains on that open
synthetic cohort, evaluates on real held-out participants (true TSTR).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cyclebench.baselines.models import fit_baseline, naive_predict
from cyclebench.benchmark.windows import build_windows
from cyclebench.config import (
    DEFAULT_TARGET,
    N_SYNTH_PARTICIPANTS,
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


def _feat_cols(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.startswith(("hr", "hrv", "steps", "sleep", "cgm", "symptom"))]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--synth", type=Path, default=SYNTH_DIR / "synthcycle_v1.parquet")
    ap.add_argument("--real", type=Path, default=None, help="real timeline parquet for true TSTR")
    ap.add_argument("--target", default=DEFAULT_TARGET)
    ap.add_argument("--kind", default=TARGET_KIND, choices=["categorical", "continuous"])
    ap.add_argument("--out", type=Path, default=OUTPUTS_DIR / "latest_metrics.json")
    ap.add_argument("--n-synth", type=int, default=max(N_SYNTH_PARTICIPANTS, 120))
    args = ap.parse_args()

    # --- Always: prior-synth TSTS path (open, reproducible) ---
    synth_prior = _load_or_make_synth(args.synth)
    splits_prior = participant_splits(synth_prior, seed=SEED)
    leak = assert_no_participant_leakage(splits_prior)
    if leak:
        raise RuntimeError(leak)

    X_tr, y_tr, _ = _pack(synth_prior, splits_prior["train"], args.target)
    X_te, y_te, _ = _pack(synth_prior, splits_prior["test"], args.target)
    model_prior = fit_baseline(X_tr, y_tr, kind=args.kind, seed=SEED)
    pred_te = model_prior.predict(X_te)
    proba = model_prior.predict_proba(X_te) if args.kind == "categorical" else None
    labels = list(model_prior.label_encoder.classes_) if args.kind == "categorical" else None
    tsts = evaluate_predictions(y_te, pred_te, args.kind, proba, labels)
    naive_tsts = evaluate_predictions(
        y_te, naive_predict(y_tr, len(y_te), args.kind), args.kind
    )

    tstr = None
    trtr = None
    naive_real = None
    real_splits = None
    synth_train = synth_prior
    model = model_prior
    fitted_from = "default_physiology_priors"

    # --- True TSTR: fit SynthCycle on real TRAIN only → train → test real ---
    if args.real and args.real.exists():
        real = pd.read_parquet(args.real) if args.real.suffix == ".parquet" else pd.read_csv(args.real)
        real_splits = participant_splits(real, seed=SEED)
        leak_r = assert_no_participant_leakage(real_splits)
        if leak_r:
            raise RuntimeError(leak_r)

        real_train = filter_split(real, real_splits["train"])
        gen = SynthCycleGenerator(seed=SEED).fit_from_dataframe(real_train)
        fitted_from = gen.fitted_from
        synth_fitted = gen.generate(n_participants=args.n_synth, days=60)
        fitted_path = SYNTH_DIR / "synthcycle_fitted_from_real_train.parquet"
        synth_fitted.to_parquet(fitted_path, index=False)
        # Manifest is OK to commit (no PHI) — stats only
        (SYNTH_DIR / "synthcycle_fitted_manifest.json").write_text(
            json.dumps(gen.manifest(synth_fitted), indent=2)
        )
        print(f"fitted SynthCycle from real train → {fitted_path}")

        X_str, y_str, _ = _pack(synth_fitted, list(synth_fitted["participant_id"].unique()), args.target)
        # Use all fitted synth for training (it's already privacy-safe); hold out 20% synth for sanity TSTS
        synth_splits = participant_splits(synth_fitted, seed=SEED)
        X_str, y_str, _ = _pack(synth_fitted, synth_splits["train"], args.target)
        model = fit_baseline(X_str, y_str, kind=args.kind, seed=SEED)
        synth_train = synth_fitted

        X_rte, y_rte, _ = _pack(real, real_splits["test"], args.target)
        pred_tstr = model.predict(X_rte)
        proba_tstr = model.predict_proba(X_rte) if args.kind == "categorical" else None
        tstr_labels = list(model.label_encoder.classes_) if args.kind == "categorical" else None
        tstr = evaluate_predictions(y_rte, pred_tstr, args.kind, proba_tstr, tstr_labels)
        naive_real = evaluate_predictions(
            y_rte, naive_predict(y_str, len(y_rte), args.kind), args.kind
        )

        X_rtr, y_rtr, _ = _pack(real, real_splits["train"], args.target)
        real_model = fit_baseline(X_rtr, y_rtr, kind=args.kind, seed=SEED)
        pred_trtr = real_model.predict(X_rte)
        proba_trtr = real_model.predict_proba(X_rte) if args.kind == "categorical" else None
        trtr_labels = list(real_model.label_encoder.classes_) if args.kind == "categorical" else None
        trtr = evaluate_predictions(y_rte, pred_trtr, args.kind, proba_trtr, trtr_labels)

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
        "synth_fitted_from": fitted_from,
        "splits_synth_prior": {k: len(v) for k, v in splits_prior.items()},
        "splits_real": {k: len(v) for k, v in real_splits.items()} if real_splits else None,
        "missingness_synth": missingness_report(synth_train, _feat_cols(synth_train)),
        "tsts": tsts,
        "naive_tsts": naive_tsts,
        "delta_tsts": delta_vs_naive(tsts, naive_tsts),
        "tstr": tstr,
        "trtr": trtr,
        "naive_real": naive_real,
        "delta_tstr": delta_vs_naive(tstr, naive_real) if tstr and naive_real else None,
        "headline": headline,
        "mode": "TSTR" if tstr else "TSTS_synth_only",
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2))

    joblib.dump(model, OUTPUTS_DIR / "baseline.joblib")
    # Public demo: always a SYNTHETIC participant (never real mcp_*)
    demo_pid = participant_splits(synth_train, seed=SEED)["test"][0]
    demo_df = filter_split(synth_train, [demo_pid]).copy()
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

    # Public JSON for Lovable / video (synthetic only + aggregate metrics)
    public = {
        "disclaimer": "Research only — not a diagnosis. Demo participant is synthetic (SynthCycle). Restricted clinical rows are not redistributed.",
        "participant_id": demo_pid,
        "streams": demo_df.head(28).to_dict(orient="records"),
        "metrics": {
            "protocol": protocol,
            "headline": headline.get("headline"),
            "balanced_accuracy": headline.get("tstr"),
            "naive": headline.get("naive"),
            "trtr": headline.get("trtr"),
            "beats_naive": headline.get("beats_naive"),
            "pct_of_trtr": headline.get("pct_of_trtr"),
        },
        "split_real": report["splits_real"],
        "model_version": "baseline.joblib",
        "synth_fitted_from": fitted_from,
    }
    if (OUTPUTS_DIR / "demo_predictions.csv").exists():
        public["predictions"] = pd.read_csv(OUTPUTS_DIR / "demo_predictions.csv").head(28).to_dict(
            orient="records"
        )
    (OUTPUTS_DIR / "demo_public.json").write_text(json.dumps(public, indent=2, default=str))

    print(json.dumps(headline, indent=2))
    print(f"wrote {args.out}")
    print(f"mode={report['mode']}")
    print(f"demo_public={OUTPUTS_DIR / 'demo_public.json'}")


if __name__ == "__main__":
    main()
