#!/usr/bin/env python3
"""Modality ablation + estrogen continuous reconstruction (local real data).

Writes aggregate metrics only (no patient rows):
  outputs/ablation.json
  outputs/metrics_estrogen.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cyclebench.baselines.models import fit_baseline, naive_predict
from cyclebench.benchmark.windows import build_windows
from cyclebench.config import FEATURE_COLUMNS, OUTPUTS_DIR, PROCESSED_DIR, SEED, SYNTH_DIR
from cyclebench.evaluation.metrics import evaluate_predictions, tstr_summary
from cyclebench.preprocess.splits import filter_split, participant_splits
from cyclebench.synthcycle.generator import SynthCycleGenerator

MODALITY_GROUPS = {
    "full": [c for c in FEATURE_COLUMNS if c != "cycle_day"],
    "no_cgm": [c for c in FEATURE_COLUMNS if c != "cycle_day" and not c.startswith("cgm")],
    "no_sleep": [c for c in FEATURE_COLUMNS if c != "cycle_day" and not c.startswith("sleep")],
    "no_symptoms": [c for c in FEATURE_COLUMNS if c != "cycle_day" and not c.startswith("symptom")],
    "no_hrv": [c for c in FEATURE_COLUMNS if c != "cycle_day" and c != "hrv_rmssd"],
    "wearable_only": ["hr_mean", "hrv_rmssd", "steps", "sleep_hours", "sleep_efficiency"],
}


def _eval_phase(real: pd.DataFrame, synth: pd.DataFrame, feature_cols: list[str], label: str) -> dict:
    real_splits = participant_splits(real, seed=SEED)
    synth_splits = participant_splits(synth, seed=SEED)
    X_tr, y_tr, _ = build_windows(
        filter_split(synth, synth_splits["train"]), target="cycle_phase", feature_cols=feature_cols
    )
    X_te, y_te, _ = build_windows(
        filter_split(real, real_splits["test"]), target="cycle_phase", feature_cols=feature_cols
    )
    model = fit_baseline(X_tr, y_tr, kind="categorical", seed=SEED)
    pred = model.predict(X_te)
    labels = list(model.label_encoder.classes_)
    proba = model.predict_proba(X_te)
    tstr = evaluate_predictions(y_te, pred, "categorical", proba, labels)
    naive = evaluate_predictions(y_te, naive_predict(y_tr, len(y_te), "categorical"), "categorical")

    # TRTR with same feature set
    X_rtr, y_rtr, _ = build_windows(
        filter_split(real, real_splits["train"]), target="cycle_phase", feature_cols=feature_cols
    )
    real_model = fit_baseline(X_rtr, y_rtr, kind="categorical", seed=SEED)
    trtr = evaluate_predictions(
        y_te,
        real_model.predict(X_te),
        "categorical",
        real_model.predict_proba(X_te),
        list(real_model.label_encoder.classes_),
    )
    headline = tstr_summary(tstr, trtr, naive, protocol="TSTR")
    return {"ablation": label, "features": feature_cols, "tstr": tstr, "trtr": trtr, "naive": naive, "headline": headline}


def _eval_estrogen(real: pd.DataFrame, synth: pd.DataFrame) -> dict:
    if "estrogen" not in real.columns:
        return {"error": "estrogen column missing — re-run adapt_mcphases.py"}
    # Ensure synth has estrogen (regenerate with hormones if needed)
    if "estrogen" not in synth.columns:
        real_splits = participant_splits(real, seed=SEED)
        gen = SynthCycleGenerator(seed=SEED).fit_from_dataframe(
            filter_split(real, real_splits["train"]), mode="correlated", include_hormones=True
        )
        synth = gen.generate(n_participants=120, days=60)
        synth.to_parquet(SYNTH_DIR / "synthcycle_with_hormones.parquet", index=False)

    real_splits = participant_splits(real, seed=SEED)
    synth_splits = participant_splits(synth, seed=SEED)
    feats = [c for c in FEATURE_COLUMNS if c != "cycle_day"]
    # drop rows without estrogen target
    real_te = filter_split(real, real_splits["test"]).dropna(subset=["estrogen"])
    real_tr = filter_split(real, real_splits["train"]).dropna(subset=["estrogen"])
    synth_tr = filter_split(synth, synth_splits["train"]).dropna(subset=["estrogen"])
    if len(real_te) < 50 or len(synth_tr) < 50:
        return {"error": "insufficient estrogen density", "n_real_te": len(real_te), "n_synth_tr": len(synth_tr)}

    X_tr, y_tr, _ = build_windows(synth_tr, target="estrogen", feature_cols=feats)
    X_te, y_te, _ = build_windows(real_te, target="estrogen", feature_cols=feats)
    model = fit_baseline(X_tr, y_tr, kind="continuous", seed=SEED)
    tstr = evaluate_predictions(y_te, model.predict(X_te), "continuous")
    naive = evaluate_predictions(y_te, naive_predict(y_tr, len(y_te), "continuous"), "continuous")
    X_rtr, y_rtr, _ = build_windows(real_tr, target="estrogen", feature_cols=feats)
    real_model = fit_baseline(X_rtr, y_rtr, kind="continuous", seed=SEED)
    trtr = evaluate_predictions(y_te, real_model.predict(X_te), "continuous")
    headline = tstr_summary(tstr, trtr, naive, protocol="TSTR")
    return {
        "task": "estrogen_reconstruction",
        "note": "Research-only continuous hormone reconstruction. Not a diagnosis.",
        "tstr": tstr,
        "trtr": trtr,
        "naive": naive,
        "headline": headline,
        "n_test_windows": int(len(y_te)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--real", type=Path, default=PROCESSED_DIR / "mcphases_daily.parquet")
    ap.add_argument("--synth-mode", choices=["independent", "correlated"], default="correlated")
    args = ap.parse_args()

    if not args.real.exists():
        raise SystemExit(f"missing {args.real} — run scripts/adapt_mcphases.py first")

    real = pd.read_parquet(args.real)
    real_splits = participant_splits(real, seed=SEED)
    gen = SynthCycleGenerator(seed=SEED).fit_from_dataframe(
        filter_split(real, real_splits["train"]),
        mode=args.synth_mode,
        include_hormones=True,
    )
    synth = gen.generate(n_participants=120, days=60)
    synth_path = SYNTH_DIR / f"synthcycle_{args.synth_mode}.parquet"
    synth.to_parquet(synth_path, index=False)
    (SYNTH_DIR / f"synthcycle_{args.synth_mode}_manifest.json").write_text(
        json.dumps(gen.manifest(synth), indent=2)
    )

    ablations = []
    for name, feats in MODALITY_GROUPS.items():
        print(f"ablation: {name}…")
        ablations.append(_eval_phase(real, synth, feats, name))

    estrogen = _eval_estrogen(real, synth)

    out = {
        "synth_mode": args.synth_mode,
        "synth_manifest": gen.manifest(synth),
        "ablations": ablations,
        "estrogen": estrogen,
    }
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUTS_DIR / "ablation.json").write_text(json.dumps(out, indent=2, default=str))
    (OUTPUTS_DIR / "metrics_estrogen.json").write_text(
        json.dumps(estrogen, indent=2, default=str)
    )

    print("\n=== Ablation headlines ===")
    for a in ablations:
        h = a["headline"]
        print(f"{a['ablation']:14s}  {h.get('headline')}")
    print("\n=== Estrogen ===")
    print(json.dumps(estrogen.get("headline") or estrogen, indent=2))
    print(f"wrote {OUTPUTS_DIR / 'ablation.json'}")


if __name__ == "__main__":
    main()
