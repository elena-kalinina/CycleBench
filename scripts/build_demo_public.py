#!/usr/bin/env python3
"""Rebuild outputs/demo_public.json with a pool of SynthCycle people for the live demo.

Keeps headline metrics from latest_metrics.json when present.
Generates a fresh synthetic cohort + GBM so predictions match streams.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cyclebench.baselines.models import fit_baseline
from cyclebench.benchmark.windows import build_windows
from cyclebench.config import OUTPUTS_DIR, SEED, SYNTH_DIR
from cyclebench.preprocess.splits import filter_split, participant_splits
from cyclebench.synthcycle.generator import SynthCycleGenerator


def _records(df, n: int | None = 28) -> list[dict]:
    out = df if n is None else df.head(n)
    return json.loads(out.to_json(orient="records", date_format="iso"))


def main() -> None:
    metrics_path = OUTPUTS_DIR / "latest_metrics.json"
    prior = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    headline = prior.get("headline") or {}
    metrics = {
        "protocol": headline.get("protocol") or prior.get("mode") or "TSTR",
        "headline": headline.get("headline"),
        "balanced_accuracy": headline.get("tstr") or (prior.get("tstr") or {}).get("balanced_accuracy"),
        "naive": headline.get("naive") or (prior.get("naive_real") or {}).get("balanced_accuracy"),
        "trtr": headline.get("trtr") or (prior.get("trtr") or {}).get("balanced_accuracy"),
        "beats_naive": headline.get("beats_naive"),
        "pct_of_trtr": headline.get("pct_of_trtr"),
        "macro_f1": (prior.get("tstr") or prior.get("tsts") or {}).get("macro_f1"),
        "ece": (prior.get("tstr") or prior.get("tsts") or {}).get("ece"),
    }

    # Prefer previously fitted generator stats if a fitted parquet exists; else generate open cohort.
    fitted = SYNTH_DIR / "synthcycle_fitted_from_real_train.parquet"
    if fitted.exists():
        synth = __import__("pandas").read_parquet(fitted)
        fitted_from = prior.get("synth_fitted_from") or "mcphases_train_stats_correlated"
    else:
        gen = SynthCycleGenerator(seed=SEED, mode="correlated")
        # If we have no real fit on disk, still produce an open cohort for the UI.
        synth = gen.generate(n_participants=120, days=60)
        fitted_from = gen.fitted_from  # default_physiology_priors — do not inherit a stale mcPHASES claim
        print(f"note: no fitted parquet at {fitted} — using generated cohort ({fitted_from})")
        print("note: headline metrics still taken from latest_metrics.json when present")

    splits = participant_splits(synth, seed=SEED)
    X_tr, y_tr, _ = build_windows(filter_split(synth, splits["train"]), target="cycle_phase")
    model = fit_baseline(X_tr, y_tr, kind="categorical", seed=SEED)

    demo_pids = list(splits["test"][:12])
    participants = []
    for pid in demo_pids:
        pdf = filter_split(synth, [pid]).copy()
        entry = {"participant_id": pid, "streams": _records(pdf), "predictions": []}
        X_d, y_d, meta = build_windows(pdf, target="cycle_phase")
        if len(X_d):
            meta = meta.copy()
            meta["y_true"] = y_d
            meta["y_pred"] = model.predict(X_d)
            proba = model.predict_proba(X_d)
            meta["confidence"] = proba.max(axis=1) if proba is not None else 0.0
            meta["status"] = "inferred"
            entry["predictions"] = _records(meta, n=None)
        participants.append(entry)

    demo0 = participants[0]
    ablations = []
    abl_path = OUTPUTS_DIR / "ablation.json"
    if abl_path.exists():
        raw = json.loads(abl_path.read_text())
        for row in raw.get("ablations") or []:
            tstr = row.get("tstr") or {}
            ablations.append({
                "name": row.get("ablation"),
                "balanced_accuracy": tstr.get("balanced_accuracy"),
                "macro_f1": tstr.get("macro_f1"),
                "ece": tstr.get("ece"),
            })

    public = {
        "disclaimer": "Research only — not a diagnosis. Demo participants are synthetic (SynthCycle). Restricted clinical rows are not redistributed.",
        "participant_id": demo0["participant_id"],
        "streams": demo0["streams"],
        "predictions": demo0["predictions"],
        "participants": participants,
        "synth_n_participants": int(synth["participant_id"].nunique()),
        "synth_days": int(synth.groupby("participant_id").size().median()),
        "synth_missing_rate": 0.08,
        "metrics": metrics,
        "ablations": ablations,
        "split_real": prior.get("splits_real"),
        "model_version": "baseline.joblib",
        "model_name": "GradientBoostingClassifier",
        "model_meta": "sklearn · depth 3 · 80 trees · median impute",
        "synth_fitted_from": fitted_from,
    }
    out = OUTPUTS_DIR / "demo_public.json"
    out.write_text(json.dumps(public, indent=2, allow_nan=False))
    print(f"wrote {out} with {len(participants)} participants: {demo_pids}")


if __name__ == "__main__":
    main()
