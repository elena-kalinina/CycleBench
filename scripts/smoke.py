#!/usr/bin/env python3
"""End-to-end smoke: schema → synth → splits → windows → baseline → metrics.

Must stay green without mcPHASES or API keys. This is the §10 green checkpoint.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> int:
    from cyclebench.adapters import mcphases
    from cyclebench.baselines.models import fit_baseline, naive_predict
    from cyclebench.benchmark.windows import build_windows
    from cyclebench.config import DEFAULT_TARGET, OUTPUTS_DIR, SEED, TARGET_KIND
    from cyclebench.data_contract.schema import schema_dict, validate_timeline
    from cyclebench.evaluation.metrics import evaluate_predictions, tstr_summary
    from cyclebench.ingest.symptoms import structure_symptoms
    from cyclebench.preprocess.splits import assert_no_participant_leakage, filter_split, participant_splits
    from cyclebench.synthcycle.generator import SynthCycleGenerator

    checks: list[tuple[str, bool, str]] = []

    # 1. schema
    sch = schema_dict()
    checks.append(("schema", "name" in sch, sch["name"]))

    # 2. synth (small)
    gen = SynthCycleGenerator(seed=SEED)
    df = gen.generate(n_participants=12, days=28)
    errs = validate_timeline(df)
    checks.append(("synth_validate", not errs, str(errs) if errs else f"{len(df)} rows"))

    # 3. splits / leakage
    splits = participant_splits(df, seed=SEED)
    leak = assert_no_participant_leakage(splits)
    checks.append(("no_leakage", not leak, str(leak) if leak else "ok"))

    # 4. windows + baseline
    X_tr, y_tr, _ = build_windows(filter_split(df, splits["train"]), target=DEFAULT_TARGET)
    X_te, y_te, _ = build_windows(filter_split(df, splits["test"]), target=DEFAULT_TARGET)
    model = fit_baseline(X_tr, y_tr, kind=TARGET_KIND, seed=SEED)
    pred = model.predict(X_te)
    metrics = evaluate_predictions(y_te, pred, TARGET_KIND)
    naive = evaluate_predictions(y_te, naive_predict(y_tr, len(y_te), TARGET_KIND), TARGET_KIND)
    headline = tstr_summary(metrics, None, naive)
    checks.append(("baseline_metrics", "balanced_accuracy" in metrics or "mae" in metrics, json.dumps(headline)))

    # 5. symptom ingest (offline heuristic)
    s = structure_symptoms("Exhausted with brain fog and bad cramps today")
    checks.append(("symptom_ingest", s["method"] in ("heuristic", "openai"), json.dumps(s)))

    # 6. mcPHASES probe (informational — may be red until download lands)
    info = mcphases.inspect()
    checks.append(("mcphases_probe", True, f"available={info['available']} n_files={info['n_files']}"))

    # persist tiny smoke artifact
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUTS_DIR / "smoke.json").write_text(
        json.dumps({"checks": [{"name": n, "ok": ok, "detail": d} for n, ok, d in checks]}, indent=2)
    )

    print("CycleBench smoke")
    print("-" * 40)
    failed = 0
    for name, ok, detail in checks:
        mark = "OK " if ok else "FAIL"
        print(f"[{mark}] {name}: {detail[:120]}")
        if not ok:
            failed += 1
    print("-" * 40)
    print("ALL GREEN" if failed == 0 else f"{failed} FAILED")
    return failed


if __name__ == "__main__":
    raise SystemExit(main())
