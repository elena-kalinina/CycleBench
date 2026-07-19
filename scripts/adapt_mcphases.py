#!/usr/bin/env python3
"""Adapt local mcPHASES → data/processed/mcphases_daily.parquet (gitignored).

PhysioNet Restricted: do NOT commit or publish the output.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cyclebench.adapters.mcphases import adapt_and_save, inspect
from cyclebench.config import FEATURE_COLUMNS
from cyclebench.preprocess.splits import missingness_report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=None)
    args = ap.parse_args()

    info = inspect(args.root)
    print(json.dumps({k: info[k] for k in ("root", "available", "n_files", "license")}, indent=2))
    if not info["available"]:
        raise SystemExit("mcPHASES not available — symlink to data/raw/mcphases first")

    df, path = adapt_and_save(root=args.root)
    report = {
        "out": str(path),
        "n_rows": int(len(df)),
        "n_participants": int(df["participant_id"].nunique()),
        "phases": df["cycle_phase"].value_counts().to_dict(),
        "missingness": missingness_report(df, [c for c in FEATURE_COLUMNS if c != "cycle_day"]),
        "license_tag": df["license_tag"].iloc[0],
        "redistribute": False,
    }
    print(json.dumps(report, indent=2))
    print(f"wrote {path} (LOCAL ONLY — gitignored)")


if __name__ == "__main__":
    main()
