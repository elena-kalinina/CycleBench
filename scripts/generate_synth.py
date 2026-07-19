#!/usr/bin/env python3
"""Generate the open SynthCycle cohort (PHI-free)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cyclebench.config import N_SYNTH_PARTICIPANTS, DAYS_PER_PARTICIPANT, SYNTH_DIR
from cyclebench.provenance import manifest_from_timeline
from cyclebench.synthcycle.generator import SynthCycleGenerator


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=N_SYNTH_PARTICIPANTS)
    p.add_argument("--days", type=int, default=DAYS_PER_PARTICIPANT)
    p.add_argument("--out", type=Path, default=SYNTH_DIR / "synthcycle_v1.parquet")
    args = p.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    gen = SynthCycleGenerator()
    df = gen.generate(n_participants=args.n, days=args.days)
    df.to_parquet(args.out, index=False)
    # also CSV for easy inspection
    csv_out = args.out.with_suffix(".csv")
    df.to_csv(csv_out, index=False)
    manifest = gen.manifest(df)
    man_path = args.out.with_name("synthcycle_manifest.json")
    man_path.write_text(json.dumps(manifest, indent=2))
    # Provenance workflow artifact (open — safe to commit alongside generator)
    manifest_from_timeline(
        df,
        source_name="SynthCycle",
        adapter="cyclebench.synthcycle.generator",
        path=args.out.with_name("synthcycle_provenance.json"),
        redistributable=True,
        extra={"generator_manifest": manifest},
    )
    print(f"wrote {args.out} ({len(df)} rows, {df['participant_id'].nunique()} participants)")
    print(f"manifest: {man_path}")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
