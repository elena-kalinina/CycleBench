"""mcPHASES adapter stub — wire when PhysioNet download finishes.

Expected layout (adjust once you inspect the archive):
  data/raw/mcphases/   ← extracted PhysioNet package

This module maps whatever columns exist into the CycleBench daily schema.
Until real files land, `available()` returns False and the pipeline stays
on the synthetic path.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from cyclebench.config import RAW_DIR
from cyclebench.data_contract.schema import validate_timeline


def default_mcphases_root() -> Path:
    return RAW_DIR / "mcphases"


def available(root: Path | None = None) -> bool:
    root = root or default_mcphases_root()
    if not root.exists():
        return False
    # any csv/parquet/tsv under the tree counts as "present"
    for pat in ("**/*.csv", "**/*.parquet", "**/*.tsv", "**/*.txt"):
        if any(root.glob(pat)):
            return True
    return False


def inspect(root: Path | None = None) -> dict[str, Any]:
    """Lightweight inventory so H0–1 can pick the target channel."""
    root = root or default_mcphases_root()
    files = []
    if root.exists():
        for p in sorted(root.rglob("*")):
            if p.is_file() and p.suffix.lower() in {".csv", ".parquet", ".tsv", ".txt", ".xlsx"}:
                files.append(str(p.relative_to(root)))
    return {
        "root": str(root),
        "available": available(root),
        "files": files[:50],
        "n_files": len(files),
        "note": "Inspect these files next; map columns → cyclebench daily schema in adapt().",
    }


def adapt(root: Path | None = None) -> pd.DataFrame:
    """Map mcPHASES → CycleBench daily timeline.

    TODO (H1–4): implement once download completes. Raise clearly until then
    so smoke/benchmark never silently invent real-looking patient rows.
    """
    root = root or default_mcphases_root()
    if not available(root):
        raise FileNotFoundError(
            f"mcPHASES not found at {root}. Place extracted PhysioNet files there, "
            "then implement column mapping in cyclebench/adapters/mcphases.py::adapt()."
        )
    raise NotImplementedError(
        "mcPHASES files detected but column mapping not yet implemented. "
        f"Inventory: {inspect(root)}"
    )


def try_adapt() -> tuple[pd.DataFrame | None, dict[str, Any]]:
    info = inspect()
    if not info["available"]:
        return None, info
    try:
        df = adapt()
        errors = validate_timeline(df)
        info["validation_errors"] = errors
        return (df if not errors else None), info
    except NotImplementedError as e:
        info["status"] = str(e)
        return None, info
