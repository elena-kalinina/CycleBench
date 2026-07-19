"""Versioned provenance manifests for adapters and released cohorts."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


def write_provenance_manifest(
    path: Path,
    *,
    source_name: str,
    adapter: str,
    license_tag: str,
    redistributable: bool,
    n_rows: int,
    n_participants: int,
    schema: str = "cyclebench_daily_v1",
    extra: dict[str, Any] | None = None,
) -> Path:
    """Write a provenance manifest next to a derived timeline artifact."""
    payload: dict[str, Any] = {
        "schema": schema,
        "source_name": source_name,
        "adapter": adapter,
        "license_tag": license_tag,
        "redistributable": redistributable,
        "n_rows": int(n_rows),
        "n_participants": int(n_participants),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "note": (
            "Provenance for a CycleBench daily timeline. "
            "Restricted clinical rows must not be redistributed."
            if not redistributable
            else "Open / synthetic artifact — safe to share under stated license."
        ),
    }
    if extra:
        payload["extra"] = extra
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    return path


def manifest_from_timeline(
    df: pd.DataFrame,
    *,
    source_name: str,
    adapter: str,
    path: Path,
    redistributable: bool,
    extra: dict[str, Any] | None = None,
) -> Path:
    license_tag = str(df["license_tag"].iloc[0]) if "license_tag" in df.columns and len(df) else "unknown"
    return write_provenance_manifest(
        path,
        source_name=source_name,
        adapter=adapter,
        license_tag=license_tag,
        redistributable=redistributable,
        n_rows=len(df),
        n_participants=int(df["participant_id"].nunique()) if "participant_id" in df.columns else 0,
        extra=extra,
    )
