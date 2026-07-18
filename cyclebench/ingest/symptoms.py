"""Bounded OpenAI role: free-text / voice transcript → structured symptoms.

Never generates labels, metrics, or health findings. Optional — pipeline
runs without an API key (heuristic fallback).
"""

from __future__ import annotations

import json
import os
import re
from typing import Any


SYMPTOM_KEYS = ("symptom_fatigue", "symptom_mood", "symptom_pain")


def structure_symptoms(text: str) -> dict[str, Any]:
    """Map a symptom note to 0–3 scores.

    Tries OpenAI if OPENAI_API_KEY is set; otherwise a transparent keyword
    heuristic so the demo beat still works offline.
    """
    text = (text or "").strip()
    if not text:
        return {k: 0.0 for k in SYMPTOM_KEYS} | {"method": "empty", "raw": text}

    if os.getenv("OPENAI_API_KEY"):
        try:
            return _openai_structure(text)
        except Exception as e:  # noqa: BLE001 — demo must not hard-stop
            out = _heuristic_structure(text)
            out["openai_error"] = str(e)
            return out
    return _heuristic_structure(text)


def _heuristic_structure(text: str) -> dict[str, Any]:
    t = text.lower()
    def score(words: list[str]) -> float:
        hits = sum(1 for w in words if w in t)
        if any(x in t for x in ("severe", "extreme", "unbearable")):
            hits += 1
        return float(min(3, hits))

    return {
        "symptom_fatigue": score(["fatigue", "tired", "exhausted", "brain fog", "sleepy"]),
        "symptom_mood": score(["anxious", "irritable", "sad", "mood", "depressed", "angry"]),
        "symptom_pain": score(["cramp", "pain", "ache", "sore", "headache", "migraine"]),
        "method": "heuristic",
        "raw": text,
    }


def _openai_structure(text: str) -> dict[str, Any]:
    # Lazy import — optional dependency
    from openai import OpenAI

    client = OpenAI()
    prompt = (
        "Extract symptom severity scores from this women's health diary note. "
        "Return ONLY JSON with keys symptom_fatigue, symptom_mood, symptom_pain, "
        "each an integer 0-3 (0=none, 3=severe). No diagnosis.\n\n"
        f"Note: {text}"
    )
    resp = client.chat.completions.create(
        model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )
    content = resp.choices[0].message.content or "{}"
    # tolerate fenced JSON
    m = re.search(r"\{.*\}", content, re.S)
    data = json.loads(m.group(0) if m else content)
    out = {k: float(max(0, min(3, int(data.get(k, 0))))) for k in SYMPTOM_KEYS}
    out["method"] = "openai"
    out["raw"] = text
    return out
