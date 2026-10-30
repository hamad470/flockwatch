"""
Data collection & governance: validate incoming recordings before they are used.

Every clip is checked for required metadata, sample rate, duration, clipping and silence,
fingerprinted with SHA-256 (duplicate detection / lineage) and stored with an accepted or
quarantined status and a human-readable reason. Nothing is silently dropped.
"""
from __future__ import annotations
import hashlib
from dataclasses import dataclass
import numpy as np
from .config import (MIN_SAMPLE_RATE, MIN_DURATION_S, MAX_DURATION_S, MAX_CLIPPING_RATIO,
                     MIN_RMS, REQUIRED_METADATA)
from .storage import now


@dataclass
class QualityResult:
    status: str
    reason: str
    rms: float
    clipping_ratio: float
    duration_s: float
    sha256: str


def check_clip(x: np.ndarray, sr: int, meta: dict, seen_hashes: set | None = None) -> QualityResult:
    duration = len(x) / sr if sr else 0.0
    rms = float(np.sqrt(np.mean(x ** 2))) if len(x) else 0.0
    clipping = float(np.mean(np.abs(x) >= 0.999)) if len(x) else 0.0
    digest = hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
    problems = []
    missing = [k for k in REQUIRED_METADATA if not meta.get(k)]
    if missing:
        problems.append(f"missing metadata: {', '.join(missing)}")
    if sr < MIN_SAMPLE_RATE:
        problems.append(f"sample rate {sr} Hz below {MIN_SAMPLE_RATE}")
    if not (MIN_DURATION_S <= duration <= MAX_DURATION_S):
        problems.append(f"duration {duration:.2f}s outside {MIN_DURATION_S}-{MAX_DURATION_S}s")
    if clipping > MAX_CLIPPING_RATIO:
        problems.append(f"clipping {clipping:.1%} (microphone overload)")
    if rms < MIN_RMS:
        problems.append("near-silent recording (possible device fault)")
    if seen_hashes is not None:
        if digest in seen_hashes:
            problems.append("duplicate recording")
        seen_hashes.add(digest)
    return QualityResult("quarantined" if problems else "accepted", "; ".join(problems),
                         rms, clipping, duration, digest)


def store_recording(con, meta: dict, sr: int, q: QualityResult, source: str = "synthetic"):
    con.execute(
        "INSERT OR REPLACE INTO recordings VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (meta.get("clip_id"), meta.get("house_id"), meta.get("device_id"), meta.get("recorded_at"),
         sr, round(q.duration_s, 3), q.sha256, round(q.rms, 5), round(q.clipping_ratio, 5),
         q.status, q.reason, source, now()),
    )
