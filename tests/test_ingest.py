import numpy as np
from flockwatch.audio_synth import make_clip
from flockwatch.ingest import check_clip

META = {"clip_id": "c1", "house_id": "H1", "device_id": "MIC-1", "recorded_at": "2026-03-01T10:00:00"}


def clean():
    return make_clip("healthy", np.random.default_rng(5))


def test_clean_clip_accepted():
    assert check_clip(clean(), 16000, META).status == "accepted"


def test_clipping_silence_metadata_and_duplicates_quarantined():
    assert "clipping" in check_clip(np.clip(clean() * 10, -1, 1), 16000, META).reason
    assert "silent" in check_clip(clean() * 1e-5, 16000, META).reason
    assert "missing metadata" in check_clip(clean(), 16000, {**META, "device_id": ""}).reason
    assert "sample rate" in check_clip(clean(), 8000, META).reason
    seen = set()
    x = clean()
    check_clip(x, 16000, META, seen)
    assert "duplicate" in check_clip(x, 16000, META, seen).reason
