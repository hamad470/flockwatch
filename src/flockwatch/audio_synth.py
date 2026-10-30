"""
Synthetic poultry-house audio generator.

Produces 5-second clips in three classes that mirror public datasets
(e.g. Mendeley "Poultry Vocalization Signal Dataset for Early Disease Detection"):
  healthy   - normal vocalisations over ventilation noise
  unhealthy - fewer/weaker calls plus coughs, sneezes and rales (respiratory sounds)
  noise     - farm noise with no bird vocalisations (fans, feeders, impacts)
Used so the full pipeline runs without access to real recordings.
"""
from __future__ import annotations
import numpy as np
from scipy.signal import butter, sosfilt
from .config import SAMPLE_RATE as SR, CLIP_SAMPLES


def _band(x, lo, hi, order=4):
    sos = butter(order, [lo, hi], btype="band", fs=SR, output="sos")
    return sosfilt(sos, x)


def _env(n, attack, decay, rng):
    t = np.arange(n) / SR
    a = max(attack, 1e-3)
    return np.minimum(t / a, 1.0) * np.exp(-np.maximum(t - a, 0) / max(decay, 1e-3))


def _place(buf, sig, rng):
    if len(sig) >= len(buf):
        return
    start = rng.integers(0, len(buf) - len(sig))
    buf[start:start + len(sig)] += sig


def fan_noise(rng, level):
    """Ventilation fan: low-frequency broadband noise + motor hum harmonics."""
    n = CLIP_SAMPLES
    white = rng.normal(0, 1, n)
    brown = np.cumsum(white)
    brown -= np.convolve(brown, np.ones(400) / 400, mode="same")
    brown /= np.abs(brown).max() + 1e-9
    t = np.arange(n) / SR
    f0 = rng.uniform(90, 130)
    hum = sum(np.sin(2 * np.pi * f0 * k * t + rng.uniform(0, 6.28)) / k for k in range(1, 5))
    return level * (0.7 * brown + 0.08 * hum + 0.05 * _band(white, 200, 6000))


def chirp_call(rng, strength=1.0):
    """Normal chicken vocalisation: short frequency-modulated call with harmonics."""
    dur = rng.uniform(0.06, 0.22)
    n = int(dur * SR)
    t = np.arange(n) / SR
    f_start, f_end = rng.uniform(2600, 4300), rng.uniform(1700, 3300)
    phase = 2 * np.pi * (f_start * t + (f_end - f_start) * t ** 2 / (2 * dur))
    sig = np.sin(phase) + 0.35 * np.sin(2 * phase) + 0.15 * np.sin(3 * phase)
    return strength * sig * np.hanning(n)


def cough(rng, strength=1.0):
    """Cough/sneeze: very short broadband burst with sharp attack."""
    n = int(rng.uniform(0.03, 0.09) * SR)
    burst = _band(rng.normal(0, 1, n), rng.uniform(400, 900), rng.uniform(2500, 4500))
    return strength * 1.6 * burst * _env(n, 0.003, rng.uniform(0.01, 0.03), rng)


def rale(rng, strength=1.0):
    """Rale/gurgle: amplitude-modulated low-mid noise (wet respiratory sound)."""
    n = int(rng.uniform(0.2, 0.5) * SR)
    t = np.arange(n) / SR
    am = 0.5 * (1 + np.sin(2 * np.pi * rng.uniform(18, 40) * t))
    sig = _band(rng.normal(0, 1, n), 200, 900) * am
    return strength * 0.9 * sig * np.hanning(n)


def machinery(rng):
    """Feeder motor / chain: tonal hum with fluctuation."""
    n = int(rng.uniform(0.8, 3.0) * SR)
    t = np.arange(n) / SR
    f = rng.uniform(250, 600)
    sig = np.sin(2 * np.pi * f * t) + 0.5 * np.sin(2 * np.pi * 2 * f * t)
    sig *= 0.6 + 0.4 * np.sin(2 * np.pi * rng.uniform(2, 8) * t)
    return 0.5 * sig * np.hanning(n)


def impact(rng):
    """Door bang / metal impact."""
    n = int(0.25 * SR)
    sig = _band(rng.normal(0, 1, n), 100, 5000) * _env(n, 0.001, 0.05, rng)
    return 1.2 * sig


def make_clip(label: str, rng: np.random.Generator, severity: float = 1.0) -> np.ndarray:
    """Return one float32 clip in [-1, 1]. `severity` (0-1) scales respiratory signs."""
    buf = fan_noise(rng, rng.uniform(0.05, 0.25))
    if label in ("healthy", "unhealthy"):
        activity = rng.uniform(0.6, 1.0) if label == "healthy" else rng.uniform(0.25, 0.7)
        for _ in range(int(rng.integers(6, 24) * activity)):
            _place(buf, chirp_call(rng, rng.uniform(0.15, 0.6)), rng)
        if label == "healthy" and rng.random() < 0.08:           # occasional sneeze in healthy flocks
            _place(buf, cough(rng, rng.uniform(0.2, 0.5)), rng)
    if label == "unhealthy":
        for _ in range(int(rng.integers(2, 9) * max(severity, 0.2))):
            _place(buf, cough(rng, rng.uniform(0.3, 0.9) * (0.5 + severity / 2)), rng)
        for _ in range(int(rng.integers(0, 4) * severity)):
            _place(buf, rale(rng, rng.uniform(0.2, 0.6)), rng)
    if label == "noise" or rng.random() < 0.25:                   # farm noise appears in every class
        for _ in range(int(rng.integers(1, 4))):
            _place(buf, machinery(rng) if rng.random() < 0.6 else impact(rng), rng)
    peak = np.abs(buf).max()
    buf = buf / (peak + 1e-9) * rng.uniform(0.3, 0.9)
    return buf.astype(np.float32)
