"""Audio loading, resampling and log-mel spectrogram features (NumPy/SciPy only)."""
from __future__ import annotations
from functools import lru_cache
from math import gcd
import numpy as np
from scipy.io import wavfile
from scipy.signal import resample_poly, stft
from .config import SAMPLE_RATE, CLIP_SAMPLES, N_FFT, HOP, N_MELS, FMIN, FMAX


def to_float(x: np.ndarray) -> np.ndarray:
    if np.issubdtype(x.dtype, np.integer):
        x = x.astype(np.float32) / np.iinfo(x.dtype).max
    x = x.astype(np.float32)
    if x.ndim == 2:                       # stereo -> mono
        x = x.mean(axis=1)
    return x


def load_wav(path) -> tuple[np.ndarray, int]:
    sr, x = wavfile.read(path)
    return to_float(x), sr


def resample(x: np.ndarray, sr: int, target: int = SAMPLE_RATE) -> np.ndarray:
    if sr == target:
        return x
    g = gcd(sr, target)
    return resample_poly(x, target // g, sr // g).astype(np.float32)


def fix_length(x: np.ndarray, n: int = CLIP_SAMPLES) -> np.ndarray:
    if len(x) >= n:
        return x[:n]
    return np.pad(x, (0, n - len(x)))


def save_wav(path, x: np.ndarray, sr: int = SAMPLE_RATE):
    wavfile.write(path, sr, (np.clip(x, -1, 1) * 32767).astype(np.int16))


def _hz_to_mel(f):
    return 2595 * np.log10(1 + f / 700)


def _mel_to_hz(m):
    return 700 * (10 ** (m / 2595) - 1)


@lru_cache(maxsize=1)
def mel_filterbank() -> np.ndarray:
    freqs = np.linspace(0, SAMPLE_RATE / 2, N_FFT // 2 + 1)
    mels = np.linspace(_hz_to_mel(FMIN), _hz_to_mel(FMAX), N_MELS + 2)
    hz = _mel_to_hz(mels)
    fb = np.zeros((N_MELS, len(freqs)), dtype=np.float32)
    for i in range(N_MELS):
        lo, c, hi = hz[i], hz[i + 1], hz[i + 2]
        up = (freqs - lo) / (c - lo)
        down = (hi - freqs) / (hi - c)
        fb[i] = np.maximum(0, np.minimum(up, down))
    return fb


def log_mel(x: np.ndarray) -> np.ndarray:
    """Return a (N_MELS, frames) log-mel spectrogram, standardised per clip."""
    _, _, z = stft(x, fs=SAMPLE_RATE, nperseg=N_FFT, noverlap=N_FFT - HOP, boundary=None, padded=False)
    power = np.abs(z) ** 2
    mel = mel_filterbank() @ power
    logmel = np.log(mel + 1e-6)
    return ((logmel - logmel.mean()) / (logmel.std() + 1e-6)).astype(np.float32)


def clip_features(x: np.ndarray, sr: int) -> np.ndarray:
    return log_mel(fix_length(resample(x, sr)))
