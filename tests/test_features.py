import numpy as np
from flockwatch.audio_synth import make_clip
from flockwatch.config import N_MELS, CLIP_SAMPLES
from flockwatch.features import clip_features, resample, fix_length


def test_clip_features_shape_and_normalisation():
    x = make_clip("healthy", np.random.default_rng(0))
    f = clip_features(x, 16000)
    assert f.shape[0] == N_MELS and f.shape[1] > 100
    assert abs(f.mean()) < 1e-3 and abs(f.std() - 1) < 1e-2


def test_resample_and_fix_length_from_96k():
    x = np.random.default_rng(1).normal(0, 0.1, 96000 * 6).astype(np.float32)   # 6 s at 96 kHz, like the public dataset
    y = fix_length(resample(x, 96000))
    assert len(y) == CLIP_SAMPLES
