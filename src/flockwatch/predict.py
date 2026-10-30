"""Inference helpers: single clips, long recordings (sliding window) and files."""
from __future__ import annotations
from pathlib import Path
import numpy as np
import torch
from .config import MODELS, CLASSES, SAMPLE_RATE, CLIP_SAMPLES
from .features import clip_features, load_wav, resample
from .model import PoultrySoundCNN


class Predictor:
    def __init__(self, model_path: Path = MODELS / "flockwatch_cnn.pt"):
        ckpt = torch.load(model_path, map_location="cpu", weights_only=False)
        self.classes = ckpt.get("classes", CLASSES)
        self.version = ckpt.get("version", "unknown")
        self.model = PoultrySoundCNN(len(self.classes))
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()

    @torch.no_grad()
    def predict_batch(self, clips: list[np.ndarray], sr: int = SAMPLE_RATE) -> np.ndarray:
        feats = torch.tensor(np.stack([clip_features(c, sr) for c in clips])[:, None])
        return torch.softmax(self.model(feats), 1).numpy()

    def predict_clip(self, x: np.ndarray, sr: int = SAMPLE_RATE) -> dict:
        p = self.predict_batch([x], sr)[0]
        return {"label": self.classes[int(p.argmax())], **{c: float(v) for c, v in zip(self.classes, p)}}

    def predict_long(self, x: np.ndarray, sr: int, hop_s: float = 2.5) -> list[dict]:
        """Slide a 5-second window over a long recording."""
        x = resample(x, sr)
        hop = int(hop_s * SAMPLE_RATE)
        starts = range(0, max(len(x) - CLIP_SAMPLES, 0) + 1, hop)
        clips = [x[s:s + CLIP_SAMPLES] for s in starts]
        probs = self.predict_batch(clips)
        return [{"start_s": s / SAMPLE_RATE, "label": self.classes[int(p.argmax())],
                 **{c: float(v) for c, v in zip(self.classes, p)}} for s, p in zip(starts, probs)]

    def predict_file(self, path) -> list[dict]:
        x, sr = load_wav(path)
        return self.predict_long(x, sr)
