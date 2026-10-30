"""
Train and evaluate the poultry sound classifier.

Good practice applied:
  * group-aware split by recording session (no clips from one session in both train and test)
  * class weighting for imbalanced data (real datasets are imbalanced)
  * SpecAugment-style augmentation, early stopping on validation macro-F1
  * held-out test metrics, confusion matrix and model metadata saved for the model card
Usage:  python -m flockwatch.train [--epochs 30]
"""
from __future__ import annotations
import argparse, hashlib, json, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .config import RAW, MODELS, FIGURES, PROCESSED, CLASSES, SAMPLE_RATE
from .features import load_wav, clip_features
from .model import PoultrySoundCNN


def build_index(raw: Path = RAW) -> pd.DataFrame:
    meta_path = raw / "metadata.csv"
    if meta_path.exists():
        df = pd.read_csv(meta_path)
    else:   # real dataset without metadata: infer from folders, each file is its own group
        rows = [{"path": str(p.relative_to(raw)), "label": c.lower(), "session_id": p.stem}
                for c in CLASSES for p in sorted((raw / c).glob("*.wav"))]
        df = pd.DataFrame(rows)
    df = df[df.label.isin(CLASSES)].reset_index(drop=True)
    if df.empty:
        raise SystemExit("No audio found in data/raw/<healthy|unhealthy|noise>. Run `python -m flockwatch.make_dataset` first.")
    return df


def featurise(df: pd.DataFrame, raw: Path = RAW) -> np.ndarray:
    key = hashlib.md5("".join(df.path).encode()).hexdigest()[:10]
    cache = PROCESSED / f"features_{key}.npy"
    if cache.exists():
        return np.load(cache)
    feats = []
    for p in df.path:
        x, sr = load_wav(raw / p)
        feats.append(clip_features(x, sr))
    arr = np.stack(feats)[:, None]          # (N, 1, mels, frames)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    np.save(cache, arr)
    return arr


def augment(x: torch.Tensor) -> torch.Tensor:
    x = x.clone()
    b, _, m, t = x.shape
    shift = int(torch.randint(-20, 21, (1,)))
    x = torch.roll(x, shift, dims=3)
    for i in range(b):
        f0 = int(torch.randint(0, m - 8, (1,))); x[i, :, f0:f0 + int(torch.randint(0, 9, (1,))), :] = 0
        t0 = int(torch.randint(0, t - 20, (1,))); x[i, :, :, t0:t0 + int(torch.randint(0, 21, (1,)))] = 0
    return x


def split(df: pd.DataFrame, seed: int = 0):
    groups = df.session_id.values
    gss = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
    trval, test = next(gss.split(df, groups=groups))
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.18, random_state=seed)
    tr, va = next(gss2.split(df.iloc[trval], groups=groups[trval]))
    return trval[tr], trval[va], test


def evaluate(model, X, y, batch=64):
    model.eval()
    preds, probs = [], []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            p = torch.softmax(model(X[i:i + batch]), 1)
            probs.append(p); preds.append(p.argmax(1))
    return torch.cat(preds).numpy(), torch.cat(probs).numpy()


def main(epochs: int = 30, batch: int = 32, seed: int = 0):
    torch.manual_seed(seed); np.random.seed(seed)
    df = build_index()
    X = torch.tensor(featurise(df))
    y = torch.tensor(df.label.map({c: i for i, c in enumerate(CLASSES)}).values)
    tr, va, te = split(df, seed)
    assert not set(df.session_id.iloc[tr]) & set(df.session_id.iloc[te]), "session leakage"
    counts = np.bincount(y[tr].numpy(), minlength=len(CLASSES))
    weights = torch.tensor(len(tr) / (len(CLASSES) * np.maximum(counts, 1)), dtype=torch.float32)

    model = PoultrySoundCNN()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    best, best_state, patience, history = -1, None, 0, []
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(len(tr))
        total = 0.0
        for i in range(0, len(tr), batch):
            idx = torch.tensor(tr)[perm[i:i + batch]]
            opt.zero_grad()
            loss = loss_fn(model(augment(X[idx])), y[idx])
            loss.backward(); opt.step(); total += loss.item() * len(idx)
        sched.step()
        vp, _ = evaluate(model, X[va], y[va])
        vf1 = f1_score(y[va], vp, average="macro")
        history.append({"epoch": ep + 1, "train_loss": total / len(tr), "val_macro_f1": vf1})
        print(f"epoch {ep+1:02d} loss {total/len(tr):.3f} val_macroF1 {vf1:.3f}")
        if vf1 > best + 1e-4:
            best, best_state, patience = vf1, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            patience += 1
            if patience >= 6:
                print("early stopping"); break
    model.load_state_dict(best_state)
    tp, tprob = evaluate(model, X[te], y[te])
    report = classification_report(y[te], tp, target_names=CLASSES, output_dict=True, zero_division=0)
    cm = confusion_matrix(y[te], tp, labels=range(len(CLASSES)))
    source = df["source"].iloc[0] if "source" in df else "real"
    metrics = {
        "data_source": source, "n_clips": int(len(df)), "n_sessions": int(df.session_id.nunique()),
        "split_sizes": {"train": int(len(tr)), "val": int(len(va)), "test": int(len(te))},
        "test_accuracy": round(accuracy_score(y[te], tp), 4),
        "test_macro_f1": round(f1_score(y[te], tp, average="macro"), 4),
        "unhealthy_recall": round(report["unhealthy"]["recall"], 4),
        "unhealthy_precision": round(report["unhealthy"]["precision"], 4),
        "per_class": {c: {k: round(v, 4) for k, v in report[c].items()} for c in CLASSES},
        "confusion_matrix": cm.tolist(), "best_val_macro_f1": round(best, 4),
        "epochs_run": len(history), "train_seconds": round(time.time() - t0, 1),
    }
    MODELS.mkdir(exist_ok=True); FIGURES.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "classes": CLASSES, "sample_rate": SAMPLE_RATE,
                "version": "1.0.0", "trained_on": source}, MODELS / "flockwatch_cnn.pt")
    (MODELS / "metrics.json").write_text(json.dumps(metrics, indent=2))
    pd.DataFrame(history).to_csv(MODELS / "training_history.csv", index=False)

    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(3), CLASSES); ax.set_yticks(range(3), CLASSES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title("Test confusion matrix")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout(); fig.savefig(FIGURES / "confusion_matrix.png", dpi=150); plt.close(fig)
    h = pd.DataFrame(history)
    fig, ax = plt.subplots(figsize=(6, 3.2))
    ax.plot(h.epoch, h.train_loss, label="train loss"); ax2 = ax.twinx()
    ax2.plot(h.epoch, h.val_macro_f1, color="tab:orange", label="val macro-F1")
    ax.set_xlabel("epoch"); ax.set_ylabel("loss"); ax2.set_ylabel("macro-F1"); ax.set_title("Training curves")
    fig.tight_layout(); fig.savefig(FIGURES / "training_curves.png", dpi=150); plt.close(fig)
    print(json.dumps({k: metrics[k] for k in ["test_accuracy", "test_macro_f1", "unhealthy_recall", "unhealthy_precision"]}))
    return metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    main(ap.parse_args().epochs)
