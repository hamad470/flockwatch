"""Central configuration: paths and audio/feature settings."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"
RAW = DATA / "raw"                 # data/raw/<healthy|unhealthy|noise>/*.wav (same layout as public datasets)
PROCESSED = DATA / "processed"
SAMPLES = DATA / "samples"
MODELS = ROOT / "models"
FIGURES = ROOT / "reports" / "figures"
DB_PATH = PROCESSED / "flockwatch.db"

CLASSES = ["healthy", "unhealthy", "noise"]

# Audio / features
SAMPLE_RATE = 16_000               # everything is resampled to 16 kHz
CLIP_SECONDS = 5.0
CLIP_SAMPLES = int(SAMPLE_RATE * CLIP_SECONDS)
N_FFT = 1024
HOP = 512
N_MELS = 64
FMIN, FMAX = 50.0, 8000.0

# Ingestion quality rules (data governance)
MIN_SAMPLE_RATE = 16_000
MIN_DURATION_S, MAX_DURATION_S = 4.5, 5.5
MAX_CLIPPING_RATIO = 0.01          # >1% of samples at full scale -> quarantine
MIN_RMS = 1e-3                     # near-silent recordings -> quarantine
REQUIRED_METADATA = ["clip_id", "house_id", "device_id", "recorded_at"]

# Early-warning rule
BASELINE_DAYS = 7
Z_THRESHOLD = 3.0
MIN_INDEX_FOR_ALERT = 0.20
