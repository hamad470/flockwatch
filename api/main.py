"""FlockWatch REST API.  Run:  uvicorn api.main:app --reload   (docs at /docs)"""
import io, sqlite3, sys
from pathlib import Path
import pandas as pd
from fastapi import FastAPI, File, HTTPException, UploadFile
from scipy.io import wavfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from flockwatch.config import DB_PATH  # noqa: E402
from flockwatch.features import to_float  # noqa: E402
from flockwatch.predict import Predictor  # noqa: E402

app = FastAPI(title="FlockWatch API", version="1.0.0",
              description="Classify poultry-house audio and query flock early-warning status.")
_predictor = None


def predictor() -> Predictor:
    global _predictor
    if _predictor is None:
        _predictor = Predictor()
    return _predictor


def query(sql: str, params=()) -> list[dict]:
    with sqlite3.connect(DB_PATH) as con:
        return pd.read_sql(sql, con, params=params).to_dict(orient="records")


@app.get("/health")
def health():
    return {"status": "ok", "model_version": predictor().version}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        sr, x = wavfile.read(io.BytesIO(await file.read()))
    except Exception:
        raise HTTPException(400, "Please upload a valid WAV file")
    windows = predictor().predict_long(to_float(x), sr)
    return {"filename": file.filename, "n_windows": len(windows), "windows": windows}


@app.get("/houses")
def houses():
    return query("SELECT * FROM houses")


@app.get("/houses/{house_id}/status")
def house_status(house_id: str):
    rows = query("SELECT * FROM daily_index WHERE house_id = ? ORDER BY date DESC LIMIT 1", (house_id,))
    if not rows:
        raise HTTPException(404, f"Unknown house {house_id}")
    return rows[0]


@app.get("/alerts")
def alerts(level: str | None = None):
    if level:
        return query("SELECT * FROM daily_index WHERE alert_level = ? ORDER BY date", (level,))
    return query("SELECT * FROM daily_index WHERE alert_level IN ('Amber','Red') ORDER BY date")
