import pytest
from pathlib import Path
from flockwatch.config import MODELS, SAMPLES, DB_PATH

pytestmark = pytest.mark.skipif(not (MODELS / "flockwatch_cnn.pt").exists() or not DB_PATH.exists(),
                                reason="run the pipeline first")


def test_health_predict_and_status():
    from fastapi.testclient import TestClient
    from api.main import app
    c = TestClient(app)
    assert c.get("/health").json()["status"] == "ok"
    with open(SAMPLES / "sample_unhealthy.wav", "rb") as f:
        r = c.post("/predict", files={"file": ("s.wav", f, "audio/wav")})
    assert r.status_code == 200 and r.json()["windows"][0]["label"] in ("healthy", "unhealthy", "noise")
    assert c.get("/houses/H3/status").status_code == 200
    assert c.get("/houses/NOPE/status").status_code == 404
