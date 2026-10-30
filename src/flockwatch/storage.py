"""SQLite storage layer: schema, connection and audit logging."""
from __future__ import annotations
import sqlite3
from datetime import datetime, timezone
from .config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS houses (
    house_id TEXT PRIMARY KEY, farm TEXT NOT NULL, capacity_birds INTEGER, microphones INTEGER);
CREATE TABLE IF NOT EXISTS recordings (
    clip_id TEXT PRIMARY KEY, house_id TEXT, device_id TEXT, recorded_at TEXT,
    sample_rate INTEGER, duration_s REAL, sha256 TEXT, rms REAL, clipping_ratio REAL,
    status TEXT CHECK (status IN ('accepted','quarantined')), reason TEXT,
    source TEXT, ingested_at TEXT);
CREATE TABLE IF NOT EXISTS predictions (
    clip_id TEXT PRIMARY KEY REFERENCES recordings(clip_id), model_version TEXT,
    p_healthy REAL, p_unhealthy REAL, p_noise REAL, label TEXT, predicted_at TEXT);
CREATE TABLE IF NOT EXISTS record_cards (
    house_id TEXT, date TEXT, age_days INTEGER, birds_alive INTEGER, mortality INTEGER, culls INTEGER,
    feed_kg REAL, water_l REAL, avg_weight_g REAL, avg_temp_c REAL, PRIMARY KEY (house_id, date));
CREATE TABLE IF NOT EXISTS daily_index (
    house_id TEXT, date TEXT, n_clips INTEGER, n_bird_clips INTEGER, sound_index REAL,
    baseline REAL, z_score REAL, water_change REAL, alert_level TEXT, PRIMARY KEY (house_id, date));
CREATE TABLE IF NOT EXISTS audit_log (ts TEXT, action TEXT, detail TEXT);
CREATE INDEX IF NOT EXISTS idx_rec_house ON recordings(house_id, recorded_at);
"""


def connect(path=DB_PATH) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.executescript(SCHEMA)
    return con


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def audit(con: sqlite3.Connection, action: str, detail: str):
    con.execute("INSERT INTO audit_log VALUES (?,?,?)", (now(), action, detail))
    con.commit()
