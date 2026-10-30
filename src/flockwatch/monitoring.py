"""
Simulate a monitoring deployment: 4 broiler houses, one 40-day crop, microphones recording a
5-second clip every hour. Two houses develop respiratory disease. Clips pass through the
ingestion checks, accepted clips are scored by the trained model, and daily flock record
cards (mortality, feed, water, weight) are generated alongside.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from .config import PROCESSED, SAMPLES, SAMPLE_RATE
from .audio_synth import make_clip
from .features import save_wav
from .ingest import check_clip, store_recording
from .predict import Predictor
from .storage import connect, audit, now
from .early_warning import daily_sound_index, add_alerts, evaluate_lead_time

HOUSES = pd.DataFrame({
    "house_id": ["H1", "H2", "H3", "H4"], "farm": ["Farm A", "Farm A", "Farm B", "Farm B"],
    "capacity_birds": [25000, 25000, 30000, 30000], "microphones": [4, 4, 6, 6],
})
OUTBREAKS = {"H3": (24, 1.0), "H4": (19, 0.6)}      # house: (onset age in days, severity)
DAYS, CLIPS_PER_DAY = 40, 24
PLACEMENT = pd.Timestamp("2026-03-02")
TARGET = {0: 42, 7: 190, 14: 500, 21: 1000, 28: 1650, 35: 2300, 42: 2900}


def disease_stage(house, age):
    if house not in OUTBREAKS:
        return 0.0, 0.0
    onset, sev = OUTBREAKS[house]
    d = age - onset
    sound = sev * min(d + 1, 4) / 4 if 0 <= d <= 10 else 0.0           # coughing rises first
    mort = sev * min(d - 1, 4) / 4 if 2 <= d <= 11 else 0.0              # deaths follow ~2 days later
    return sound, mort


def record_card(rng, house, cap, age, alive):
    _, mort_stage = disease_stage(house, age)
    base = 0.0025 if age < 7 else 0.0006
    mortality = int(rng.poisson(alive * base * (1 + 5 * mort_stage)))
    culls = int(rng.poisson(alive * 0.0001))
    w = np.interp(age, list(TARGET), list(TARGET.values())) * (1 - 0.05 * mort_stage) * rng.normal(0.99, 0.015)
    alive_end = alive - mortality - culls
    feed = alive_end * max(12, 0.072 * w + 8) / 1000 * rng.normal(1, 0.03)
    sound_stage, _ = disease_stage(house, age)
    water = feed * 1.8 * (1 - 0.15 * max(sound_stage, mort_stage)) * rng.normal(1, 0.02)
    temp = max(20.0, 32 - age * 0.3) + rng.normal(0, 0.6)
    return alive_end, dict(age_days=age, birds_alive=alive_end, mortality=mortality, culls=culls,
                           feed_kg=round(feed, 1), water_l=round(water, 1), avg_weight_g=round(w),
                           avg_temp_c=round(temp, 1))


def run(seed: int = 11):
    rng = np.random.default_rng(seed)
    con = connect()
    for t in ["recordings", "predictions", "record_cards", "daily_index", "houses"]:
        con.execute(f"DELETE FROM {t}")
    HOUSES.to_sql("houses", con, if_exists="append", index=False)
    audit(con, "simulation_start", f"{len(HOUSES)} houses x {DAYS} days x {CLIPS_PER_DAY} clips/day")
    predictor = Predictor()
    seen, accepted_clips, accepted_meta, cards = set(), [], [], []
    for _, h in HOUSES.iterrows():
        alive = int(h.capacity_birds * 0.98)
        start = PLACEMENT + pd.Timedelta(days=int(h.name) * 2)
        for age in range(DAYS):
            date = start + pd.Timedelta(days=age)
            alive, card = record_card(rng, h.house_id, h.capacity_birds, age, alive)
            cards.append({"house_id": h.house_id, "date": str(date.date()), **card})
            if h.house_id == "H2" and age in (15, 16):          # microphone outage: no clips arrive
                continue
            sound_stage, _ = disease_stage(h.house_id, age)
            for k in range(CLIPS_PER_DAY):
                hour = k
                ts = date + pd.Timedelta(hours=hour, minutes=int(rng.integers(0, 60)))
                p_unhealthy = 0.03 + 0.7 * sound_stage
                p_noise = 0.25 if hour in (6, 18) else 0.08          # feeding times are noisy
                r = rng.random()
                label = "noise" if r < p_noise else ("unhealthy" if r < p_noise + p_unhealthy else "healthy")
                x = make_clip(label, rng, severity=max(sound_stage, 0.35))
                meta = {"clip_id": f"{h.house_id}-{date:%Y%m%d}-{hour:02d}", "house_id": h.house_id,
                        "device_id": f"MIC-{h.house_id}-{1 + k % h.microphones}", "recorded_at": ts.isoformat()}
                u = rng.random()                                     # realistic data problems
                if u < 0.01:
                    x = np.clip(x * 8, -1, 1)                        # overloaded microphone
                elif u < 0.02:
                    x = x * 1e-5                                     # dead microphone
                elif u < 0.025:
                    meta["device_id"] = ""                           # missing metadata
                q = check_clip(x, SAMPLE_RATE, meta, seen)
                store_recording(con, meta, SAMPLE_RATE, q)
                if q.status == "accepted":
                    accepted_clips.append(x); accepted_meta.append(meta)
    con.commit()
    audit(con, "ingestion_complete", f"{len(accepted_meta)} accepted clips")

    probs = np.concatenate([predictor.predict_batch(accepted_clips[i:i + 256])
                            for i in range(0, len(accepted_clips), 256)])
    preds = pd.DataFrame(accepted_meta)[["clip_id", "house_id", "recorded_at"]]
    for i, c in enumerate(predictor.classes):
        preds[f"p_{c}"] = probs[:, i].round(4)
    preds["label"] = [predictor.classes[i] for i in probs.argmax(1)]
    preds["model_version"], preds["predicted_at"] = predictor.version, now()
    preds[["clip_id", "model_version", "p_healthy", "p_unhealthy", "p_noise", "label", "predicted_at"]] \
        .to_sql("predictions", con, if_exists="append", index=False)
    audit(con, "scoring_complete", f"model v{predictor.version} scored {len(preds)} clips")

    cards = pd.DataFrame(cards)
    cards.to_sql("record_cards", con, if_exists="append", index=False)
    daily = add_alerts(daily_sound_index(preds), cards)
    daily[["house_id", "date", "n_clips", "n_bird_clips", "sound_index", "baseline", "z_score",
           "water_change", "alert_level"]].to_sql("daily_index", con, if_exists="append", index=False)
    lead = evaluate_lead_time(daily, cards)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    lead.to_csv(PROCESSED / "lead_time_evaluation.csv", index=False)
    audit(con, "alerts_computed", lead.to_json(orient="records"))
    con.close()

    SAMPLES.mkdir(parents=True, exist_ok=True)                     # small demo clips for the app
    srng = np.random.default_rng(3)
    for lab in ["healthy", "unhealthy", "noise"]:
        save_wav(SAMPLES / f"sample_{lab}.wav", make_clip(lab, srng, severity=0.9))
    long = np.concatenate([make_clip(l, srng, 0.9) for l in ["healthy", "healthy", "noise", "unhealthy", "unhealthy", "healthy"]])
    save_wav(SAMPLES / "sample_30s_mixed.wav", long)
    print(lead.to_string(index=False))
    return lead


if __name__ == "__main__":
    run()
