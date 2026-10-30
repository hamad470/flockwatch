"""
Turn clip-level predictions into a daily Respiratory Sound Index per house and raise alerts
when it rises well above that house's own recent baseline. Water intake is used as a
second, independent signal to corroborate alerts.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from .config import BASELINE_DAYS, Z_THRESHOLD, MIN_INDEX_FOR_ALERT


def daily_sound_index(preds: pd.DataFrame) -> pd.DataFrame:
    """preds: clip_id, house_id, recorded_at, label, p_unhealthy (accepted clips only)."""
    p = preds.copy()
    p["date"] = pd.to_datetime(p.recorded_at).dt.date.astype(str)
    birds = p[p.label != "noise"]                              # ignore clips dominated by farm noise
    idx = birds.groupby(["house_id", "date"]).p_unhealthy.mean().rename("sound_index")
    counts = p.groupby(["house_id", "date"]).size().rename("n_clips")
    bird_counts = birds.groupby(["house_id", "date"]).size().rename("n_bird_clips")
    return pd.concat([counts, bird_counts, idx], axis=1).reset_index().fillna({"n_bird_clips": 0})


def add_alerts(daily: pd.DataFrame, cards: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Robust, self-calibrating alert rule per house:
      baseline = median of the last BASELINE_DAYS *normal* days (alert days are excluded, so an
                 ongoing outbreak cannot drag the baseline up and switch its own alert off)
      scale    = max(1.4826 * MAD, 0.03)   (robust standard deviation with a floor)
      score    = (sound_index - baseline) / scale
      Amber    = score >= Z_THRESHOLD and sound_index >= MIN_INDEX_FOR_ALERT
      Red      = Amber and (sound_index >= 2 * MIN_INDEX_FOR_ALERT or water intake down >= 8%)
    """
    d = daily.sort_values(["house_id", "date"]).copy()
    if cards is not None:
        c = cards.sort_values(["house_id", "date"]).copy()
        c["water_per_bird"] = c.water_l / c.birds_alive
        c["expected"] = c.groupby("house_id").water_per_bird.transform(
            lambda s: s.shift(1).rolling(3, min_periods=2).mean() * 1.03)   # birds drink ~3% more each day
        c["water_change"] = c.water_per_bird / c.expected - 1
        d = d.merge(c[["house_id", "date", "water_change"]], on=["house_id", "date"], how="left")
    else:
        d["water_change"] = np.nan
    out = []
    for _, g in d.groupby("house_id", sort=False):
        normal: list[float] = []
        for _, r in g.iterrows():
            r = r.copy()
            hist = np.array(normal[-BASELINE_DAYS:])
            if pd.isna(r.sound_index):
                r["baseline"], r["z_score"], r["alert_level"] = np.nan, np.nan, "No data"
            elif len(hist) < 4:
                r["baseline"], r["z_score"], r["alert_level"] = np.nan, np.nan, "Green"
                normal.append(r.sound_index)
            else:
                base = float(np.median(hist))
                scale = max(1.4826 * float(np.median(np.abs(hist - base))), 0.03)
                z = (r.sound_index - base) / scale
                amber = z >= Z_THRESHOLD and r.sound_index >= MIN_INDEX_FOR_ALERT
                water_drop = pd.notna(r.water_change) and r.water_change <= -0.08
                red = amber and (r.sound_index >= 2 * MIN_INDEX_FOR_ALERT or water_drop)
                r["baseline"], r["z_score"] = base, z
                r["alert_level"] = "Red" if red else "Amber" if amber else "Green"
                if not amber:
                    normal.append(r.sound_index)
            out.append(r)
    return pd.DataFrame(out).reset_index(drop=True)


def evaluate_lead_time(alerts: pd.DataFrame, cards: pd.DataFrame) -> pd.DataFrame:
    """Compare first alert date with the first day mortality exceeds 2x its 7-day baseline."""
    c = cards.sort_values(["house_id", "date"]).copy()
    base = c.groupby("house_id").mortality.transform(lambda s: s.shift(1).rolling(7, min_periods=4).mean())
    c["spike"] = (c.mortality > 2 * base) & (c.age_days >= 10)     # ignore first-week deaths
    rows = []
    for h in sorted(c.house_id.unique()):
        spikes = c[(c.house_id == h) & c.spike].date
        al = alerts[(alerts.house_id == h) & alerts.alert_level.isin(["Amber", "Red"])].date
        first_spike = spikes.min() if len(spikes) else None
        first_alert = al.min() if len(al) else None
        lead = (pd.Timestamp(first_spike) - pd.Timestamp(first_alert)).days if first_spike and first_alert else None
        rows.append({"house_id": h, "first_alert": first_alert, "first_mortality_spike": first_spike,
                     "lead_time_days": lead, "alert_days": int(len(al))})
    return pd.DataFrame(rows)
