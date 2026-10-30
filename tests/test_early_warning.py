import numpy as np
import pandas as pd
from flockwatch.early_warning import add_alerts


def series(values, house="H1"):
    dates = pd.date_range("2026-03-01", periods=len(values)).date.astype(str)
    return pd.DataFrame({"house_id": house, "date": dates, "n_clips": 24, "n_bird_clips": 20, "sound_index": values})


def test_no_alerts_on_stable_noisy_signal():
    rng = np.random.default_rng(0)
    d = add_alerts(series(np.clip(rng.normal(0.1, 0.03, 30), 0, 1)))
    assert not d.alert_level.isin(["Amber", "Red"]).any()


def test_outbreak_detected_and_alert_persists():
    values = [0.1] * 15 + [0.5, 0.6, 0.7, 0.7] + [0.1] * 5
    d = add_alerts(series(values))
    lv = d.alert_level.tolist()
    assert lv[14] == "Green" and all(l in ("Amber", "Red") for l in lv[15:19])
    assert lv[-1] == "Green"
