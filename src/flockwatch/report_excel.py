"""
Weekly stakeholder report (Excel). One workbook per farm with:
  Summary (house status, traffic lights), Alerts, Record cards, Data quality.
Open it and run the VBA macro in excel/WeeklyReportMacros.bas for extra formatting.
"""
from __future__ import annotations
import argparse
import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils.dataframe import dataframe_to_rows
from .config import ROOT
from .storage import connect

FILL = {"Red": "F8D7DA", "Amber": "FFE8CC", "Green": "D4EDDA", "No data": "E2E6EA"}


def _sheet(wb, title, df):
    ws = wb.create_sheet(title)
    for r in dataframe_to_rows(df, index=False, header=True):
        ws.append(r)
    for c in ws[1]:
        c.font = Font(bold=True, color="FFFFFF"); c.fill = PatternFill("solid", fgColor="1F3864")
        c.alignment = Alignment(wrap_text=True)
    for col in ws.columns:
        ws.column_dimensions[col[0].column_letter].width = max(12, min(40, max(len(str(c.value or "")) for c in col) + 2))
    if "alert_level" in df.columns:
        j = list(df.columns).index("alert_level") + 1
        for row in ws.iter_rows(min_row=2, min_col=j, max_col=j):
            for c in row:
                if c.value in FILL:
                    c.fill = PatternFill("solid", fgColor=FILL[c.value])
    ws.freeze_panes = "A2"
    return ws


def build(farm: str = "Farm B", week_ending: str | None = None) -> str:
    con = connect()
    houses = pd.read_sql("SELECT * FROM houses WHERE farm = ?", con, params=(farm,))
    ids = tuple(houses.house_id)
    q = f"({','.join('?' * len(ids))})"
    daily = pd.read_sql(f"SELECT * FROM daily_index WHERE house_id IN {q}", con, params=ids)
    cards = pd.read_sql(f"SELECT * FROM record_cards WHERE house_id IN {q}", con, params=ids)
    rec = pd.read_sql(f"SELECT house_id, status, reason FROM recordings WHERE house_id IN {q}", con, params=ids)
    end = pd.Timestamp(week_ending) if week_ending else pd.to_datetime(daily.date).max()
    start = end - pd.Timedelta(days=6)
    in_week = lambda d: pd.to_datetime(d.date).between(start, end)
    dw, cw = daily[in_week(daily)], cards[in_week(cards)]

    summary = []
    for h in ids:
        d, c = dw[dw.house_id == h], cw[cw.house_id == h]
        worst = "Red" if (d.alert_level == "Red").any() else "Amber" if (d.alert_level == "Amber").any() else "Green"
        summary.append({"house_id": h, "alert_level": worst, "red_or_amber_days": int(d.alert_level.isin(["Red", "Amber"]).sum()),
                        "mean_sound_index": round(d.sound_index.mean(), 3), "mortality_week": int(c.mortality.sum()),
                        "birds_alive_end": int(c.birds_alive.iloc[-1]) if len(c) else None,
                        "avg_weight_g_end": int(c.avg_weight_g.iloc[-1]) if len(c) else None})
    dq = rec.groupby("house_id").status.value_counts().unstack(fill_value=0).reset_index()
    dq["acceptance_rate"] = (dq.get("accepted", 0) / dq[["accepted", "quarantined"]].sum(axis=1)).round(3)
    reasons = rec[rec.status == "quarantined"].groupby(["house_id", "reason"]).size().rename("clips").reset_index()

    wb = Workbook(); wb.remove(wb.active)
    ws = _sheet(wb, "Summary", pd.DataFrame(summary))
    ws.insert_rows(1, 2)
    ws["A1"] = f"FlockWatch weekly report - {farm} - {start:%d %b} to {end:%d %b %Y}"
    ws["A1"].font = Font(bold=True, size=14, color="1F3864")
    ws["A2"] = "Sound index = average probability that bird sounds are respiratory-type. Alerts compare each house with its own recent normal days."
    _sheet(wb, "Alerts", dw[dw.alert_level.isin(["Red", "Amber"])][["house_id", "date", "alert_level", "sound_index", "baseline", "z_score", "water_change"]].round(3))
    _sheet(wb, "Record cards", cw)
    _sheet(wb, "Data quality", dq)
    _sheet(wb, "Quarantine reasons", reasons)
    out = ROOT / "reports" / f"weekly_report_{farm.replace(' ', '_')}_{end:%Y%m%d}.xlsx"
    out.parent.mkdir(parents=True, exist_ok=True)
    wb.save(out)
    con.close()
    return str(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--farm", default="Farm B"); ap.add_argument("--week-ending")
    a = ap.parse_args(); print(build(a.farm, a.week_ending))
