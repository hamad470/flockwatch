"""End-to-end pipeline: data -> model -> monitoring -> alerts -> reports."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))
from flockwatch import make_dataset, train, monitoring, report_excel
from flockwatch.config import RAW

if __name__ == "__main__":
    if not (RAW / "metadata.csv").exists() and not any(RAW.glob("*/*.wav")):
        make_dataset.main()
    if "--skip-train" not in sys.argv:
        train.main()
    monitoring.run()
    print(report_excel.build("Farm A", "2026-04-12"))
    print(report_excel.build("Farm B", "2026-04-04"))   # week covering the H3/H4 outbreaks
