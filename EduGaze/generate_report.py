"""
Helper script to generate final_report.xlsx from an engagement_log.csv file.

Usage (from project root, with venv activated):

    python generate_report.py --csv engagement_log.csv --out final_report.xlsx
"""

import argparse
import os

from core.logger import EngagementLogger
from core.report import ReportGenerator


def main():
    parser = argparse.ArgumentParser(description="Generate final engagement report (Excel) from CSV log.")
    parser.add_argument(
        "--csv",
        type=str,
        default="engagement_log.csv",
        help="Path to engagement_log.csv (default: engagement_log.csv in project root)",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="final_report.xlsx",
        help="Output Excel report path (default: final_report.xlsx in project root)",
    )

    args = parser.parse_args()

    csv_path = os.path.abspath(args.csv)
    out_path = os.path.abspath(args.out)

    if not os.path.exists(csv_path):
        raise SystemExit(f"CSV file not found: {csv_path}")

    logger = EngagementLogger(csv_path)
    df = logger.to_dataframe()

    if df.empty:
        print("Warning: CSV is empty, report will contain no data.")

    report = ReportGenerator(df)
    report.export_excel(out_path)
    print(f"Final report written to: {out_path}")


if __name__ == "__main__":
    main()





