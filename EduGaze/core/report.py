from dataclasses import dataclass
from typing import Dict

import pandas as pd


@dataclass
class EngagementSummary:
    per_student: pd.DataFrame
    highest_student: str
    lowest_student: str


class ReportGenerator:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def summarize(self) -> EngagementSummary:
        if self.df.empty:
            return EngagementSummary(
                per_student=pd.DataFrame(),
                highest_student="",
                lowest_student="",
            )
        grouped = self.df.groupby("student_name")["engagement_score"].mean().reset_index()
        grouped = grouped.sort_values("engagement_score", ascending=False)
        highest = grouped.iloc[0]["student_name"]
        lowest = grouped.iloc[-1]["student_name"]
        return EngagementSummary(per_student=grouped, highest_student=highest, lowest_student=lowest)

    def export_excel(self, path: str):
        summary = self.summarize()
        with pd.ExcelWriter(path) as writer:
            self.df.to_excel(writer, sheet_name="Raw Log", index=False)
            summary.per_student.to_excel(writer, sheet_name="Per-Student Averages", index=False)
            # Simple metadata sheet
            meta = pd.DataFrame(
                [
                    {"metric": "highest_engagement_student", "value": summary.highest_student},
                    {"metric": "lowest_engagement_student", "value": summary.lowest_student},
                ]
            )
            meta.to_excel(writer, sheet_name="Summary", index=False)





