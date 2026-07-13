"""Buffers gaze/blink samples in memory and flushes to CSV/Excel on demand.

The previous implementation appended a single-row DataFrame to disk on every
sample (tens of times per second), which meant I/O cost grew linearly with
session length. Buffering in a list and writing once on export removes that
bottleneck entirely.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import datetime

import pandas as pd

from .heatmap import GazeHeatmap


@dataclass
class Sample:
    timestamp: str
    left_x: int | None
    left_y: int | None
    right_x: int | None
    right_y: int | None
    gaze: str
    blink: bool
    blink_count: int


@dataclass
class SessionLogger:
    output_dir: str
    output_prefix: str
    samples: list[Sample] = field(default_factory=list)

    def record(self, **kwargs) -> None:
        self.samples.append(Sample(**kwargs))

    def _paths(self) -> tuple[str, str, str]:
        os.makedirs(self.output_dir, exist_ok=True)
        base = os.path.join(self.output_dir, self.output_prefix)
        return f"{base}_raw_log.csv", f"{base}_log_and_heatmap.xlsx", f"{base}_heatmap.png"

    def export(self, heatmap: GazeHeatmap) -> dict[str, str]:
        csv_path, excel_path, heatmap_path = self._paths()
        df = pd.DataFrame(self.samples)

        df.to_csv(csv_path, index=False)

        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="raw_logs", index=False)
            pd.DataFrame(heatmap.raw()).to_excel(writer, sheet_name="heatmap_matrix", index=False)
            summary = pd.DataFrame(
                {
                    "recorded_at": [datetime.now().isoformat()],
                    "rows_logged": [len(self.samples)],
                    "total_blinks": [self.samples[-1].blink_count if self.samples else 0],
                }
            )
            summary.to_excel(writer, sheet_name="summary", index=False)

        heatmap.save_png(heatmap_path)

        return {"csv": csv_path, "excel": excel_path, "heatmap_png": heatmap_path}
