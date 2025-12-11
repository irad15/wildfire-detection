from datetime import datetime
from typing import List

import numpy as np
from scipy.signal import savgol_filter  # scipy is allowed (open-source)


class DataProcessor:
    def __init__(self, raw_data: List[dict]):
        self.raw_data = raw_data

    def _print_comparison(self, before: List[dict], after: List[dict]) -> None:
        """Print a side-by-side comparison table of raw vs processed rows."""

        print("\n=== DataProcessor comparison (before vs after) ===")

        header = (
            f"+----+---------------------+---------------+---------------+---------------+-----------+\n"
            f"| #  | Timestamp           | Temp Raw      | Temp Smoothed | Smoke Raw     | Smoke Sm. |\n"
            f"+----+---------------------+---------------+---------------+---------------+-----------+"
        )
        print(header)

        max_len = max(len(before), len(after))

        for idx in range(max_len):
            if idx < len(before) and idx < len(after):
                b = before[idx]
                a = after[idx]

                row = (
                    f"| {idx+1:02d} "
                    f"| {str(b.get('timestamp')).ljust(19)} "
                    f"| {b.get('temperature'):>13.2f} "
                    f"| {a.get('smoothed_temp'):>13.2f} "
                    f"| {b.get('smoke'):>13.4f} "
                    f"| {a.get('smoothed_smoke'):>9.4f} |"
                )
                print(row)

        footer = (
            "\n+----+---------------------+---------------+---------------+---------------+-----------+"
        )
        print(footer)
        print("=== end comparison ===\n")


    def process(self) -> List[dict]:
        """Sort records, smooth temp/smoke using Savitzky-Golay filter, keep originals."""
        print("PROCESSING with Savitzky-Golay filter...")

        if not self.raw_data:
            return []

        # Sort by timestamp
        sorted_data = sorted(
            self.raw_data,
            key=lambda item: datetime.fromisoformat(item["timestamp"].replace("Z", "+00:00")),
        )

        temps = np.array([item["temperature"] for item in sorted_data], dtype=float)
        smokes = np.array([item["smoke"] for item in sorted_data], dtype=float)

        # Savitzky-Golay parameters
        window_length = 13
        polyorder = 2

        if len(temps) < window_length:
            smoothed_temps = temps.copy()
            smoothed_smokes = smokes.copy()
        else:
            smoothed_temps = savgol_filter(temps, window_length=window_length, polyorder=polyorder)
            smoothed_smokes = savgol_filter(smokes, window_length=window_length, polyorder=polyorder)

        # === FIX: Clip to physical bounds ===
        smoothed_temps = np.clip(smoothed_temps, 0.0, None)   # Temperature can't be negative
        smoothed_smokes = np.clip(smoothed_smokes, 0.0, 1.0) # Smoke strictly 0–1

        processed: List[dict] = []
        for item, smooth_temp, smooth_smoke in zip(sorted_data, smoothed_temps, smoothed_smokes):
            processed.append(
                {
                    **item,
                    "smoothed_temp": round(float(smooth_temp), 2),
                    "smoothed_smoke": round(float(smooth_smoke), 4),
                }
            )

        # Debug comparison
        self._print_comparison(sorted_data, processed)

        return processed