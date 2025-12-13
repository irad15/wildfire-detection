from datetime import datetime
from typing import List

import numpy as np
from scipy.signal import savgol_filter

from .config import SAVITZKY_GOLAY_POLYORDER, SAVITZKY_GOLAY_WINDOW
from .models import DataPoint


class DataProcessor:
    """Processes raw sensor data: sorts, smooths signals, enforces physical bounds."""


    @classmethod
    def process(cls, raw_data: List[DataPoint]) -> List[DataPoint]:
        """Main pipeline: sort → smooth → clip → build processed points."""

        if not raw_data:
            return []

        # 1. Chronological order
        sorted_data = cls._sort_by_timestamp(raw_data)

        # 2. Extract raw signals
        temps, smokes = cls._extract_signals(sorted_data)

        # 3. Smooth both signals
        smoothed_temps = cls._smooth_signal(temps)
        smoothed_smokes = cls._smooth_signal(smokes)

        smoothed_smokes = np.clip(smoothed_smokes, 0, 1.0) 

        # 5. Build final objects
        processed = cls._build_processed_points(sorted_data, smoothed_temps, smoothed_smokes)

        # Optional debug output
        cls._print_comparison(raw_data, processed)

        return processed

    # -------------------------------------------------------------------------
    # Helper methods 
    # -------------------------------------------------------------------------
    @staticmethod
    def _sort_by_timestamp(raw_data: List[DataPoint]) -> List[DataPoint]:
        """Sort data points chronologically by ISO-8601 timestamp."""
        return sorted(
            raw_data,
            key=lambda dp: datetime.fromisoformat(dp.timestamp.replace("Z", "+00:00")),
        )

    @staticmethod
    def _extract_signals(data_points: List[DataPoint]) -> tuple[np.ndarray, np.ndarray]:
        """Extract temperature and smoke as NumPy arrays for vectorized processing."""
        temps = np.array([dp.temperature for dp in data_points], dtype=float)
        smokes = np.array([dp.smoke for dp in data_points], dtype=float)
        return temps, smokes

    @staticmethod
    def _smooth_signal(signal: np.ndarray) -> np.ndarray:
        """Apply Savitzky-Golay filter; ignore if data too short."""
        if len(signal) < SAVITZKY_GOLAY_WINDOW:
            return signal
        return savgol_filter(
            signal,
            window_length=SAVITZKY_GOLAY_WINDOW,
            polyorder=SAVITZKY_GOLAY_POLYORDER,
        )

    @staticmethod
    def _build_processed_points(
        original: List[DataPoint],
        smoothed_temps: np.ndarray,
        smoothed_smokes: np.ndarray,
    ) -> List[DataPoint]:
        """Construct final DataPoint objects with rounded smoothed values."""
        processed = []
        for dp, s_temp, s_smoke in zip(original, smoothed_temps, smoothed_smokes):
            processed.append(
                DataPoint(
                    timestamp=dp.timestamp,
                    temperature=round(float(s_temp), 2),
                    smoke=round(float(s_smoke), 4),
                    wind=dp.wind,
                )
            )
        return processed

    @staticmethod
    def _print_comparison(before: List[DataPoint], after: List[DataPoint]) -> None:
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
                    f"| {str(b.timestamp).ljust(19)} "
                    f"| {b.temperature:>13.2f} "
                    f"| {a.temperature:>13.2f} "
                    f"| {b.smoke:>13.4f} "
                    f"| {a.smoke:>9.4f} |"
                )
                print(row)

        footer = (
            "\n+----+---------------------+---------------+---------------+---------------+-----------+"
        )
        print(footer)
        print("=== end comparison ===\n")