from datetime import datetime
from typing import List

import numpy as np
from scipy.signal import savgol_filter

from .config import (
    SAVITZKY_GOLAY_POLYORDER,
    SAVITZKY_GOLAY_WINDOW,
    TEMP_SPIKE_THRESHOLD,
    SMOKE_SPIKE_THRESHOLD,
)
from .models import DataPoint


class DataProcessor:
    """ 
    Handles input ordering and signal smoothing using Savitzky-Golay filter to reduce sensor noise.
    V2 adds spike suppression before smoothing.
    """

    @classmethod
    def process(cls, raw_data: List[DataPoint]) -> List[DataPoint]:
        """Main pipeline: sort → smooth → clip → build processed points."""

        return cls._run_pipeline(raw_data, suppress_spikes=False)

    @classmethod
    def process_v2(cls, raw_data: List[DataPoint]) -> List[DataPoint]:
        """V2 pipeline: adds spike suppression before smoothing."""

        return cls._run_pipeline(raw_data, suppress_spikes=True)

    @classmethod
    def _run_pipeline(cls, raw_data: List[DataPoint], suppress_spikes: bool) -> List[DataPoint]:
        """Shared pipeline used by V1/V2; optional spike suppression toggled via flag."""

        if not raw_data:
            return []

        # 1. Chronological order
        sorted_data = cls._sort_by_timestamp(raw_data)

        # 2. Extract raw signals
        temps, smokes = cls._extract_signals(sorted_data)

        # 3. Optional spike suppression
        if suppress_spikes:
            temps = cls._suppress_spikes(temps, TEMP_SPIKE_THRESHOLD)
            smokes = cls._suppress_spikes(smokes, SMOKE_SPIKE_THRESHOLD)

        # 4. Smooth both signals
        smoothed_temps = cls._smooth_signal(temps)
        smoothed_smokes = cls._smooth_signal(smokes)

        # 5. Clip to physical bounds (SG smoothing may introduce small negatives near spikes)
        smoothed_smokes = np.clip(smoothed_smokes, 0.0, 1.0)

        # 6. Build final objects
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

    @classmethod
    def _print_comparison(cls, before: List[DataPoint], after: List[DataPoint]) -> None:
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

    # -------------------------------------------------------------------------
    # V2 pipeline methods 
    # -------------------------------------------------------------------------
    @staticmethod
    def _suppress_spikes(signal: np.ndarray, threshold: float) -> np.ndarray:
        """
        Suppress isolated single-point spikes (both peaks and dips).

        A value is considered a spike if it is significantly higher than both
        neighbors or significantly lower than both neighbors by the given threshold.
        Edge values are left unchanged because they lack sufficient neighboring
        context to reliably distinguish sensor noise from real signal changes.
        """ 

        if len(signal) < 3:
            return signal

        fixed = signal.copy()

        for i in range(1, len(fixed) - 1):
            prev_val = fixed[i - 1]
            curr_val = fixed[i]
            next_val = fixed[i + 1]

            is_peak = (
                curr_val - prev_val > threshold and
                curr_val - next_val > threshold
            )

            is_dip = (
                prev_val - curr_val > threshold and
                next_val - curr_val > threshold
            )

            if is_peak or is_dip:
                fixed[i] = (prev_val + next_val) / 2.0

        return fixed
