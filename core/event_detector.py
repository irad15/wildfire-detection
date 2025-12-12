from typing import List

import numpy as np

from .models import Event, EventsSummary, DataPoint


class EventDetector:
    """Detects suspicious fire events from processed sensor data."""

    # Constants for anomaly scoring
    TEMP_WEIGHT = 40
    SMOKE_WEIGHT = 50
    WIND_BASE_WEIGHT = 10
    WIND_MULTIPLIER_DIVISOR = 15.0
    ALERT_THRESHOLD = 70

    PIVOT = 0.6
    STEEPNESS = 6.0
    REFERENCE_NOISE = 0.008
    POWER = 8



    # -------------------------------------------------------------------------
    # Main public method
    # -------------------------------------------------------------------------

    @classmethod
    def detect(cls, processed_data: List[DataPoint]) -> EventsSummary:
        """Main pipeline: compute statistics → score each point → build summary."""
        if not processed_data:
            return EventsSummary(events=[], event_count=0, max_score=0.0)

        # 1. Extract signals and compute global statistics
        temps, smokes, winds = cls._extract_signals(processed_data)
        mean_temp, std_temp = cls._compute_stats(temps)
        mean_smoke, std_smoke = cls._compute_stats(smokes)

        # 2. Dynamic damping based on variance (no hardcoded thresholds)
        temp_damping = cls._dynamic_damping(std_temp, cls.PIVOT, cls.STEEPNESS)
        smoke_damping = cls._dynamic_power_damping(std_smoke, cls.REFERENCE_NOISE, cls.POWER)

        # 3. Score each point and collect events
        events, max_score = cls._score_points(
            processed_data, mean_temp, std_temp, mean_smoke, std_smoke,
            temp_damping, smoke_damping
        )

        # Optional debug output (comment out before submission)
        cls._print_debug_scores(
            processed_data, temps, smokes, winds, mean_temp, std_temp, mean_smoke, std_smoke,
            temp_damping, smoke_damping, events, max_score
        )

        return EventsSummary(
            events=events,
            event_count=len(events),
            max_score=round(max_score, 1) if events or max_score > 0 else 0.0,
        )

    # -------------------------------------------------------------------------
    # Helper methods (private, static)
    # -------------------------------------------------------------------------

    @staticmethod
    def _extract_signals(data: List[DataPoint]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract temperature, smoke, and wind arrays."""
        temps = np.array([dp.temperature for dp in data], dtype=float)
        smokes = np.array([dp.smoke for dp in data], dtype=float)
        winds = np.array([dp.wind for dp in data], dtype=float)
        return temps, smokes, winds

    @staticmethod
    def _compute_stats(signal: np.ndarray) -> tuple[float, float]:
        """Compute mean and standard deviation with fallback for zero std."""
        mean = np.mean(signal)
        std = np.std(signal) or 1.0
        return mean, std

    @staticmethod
    def _dynamic_damping(std: float, pivot: float, steepness: float) -> float:
        """Sigmoid damping — low std → low weight, high std → full weight."""
        return 1.0 / (1.0 + np.exp(steepness * (pivot - std)))

    @staticmethod
    def _dynamic_power_damping(std: float, reference_noise: float, power: float) -> float:
        """Power-law damping — ultra-low std → near-zero weight (fully dynamic)."""
        ratio = std / reference_noise
        damping = ratio ** power
        return min(1.0, max(0.0, damping))

    @staticmethod
    def _score_points(
        data: List[DataPoint],
        mean_temp: float, std_temp: float,
        mean_smoke: float, std_smoke: float,
        temp_damping: float, smoke_damping: float,
    ) -> tuple[List[Event], float]:
        """Score each data point and collect events above threshold."""
        events: List[Event] = []
        max_score = 0.0

        for dp in data:
            t_z = max(0.0, (dp.temperature - mean_temp) / std_temp)
            s_z = max(0.0, (dp.smoke - mean_smoke) / std_smoke)

            # Damping the z scores:
            temp_anomaly = t_z * temp_damping
            smoke_anomaly = s_z * smoke_damping

            wind_mult = 1.0 + dp.wind / EventDetector.WIND_MULTIPLIER_DIVISOR
            risk_score = min(100.0, max(0.0,
                EventDetector.TEMP_WEIGHT * temp_anomaly +
                EventDetector.SMOKE_WEIGHT * smoke_anomaly +
                EventDetector.WIND_BASE_WEIGHT * (dp.wind / 2)
            ) * wind_mult)

            if risk_score > max_score:
                max_score = risk_score
            if risk_score > EventDetector.ALERT_THRESHOLD:
                events.append(Event(timestamp=dp.timestamp, score=round(risk_score, 1)))

        return events, max_score

    @staticmethod
    def _print_debug_scores(
        data: List[DataPoint],
        temps: np.ndarray, smokes: np.ndarray, winds: np.ndarray,
        mean_temp: float, std_temp: float, mean_smoke: float, std_smoke: float,
        temp_damping: float, smoke_damping: float,
        events: List[Event], max_score: float,
    ) -> None:
        """Rich debug output — comment out before submission."""
        print("\n" + "═" * 130)
        print("DEBUG: Event Detection Summary")
        print(f"Mean temp: {mean_temp:6.2f}°C | Std temp: {std_temp:6.4f}°C | Temp damping: {temp_damping:5.3f}")
        print(f"Mean smoke: {mean_smoke:6.4f} | Std smoke: {std_smoke:8.6f} | Smoke damping: {smoke_damping:5.3f}")
        print(f"Total points: {len(data)} | Events detected: {len(events)} | Max score: {max_score:5.1f}")
        print("─" * 130)
        print(f"{'idx':>3} | {'timestamp':^19} | {'temp':>7} | {'t_z':>8} | {'smoke':>9} | {'s_z':>9} | {'wind':>5} | {'risk':>8} {'status':>8}")
        print("─" * 130)

        for idx, dp in enumerate(data, 1):
            t_z = max(0.0, (dp.temperature - mean_temp) / std_temp)
            s_z = max(0.0, (dp.smoke - mean_smoke) / std_smoke)
            temp_anom = t_z * temp_damping
            smoke_anom = s_z * smoke_damping
            wind_mult = 1.0 + dp.wind / EventDetector.WIND_MULTIPLIER_DIVISOR
            risk = min(100.0, max(0.0,
                40 * temp_anom + 50 * smoke_anom + 10 * (dp.wind / 2)
            ) * wind_mult)
            status = "ALERT" if risk > 70 else ""
            print(
                f"{idx:03d} | {dp.timestamp[-19:]:19} | "
                f"{dp.temperature:7.2f} | {t_z:8.3f} | "
                f"{dp.smoke:9.4f} | {s_z:9.3f} | "
                f"{dp.wind:5.1f} | {risk:8.1f} {status:>8}"
            )
        print("═" * 130 + "\n")