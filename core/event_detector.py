from typing import List
import math

import numpy as np

from .models import Event, EventsSummary, DataPoint


class EventDetector:
    """Detects suspicious fire events from processed sensor data."""

    # Constants for anomaly scoring
    TEMP_WEIGHT = 60
    SMOKE_WEIGHT = 60
    WIND_BASE_WEIGHT = 15

    WIND_MULTIPLIER_DIVISOR = 15.0

    ALERT_THRESHOLD = 70

    TEMP_PIVOT = 4.0
    TEMP_STEEPNESS = 3.0

    SMOKE_PIVOT = 0.02      # tuned to typical smoke std
    SMOKE_STEEPNESS = 20.0  # sharper transition for small std

    WIND_PIVOT = 6.0        # m/s
    WIND_STEEPNESS = 0.8


    @classmethod
    def detect(cls, processed_data: List[DataPoint]) -> EventsSummary:
        """Main pipeline: compute statistics → score each point → build summary."""
        if not processed_data:
            return EventsSummary(events=[], event_count=0, max_score=0.0)

        # Score each point (internal statistics + damping)
        events, max_score = cls._score_points(processed_data)

        # Optional debug output
        cls._print_debug_scores(processed_data, events=events, max_score=max_score)

        return EventsSummary(
            events=events,
            event_count=len(events),
            max_score=round(max_score, 1) if events or max_score > 0 else 0.0,
        )


    @staticmethod
    def _score_points(data: List[DataPoint]) -> tuple[List[Event], float]:
        """Score each processed data point, compute damping internally."""
        events: List[Event] = []
        max_score = 0.0

        if not data:
            return events, max_score

        # 1. Extract signals and compute statistics
        temps = np.array([dp.temperature for dp in data], dtype=float)
        smokes = np.array([dp.smoke for dp in data], dtype=float)
        mean_temp, std_temp = np.mean(temps), np.std(temps, ddof=1)
        mean_smoke, std_smoke = np.mean(smokes), np.std(smokes, ddof=1)

        # 2. Compute dynamic damping
        temp_damping = EventDetector._dynamic_damping(std_temp, EventDetector.TEMP_PIVOT, EventDetector.TEMP_STEEPNESS)
        smoke_damping = EventDetector._dynamic_damping(std_smoke, EventDetector.SMOKE_PIVOT, EventDetector.SMOKE_STEEPNESS)

        # 3. Score each point
        for dp in data:
            t_z = (dp.temperature - mean_temp) / std_temp
            s_z = (dp.smoke - mean_smoke) / std_smoke

            temp_severity = EventDetector._z_to_severity(t_z)
            smoke_severity = EventDetector._z_to_severity(s_z)

            temp_anomaly = temp_severity * temp_damping
            smoke_anomaly = smoke_severity * smoke_damping

            wind_score = EventDetector._wind_to_score(dp.wind, EventDetector.WIND_PIVOT, EventDetector.WIND_STEEPNESS)

            risk_score = (
                EventDetector.TEMP_WEIGHT * temp_anomaly +
                EventDetector.SMOKE_WEIGHT * smoke_anomaly +
                EventDetector.WIND_BASE_WEIGHT * wind_score
            )

            risk_score = max(0.0, min(100.0, risk_score))
            max_score = max(max_score, risk_score)

            if risk_score > EventDetector.ALERT_THRESHOLD:
                events.append(Event(timestamp=dp.timestamp, score=round(risk_score, 1)))

        return events, max_score


    # -------------------------------------------------------------------------
    # Helper methods
    # -------------------------------------------------------------------------

    @staticmethod
    def _dynamic_damping(std: float, pivot: float, steepness: float) -> float:
        """Sigmoid damping — low std → low weight, high std → full weight."""
        return 1.0 / (1.0 + np.exp(steepness * (pivot - std)))

    @staticmethod
    def _wind_to_score(wind: float, pivot: float, steepness: float) -> float:
        """Sigmoid mapping of wind speed to bounded [0,1] risk contribution."""
        return 1.0 / (1.0 + np.exp(-steepness * (wind - pivot)))

    @staticmethod
    def _z_to_severity(z: float) -> float:
        """
        One-sided CDF-based severity score.

        Converts a positive z-score into a bounded anomaly severity in [0, 1].

        This is not a standard z → p normalization. Instead of measuring statistical
        likelihood, it measures anomaly *severity* for risk scoring.

        Key properties:
        - Only positive deviations matter (z <= 0 → severity = 0)
        - Mean maps to 0 (not 0.5 like a normal CDF)
        - Designed for weighted scoring systems (e.g. weights up to 100)

        Intuition:
        - z = 0.0  → severity = 0.00 (normal)
        - z ≈ 1.0 → severity ≈ 0.68 (moderate anomaly)
        - z ≈ 2.0 → severity ≈ 0.95 (strong anomaly)
        - z ≥ 3.0 → severity ≈ 1.00 (extreme anomaly)
        """

        if z <= 0.0:
            return 0.0

        cdf = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

        # Re-center and stretch upper half of the CDF:
        # Φ(z) ∈ [0.5, 1.0]  →  severity ∈ [0.0, 1.0]
        return min(1.0, max(0.0, 2.0 * (cdf - 0.5)))

    @classmethod
    def _print_debug_scores(
        cls,
        data: List[DataPoint],
        events: List[Event],
        max_score: float,
    ) -> None:
        """Rich debug output — mirrors scoring logic exactly, damping computed internally."""

        if not data:
            print("No data to print.")
            return

        # Compute statistics
        temps = np.array([dp.temperature for dp in data], dtype=float)
        smokes = np.array([dp.smoke for dp in data], dtype=float)
        mean_temp, std_temp = np.mean(temps), np.std(temps, ddof=1)
        mean_smoke, std_smoke = np.mean(smokes), np.std(smokes, ddof=1)

        # Compute dynamic damping
        temp_damping = cls._dynamic_damping(std_temp, cls.TEMP_PIVOT, cls.TEMP_STEEPNESS)
        smoke_damping = cls._dynamic_damping(std_smoke, cls.SMOKE_PIVOT, cls.SMOKE_STEEPNESS)

        print("\n" + "═" * 170)
        print("DEBUG: Event Detection Summary")
        print(
            f"Mean temp: {mean_temp:6.2f}°C | Std temp: {std_temp:8.4f} | Temp damping: {temp_damping:5.3f}\n"
            f"Mean smoke: {mean_smoke:8.4f} | Std smoke: {std_smoke:10.6f} | Smoke damping: {smoke_damping:5.3f}\n"
            f"Total points: {len(data)} | Events detected: {len(events)} | Max score: {max_score:5.1f}"
        )
        print("─" * 170)

        print(
            f"{'idx':>3} | {'timestamp':^19} | "
            f"{'temp':>7} {'t_z':>7} {'t_sev':>7} {'t_dmp':>7} | "
            f"{'smoke':>8} {'s_z':>7} {'s_sev':>7} {'s_dmp':>7} | "
            f"{'wind':>5} {'w_scr':>6} | "
            f"{'risk':>8} {'status':>8}"
        )
        print("─" * 170)

        for idx, dp in enumerate(data, 1):
            # Z-scores
            t_z = (dp.temperature - mean_temp) / std_temp
            s_z = (dp.smoke - mean_smoke) / std_smoke

            # Undamped severities
            t_sev = cls._z_to_severity(t_z)
            s_sev = cls._z_to_severity(s_z)

            # Damped severities
            t_dmp = t_sev * temp_damping
            s_dmp = s_sev * smoke_damping

            # Wind contribution (bounded 0–1)
            w_scr = cls._wind_to_score(dp.wind, cls.WIND_PIVOT, cls.WIND_STEEPNESS)

            # Final risk score
            risk = (
                cls.TEMP_WEIGHT * t_dmp +
                cls.SMOKE_WEIGHT * s_dmp +
                cls.WIND_BASE_WEIGHT * w_scr
            )
            risk = max(0.0, min(100.0, risk))

            status = "ALERT" if risk > cls.ALERT_THRESHOLD else ""

            print(
                f"{idx:03d} | {dp.timestamp[-19:]:19} | "
                f"{dp.temperature:7.2f} {t_z:7.2f} {t_sev:7.3f} {t_dmp:7.3f} | "
                f"{dp.smoke:8.4f} {s_z:7.2f} {s_sev:7.3f} {s_dmp:7.3f} | "
                f"{dp.wind:5.1f} {w_scr:6.3f} | "
                f"{risk:8.1f} {status:>8}"
            )

        print("═" * 170 + "\n")


