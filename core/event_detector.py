from typing import List
import math

import numpy as np

from .config import (
    ALERT_THRESHOLD,
    HYSTERESIS_RESET_THRESHOLD,
    SMOKE_PIVOT,
    SMOKE_STEEPNESS,
    SMOKE_WEIGHT,
    TEMP_PIVOT,
    TEMP_STEEPNESS,
    TEMP_WEIGHT,
    WIND_BASE_WEIGHT,
    WIND_PIVOT,
    WIND_STEEPNESS,
)
from .models import Event, EventsSummary, DataPoint


class EventDetector:
    """Detects suspicious fire events from processed sensor data."""


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

    @classmethod
    def _score_points(cls, data: List[DataPoint]) -> tuple[List[Event], float]:
        """Score each data point using variance-aware anomaly detection."""

        events: List[Event] = []
        max_score = 0.0

        if not data:
            return events, max_score

        # 1. Extract signals and compute statistics
        temps = np.array([dp.temperature for dp in data], dtype=float)
        smokes = np.array([dp.smoke for dp in data], dtype=float)

        mean_temp = np.mean(temps)
        mean_smoke = np.mean(smokes)

        # Prevent division by zero in z-score calculation
        EPS = 1e-6
        std_temp = max(np.std(temps, ddof=1), EPS)
        std_smoke = max(np.std(smokes, ddof=1), EPS)

        # 2. Compute dynamic damping
        temp_damping = cls._dynamic_damping(std_temp, TEMP_PIVOT, TEMP_STEEPNESS)
        smoke_damping = cls._dynamic_damping(std_smoke, SMOKE_PIVOT, SMOKE_STEEPNESS)

        # 3. Score each point in the processed data
        for dp in data:
            # Compute positive-only z-scores (only above mean considered abnormal)
            t_z = max(0.0, (dp.temperature - mean_temp) / std_temp)
            s_z = max(0.0, (dp.smoke - mean_smoke) / std_smoke)

            # Convert z-scores to severity (bounded [0,1] using CDF)
            temp_severity = cls._z_to_severity(t_z)
            smoke_severity = cls._z_to_severity(s_z)

            # Apply variance-aware damping factors to each signal
            temp_anomaly = temp_severity * temp_damping
            smoke_anomaly = smoke_severity * smoke_damping

            # Map wind value to score contribution [0, 1]
            wind_score = cls._wind_to_score(dp.wind, WIND_PIVOT, WIND_STEEPNESS)

            # Calculate total risk score (weighted sum of signals)
            risk_score = (
                TEMP_WEIGHT * temp_anomaly +
                SMOKE_WEIGHT * smoke_anomaly +
                WIND_BASE_WEIGHT * wind_score
            )

            # Ensure bounded in [0,100]
            risk_score = max(0.0, min(100.0, risk_score))
            max_score = max(max_score, risk_score)

            # If score exceeds alert threshold, mark as suspicious event
            if risk_score > ALERT_THRESHOLD:
                events.append(Event(timestamp=dp.timestamp, score=round(risk_score, 1)))

        return events, max_score


    # same as detect but uses _score_points_v2 + _print_debug_scores_v2
    @classmethod
    def detect_v2(cls, processed_data: List[DataPoint]) -> EventsSummary:
        """V2 pipeline: same scoring, but with hysteresis to avoid repeat alerts."""
        if not processed_data:
            return EventsSummary(events=[], event_count=0, max_score=0.0)

        # Score each point (internal statistics + damping)
        events, max_score = cls._score_points_v2(processed_data)

        # Optional debug output
        cls._print_debug_scores_v2(processed_data, events=events, max_score=max_score)

        return EventsSummary(
            events=events,
            event_count=len(events),
            max_score=round(max_score, 1) if events or max_score > 0 else 0.0,
        )

    @classmethod
    def _score_points_v2(cls, data: List[DataPoint]) -> tuple[List[Event], float]:
        """Score each data point with alert hysteresis to suppress repeated alerts."""

        events: List[Event] = []
        max_score = 0.0

        if not data:
            return events, max_score

        # 1. Extract signals and compute statistics
        temps = np.array([dp.temperature for dp in data], dtype=float)
        smokes = np.array([dp.smoke for dp in data], dtype=float)

        mean_temp = np.mean(temps)
        mean_smoke = np.mean(smokes)

        EPS = 1e-6
        std_temp = max(np.std(temps, ddof=1), EPS)
        std_smoke = max(np.std(smokes, ddof=1), EPS)

        # 2. Compute dynamic damping
        temp_damping = cls._dynamic_damping(std_temp, TEMP_PIVOT, TEMP_STEEPNESS)
        smoke_damping = cls._dynamic_damping(std_smoke, SMOKE_PIVOT, SMOKE_STEEPNESS)

        # Track whether we are already inside an active incident (V2)
        in_active_incident = False

        # 3. Score each point
        for dp in data:
            # Compute positive-only z-scores for temperature and smoke
            t_z = max(0.0, (dp.temperature - mean_temp) / std_temp)
            s_z = max(0.0, (dp.smoke - mean_smoke) / std_smoke)

            # Convert z-scores to bounded severity values
            temp_severity = cls._z_to_severity(t_z)
            smoke_severity = cls._z_to_severity(s_z)

            # Apply variance-based damping to reduce sensitivity in stable environments
            temp_anomaly = temp_severity * temp_damping
            smoke_anomaly = smoke_severity * smoke_damping

            # Compute wind contribution independently
            wind_score = cls._wind_to_score(dp.wind, WIND_PIVOT, WIND_STEEPNESS)

            # Combine all signal contributions into a final risk score
            risk_score = (
                TEMP_WEIGHT * temp_anomaly +
                SMOKE_WEIGHT * smoke_anomaly +
                WIND_BASE_WEIGHT * wind_score
            )

            # Clamp risk score to a valid range
            risk_score = max(0.0, min(100.0, risk_score))
            max_score = max(max_score, risk_score)

            # --- Hysteresis logic (V2) ---
            if not in_active_incident and risk_score > ALERT_THRESHOLD:
                # Trigger a single alert at the start of a high-risk incident
                events.append(Event(timestamp=dp.timestamp, score=round(risk_score, 1)))
                in_active_incident = True

            elif in_active_incident and risk_score < HYSTERESIS_RESET_THRESHOLD:
                # Risk dropped sufficiently; allow future alerts for a new incident
                in_active_incident = False

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
        """Rich debug output (V1) — mirrors scoring logic exactly, damping computed internally."""

        if not data:
            print("No data to print.")
            return

        # Compute statistics
        temps = np.array([dp.temperature for dp in data], dtype=float)
        smokes = np.array([dp.smoke for dp in data], dtype=float)
        mean_temp, std_temp = np.mean(temps), np.std(temps, ddof=1)
        mean_smoke, std_smoke = np.mean(smokes), np.std(smokes, ddof=1)

        EPS = 1e-6
        std_temp = max(np.std(temps, ddof=1), EPS)
        std_smoke = max(np.std(smokes, ddof=1), EPS)

        # Compute dynamic damping
        temp_damping = cls._dynamic_damping(std_temp, TEMP_PIVOT, TEMP_STEEPNESS)
        smoke_damping = cls._dynamic_damping(std_smoke, SMOKE_PIVOT, SMOKE_STEEPNESS)

        print("\n" + "═" * 170)
        print("DEBUG: Event Detection Summary (V1)")
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
            w_scr = cls._wind_to_score(dp.wind, WIND_PIVOT, WIND_STEEPNESS)

            # Final risk score
            risk = (
                TEMP_WEIGHT * t_dmp +
                SMOKE_WEIGHT * s_dmp +
                WIND_BASE_WEIGHT * w_scr
            )
            risk = max(0.0, min(100.0, risk))

            status = "ALERT" if risk > ALERT_THRESHOLD else ""

            print(
                f"{idx:03d} | {dp.timestamp[-19:]:19} | "
                f"{dp.temperature:7.2f} {t_z:7.2f} {t_sev:7.3f} {t_dmp:7.3f} | "
                f"{dp.smoke:8.4f} {s_z:7.2f} {s_sev:7.3f} {s_dmp:7.3f} | "
                f"{dp.wind:5.1f} {w_scr:6.3f} | "
                f"{risk:8.1f} {status:>8}"
            )

        print("═" * 170 + "\n")

    @classmethod
    def _print_debug_scores_v2(
        cls,
        data: List[DataPoint],
        events: List[Event],
        max_score: float,
    ) -> None:
        """Debug output for V2 — prints ALERT only at incident start."""

        if not data:
            print("No data to print.")
            return

        # Compute statistics
        temps = np.array([dp.temperature for dp in data], dtype=float)
        smokes = np.array([dp.smoke for dp in data], dtype=float)
        mean_temp, std_temp = np.mean(temps), np.std(temps, ddof=1)
        mean_smoke, std_smoke = np.mean(smokes), np.std(smokes, ddof=1)

        temp_damping = cls._dynamic_damping(std_temp, TEMP_PIVOT, TEMP_STEEPNESS)
        smoke_damping = cls._dynamic_damping(std_smoke, SMOKE_PIVOT, SMOKE_STEEPNESS)

        RESET_THRESHOLD = 65
        in_active_incident = False

        print("\n" + "═" * 170)
        print("DEBUG: Event Detection Summary (V2)")
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
            t_z = (dp.temperature - mean_temp) / std_temp
            s_z = (dp.smoke - mean_smoke) / std_smoke

            t_sev = cls._z_to_severity(t_z)
            s_sev = cls._z_to_severity(s_z)

            t_dmp = t_sev * temp_damping
            s_dmp = s_sev * smoke_damping

            w_scr = cls._wind_to_score(dp.wind, WIND_PIVOT, WIND_STEEPNESS)

            risk = (
                TEMP_WEIGHT * t_dmp +
                SMOKE_WEIGHT * s_dmp +
                WIND_BASE_WEIGHT * w_scr
            )
            risk = max(0.0, min(100.0, risk))

            status = ""

            # --- hysteresis-aware ALERT labeling ---
            if not in_active_incident and risk > ALERT_THRESHOLD:
                status = "ALERT"
                in_active_incident = True
            elif in_active_incident and risk < HYSTERESIS_RESET_THRESHOLD:
                in_active_incident = False

            print(
                f"{idx:03d} | {dp.timestamp[-19:]:19} | "
                f"{dp.temperature:7.2f} {t_z:7.2f} {t_sev:7.3f} {t_dmp:7.3f} | "
                f"{dp.smoke:8.4f} {s_z:7.2f} {s_sev:7.3f} {s_dmp:7.3f} | "
                f"{dp.wind:5.1f} {w_scr:6.3f} | "
                f"{risk:8.1f} {status:>8}"
            )

        print("═" * 170 + "\n")
