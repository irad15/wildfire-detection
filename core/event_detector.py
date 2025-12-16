from typing import List, Callable
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
    """
    Performs anomaly detection and computes a risk score per data point using variance-aware scoring.
    V2 adds alert hysteresis to reduce duplicate alerts within the same incident.
    """

    @classmethod
    def detect(cls, processed_data: List[DataPoint]) -> EventsSummary:
        """Main pipeline: compute statistics → score each point → build summary."""
        return cls._run_detection_pipeline(
            processed_data,
            use_hysteresis=False,
            debug_printer=cls._print_debug_scores,
        )

    @classmethod
    def detect_v2(cls, processed_data: List[DataPoint]) -> EventsSummary:
        """V2 pipeline: same scoring, but with hysteresis to avoid repeat alerts."""
        return cls._run_detection_pipeline(
            processed_data,
            use_hysteresis=True,
            debug_printer=cls._print_debug_scores_v2,
        )

    @classmethod
    def _run_detection_pipeline(
        cls,
        processed_data: List[DataPoint],
        use_hysteresis: bool,
        debug_printer: Callable[..., None],
    ) -> EventsSummary:
        """
        Shared wrapper to handle empty checks, execution, and summarization.
        """
        if not processed_data:
            return EventsSummary(events=[], event_count=0, max_score=0.0)

        # Call the calculation logic directly
        events, max_score = cls._calculate_risk_scores(processed_data, use_hysteresis=use_hysteresis)

        # Optional debug output
        debug_printer(processed_data, events=events, max_score=max_score)

        return EventsSummary(
            events=events,
            event_count=len(events),
            max_score=round(max_score, 1) if events or max_score > 0 else 0.0,
        )

    @classmethod
    def _calculate_risk_scores(cls, data: List[DataPoint], use_hysteresis: bool) -> tuple[List[Event], float]:
        """
        Main Loop: Prepares stats, iterates through data, and manages alert state.
        """
        if not data:
            return [], 0.0

        # 1. Prepare statistics (unpack tuple from helper)
        (mean_t, std_t, damp_t, 
         mean_s, std_s, damp_s) = cls._get_batch_statistics(data)
        
        events: List[Event] = []
        max_score = 0.0
        in_active_incident = False

        # 2. Score points and manage alerts
        for dp in data:
            # Positive-only z-scores: only above-mean deviations contribute to risk
            t_z = max(0.0, (dp.temperature - mean_t) / std_t)
            s_z = max(0.0, (dp.smoke - mean_s) / std_s)

            # Combine severity (temp/smoke) with damping and wind to form total risk
            risk_score = (
                (cls._z_to_severity(t_z) * damp_t * TEMP_WEIGHT) +
                (cls._z_to_severity(s_z) * damp_s * SMOKE_WEIGHT) +
                (cls._wind_to_score(dp.wind, WIND_PIVOT, WIND_STEEPNESS) * WIND_BASE_WEIGHT)
            )
            
            # Clamp and round
            risk_score = round(max(0.0, min(100.0, risk_score)), 1)
            max_score = max(max_score, risk_score)

            # --- Alert Logic ---
            if not use_hysteresis:
                # V1: emit an alert for every point crossing the threshold
                if risk_score > ALERT_THRESHOLD:
                    events.append(Event(timestamp=dp.timestamp, score=risk_score))
            else:
                # V2: emit once per incident while above threshold; reset after cooldown
                if not in_active_incident and risk_score > ALERT_THRESHOLD:
                    events.append(Event(timestamp=dp.timestamp, score=risk_score))
                    in_active_incident = True
                elif in_active_incident and risk_score < HYSTERESIS_RESET_THRESHOLD:
                    in_active_incident = False

        return events, max_score


    # -------------------------------------------------------------------------
    # Helper methods
    # -------------------------------------------------------------------------

    @classmethod
    def _get_batch_statistics(cls, data: List[DataPoint]) -> tuple[float, float, float, float, float, float]:
        """
        Calculates means, standard deviations, and damping factors for the batch.
        Returns: (mean_temp, std_temp, damp_temp, mean_smoke, std_smoke, damp_smoke)
        """
        temps = np.array([dp.temperature for dp in data], dtype=float)
        smokes = np.array([dp.smoke for dp in data], dtype=float)

        EPS = 1e-6
        mean_t = np.mean(temps)
        std_t = max(np.std(temps, ddof=1), EPS)
        
        mean_s = np.mean(smokes)
        std_s = max(np.std(smokes, ddof=1), EPS)

        # Calculate damping based on variance
        damp_t = cls._dynamic_damping(std_t, TEMP_PIVOT, TEMP_STEEPNESS)
        damp_s = cls._dynamic_damping(std_s, SMOKE_PIVOT, SMOKE_STEEPNESS)

        return mean_t, std_t, damp_t, mean_s, std_s, damp_s

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
        One-sided CDF-based severity score. (Cumulative Distribution Function - Normal Distribution)

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
    def _print_debug_scores(cls, data: List[DataPoint], events: List[Event], max_score: float) -> None:
        """Wrapper for V1 debug output."""
        cls._print_debug_shared(data, events, max_score, use_hysteresis=False)

    @classmethod
    def _print_debug_scores_v2(cls, data: List[DataPoint], events: List[Event], max_score: float) -> None:
        """Wrapper for V2 debug output."""
        cls._print_debug_shared(data, events, max_score, use_hysteresis=True)

    @classmethod
    def _print_debug_shared(
        cls, 
        data: List[DataPoint], 
        events: List[Event], 
        max_score: float, 
        use_hysteresis: bool
    ) -> None:
        """
        Unified debug printer. 
        Reuses batch statistics and reproduces the exact math used in detection.
        """
        if not data:
            print("No data to print.")
            return

        # 1. Reuse the exact same stats logic as the detection pipeline
        (mean_t, std_t, damp_t, 
         mean_s, std_s, damp_s) = cls._get_batch_statistics(data)

        # 2. Print Summary Header
        version_label = "V2 (Hysteresis)" if use_hysteresis else "V1 (Simple)"
        print("\n" + "═" * 120)
        print(f"DEBUG: Event Detection Summary [{version_label}]")
        print(
            f"Mean temp: {mean_t:6.2f}°C | Std temp: {std_t:8.4f} | Temp damping: {damp_t:5.3f}\n"
            f"Mean smoke: {mean_s:8.4f} | Std smoke: {std_s:8.4f} | Smoke damping: {damp_s:5.3f}\n"
            f"Total points: {len(data)} | Events detected: {len(events)} | Max score: {max_score:5.1f}"
        )
        print("─" * 120)

        # 3. Print Table Header
        print(
            f"{'idx':>3} | {'timestamp':^19} | "
            f"{'temp':>7} {'t_z':>7} {'t_sev':>7} {'t_dmp':>7} | "
            f"{'smoke':>8} {'s_z':>7} {'s_sev':>7} {'s_dmp':>7} | "
            f"{'wind':>5} {'w_scr':>6} | "
            f"{'risk':>8} {'status':>8}"
        )
        print("─" * 120)

        in_active_incident = False

        # 4. Iterate and reproduce math
        for idx, dp in enumerate(data, 1):
            # Recalculate intermediate values to show them in the table
            t_z = max(0.0, (dp.temperature - mean_t) / std_t)
            s_z = max(0.0, (dp.smoke - mean_s) / std_s)

            t_sev = cls._z_to_severity(t_z)
            s_sev = cls._z_to_severity(s_z)

            t_dmp = t_sev * damp_t
            s_dmp = s_sev * damp_s
            w_scr = cls._wind_to_score(dp.wind, WIND_PIVOT, WIND_STEEPNESS)

            # Combined Risk
            risk = (
                TEMP_WEIGHT * t_dmp +
                SMOKE_WEIGHT * s_dmp +
                WIND_BASE_WEIGHT * w_scr
            )
            risk = round(max(0.0, min(100.0, risk)), 1)

            # Determine Status Label based on strategy
            status = ""
            if not use_hysteresis:
                if risk > ALERT_THRESHOLD:
                    status = "ALERT"
            else:
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

        print("═" * 120 + "\n")
