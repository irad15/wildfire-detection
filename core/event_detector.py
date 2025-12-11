from typing import List

import numpy as np

from .models import Alert, AlertsSummary


class EventDetector:
    def __init__(self, processed_data: List[dict]):
        self.data = processed_data

    def _print_debug_scores(
        self,
        temp_anomalies: List[float],
        smoke_anomalies: List[float],
        risk_scores: List[float],
        std_temp: float,
        std_smoke: float,
    ) -> None:
        """Beautiful debug table + global std values on top"""
        print("\n" + "═" * 125)
        print("DEBUG: Full Anomaly Detection Breakdown")
        print("═" * 125)
        print(f"Global std_temp  = {std_temp:7.4f} °C    |    Global std_smoke = {std_smoke:8.6f}")
        print("─" * 125)

        print(
            f"{'idx':>3} | {'timestamp':^19} | {'temp':>7} | {'t_anom':>10} | "
            f"{'smoke':>9} | {'s_anom':>11} | {'wind':>5} | {'risk':>11} {'status':>8}"
        )
        print("─" * 125)

        for idx, (item, t_anom, s_anom, risk) in enumerate(
            zip(self.data, temp_anomalies, smoke_anomalies, risk_scores), 1
        ):
            status = "ALERT" if risk > 70 else ""
            ts = item["timestamp"][-19:]  # 2025-08-02T00:00:00Z
            print(
                f"{idx:03d} | {ts} | "
                f"{item.get('smoothed_temp', item['temperature']):7.2f} | "
                f"{t_anom:10.3f} | "
                f"{item.get('smoothed_smoke', item['smoke']):9.4f} | "
                f"{s_anom:11.3f} | "
                f"{item['wind']:5.1f} | "
                f"{risk:11.1f} {status:>8}"
            )
        print("═" * 125 + "\n")


    def detect(self) -> AlertsSummary:
        if not self.data:
            return AlertsSummary(events=[], event_count=0, max_score=0.0)

        # Extract smoothed values
        smoothed_temps = np.array([item["smoothed_temp"] for item in self.data], dtype=float)
        smoothed_smokes = np.array([item["smoothed_smoke"] for item in self.data], dtype=float)
        winds = np.array([item["wind"] for item in self.data], dtype=float)

        # Global statistics (only once!)
        mean_temp = np.mean(smoothed_temps)
        std_temp = np.std(smoothed_temps) or 1.0
        mean_smoke = np.mean(smoothed_smokes)
        std_smoke = np.std(smoothed_smokes) or 1.0

        alerts: List[Alert] = []
        max_score = 0.0
        temp_anomalies = []
        smoke_anomalies = []
        risk_scores = []

        for item in self.data:
            smoothed_temp = item["smoothed_temp"]
            smoothed_smoke = item["smoothed_smoke"]
            wind = item["wind"]

            # Z-scores (positive only)
            temp_anomaly = max(0.0, (smoothed_temp - mean_temp) / std_temp)
            smoke_anomaly = max(0.0, (smoothed_smoke - mean_smoke) / std_smoke)

            # Wind multiplier
            wind_multiplier = 1.0 + (wind / 15.0)

            # Risk score
            risk_score = min(100.0, max(0.0, 40 * temp_anomaly + 50 * smoke_anomaly + 10 * (wind / 2)))

            # Store everything
            temp_anomalies.append(temp_anomaly)
            smoke_anomalies.append(smoke_anomaly)
            risk_scores.append(risk_score)

            if risk_score > max_score:
                max_score = risk_score
            if risk_score > 70:
                alerts.append(Alert(timestamp=item["timestamp"], score=round(risk_score, 1)))

        # Print beautiful debug table with correct std values
        self._print_debug_scores(
            temp_anomalies, smoke_anomalies, risk_scores, std_temp, std_smoke
        )

        return AlertsSummary(
            events=alerts,
            event_count=len(alerts),
            max_score=round(max_score, 1) if alerts or max_score > 0 else 0.0,
        )