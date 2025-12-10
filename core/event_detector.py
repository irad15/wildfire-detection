from typing import List

import numpy as np

from .models import Alert, AlertsSummary


class EventDetector:
    def __init__(self, processed_data: List[dict]):
        self.data = processed_data

    def _print_debug_scores(self, scores: List[float]) -> None:
        """Debug helper: print all entries with timestamp, temp, smoke, wind, score"""
        print("\n=== Detection Debug: All Data Points with Risk Scores ===")
        print("idx | timestamp              | temp   | smoke  | wind | score  ")
        print("-" * 68)
        for idx, (item, score) in enumerate(zip(self.data, scores), 1):
            print(
                f"{idx:02d} | {item['timestamp'][-15:]:15} | "
                f"{item.get('smoothed_temp', item['temperature']):6.2f} | "
                f"{item.get('smoothed_smoke', item['smoke']):6.4f} | "
                f"{item['wind']:4.1f} | {score:6.1f}"
            )
        print("=== End Debug ===\n")

    def detect(self) -> AlertsSummary:        
        if not self.data:
            return AlertsSummary(events=[], event_count=0, max_score=0.0)

        # Extract smoothed values
        smoothed_temps = np.array([item["smoothed_temp"] for item in self.data], dtype=float)
        smoothed_smokes = np.array([item["smoothed_smoke"] for item in self.data], dtype=float)
        winds = np.array([item["wind"] for item in self.data], dtype=float)

        # Global statistics on smoothed data
        mean_temp = np.mean(smoothed_temps)
        std_temp = np.std(smoothed_temps) if np.std(smoothed_temps) > 0 else 1.0
        mean_smoke = np.mean(smoothed_smokes)
        std_smoke = np.std(smoothed_smokes) if np.std(smoothed_smokes) > 0 else 1.0

        alerts: List[Alert] = []
        max_score = 0.0
        all_scores: List[float] = []  # To collect for debug print

        for item in self.data:
            smoothed_temp = item["smoothed_temp"]
            smoothed_smoke = item["smoothed_smoke"]
            wind = item["wind"]

            # Anomaly detection: positive z-scores only
            temp_anomaly = max(0.0, (smoothed_temp - mean_temp) / std_temp)
            smoke_anomaly = max(0.0, (smoothed_smoke - mean_smoke) / std_smoke)

            # Wind multiplier
            wind_multiplier = 1.0 + (wind / 15.0)

            # Base + final risk score
            base_score = 40 * temp_anomaly + 50 * smoke_anomaly + 10 * (wind / 2)
            risk_score = min(100.0, max(0.0, base_score * wind_multiplier))

            all_scores.append(risk_score)

            if risk_score > max_score:
                max_score = risk_score

            if risk_score > 70:
                alerts.append(Alert(
                    timestamp=item["timestamp"],
                    score=round(risk_score, 1)
                ))

        # Debug print just before returning
        self._print_debug_scores(all_scores)

        return AlertsSummary(
            events=alerts,
            event_count=len(alerts),
            max_score=round(max_score, 1) if alerts or max_score > 0 else 0.0
        )