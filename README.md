# Wildfire Detection – Statistical Anomaly-Based Analysis

## Project Overview

This project implements a small, explainable analysis pipeline for detecting suspicious wildfire-related events from environmental sensor data. The system processes time-series measurements of temperature, smoke, and wind, computes a risk score (0–100) for each data point, and marks high-risk records as suspicious events.

The solution is intentionally rule-based and deterministic, designed as a clear baseline rather than a predictive or learning-based model.

## System Architecture

The solution is organized using a clean, modular design that separates responsibilities:

- **DataProcessor**: Handles input ordering and signal smoothing using Savitzky-Golay filter to reduce sensor noise.
- **EventDetector**: Performs anomaly detection and computes a bounded risk score per data point using variance-stabilized scoring.
- **DetectionService**: Orchestrates the full pipeline and acts as a boundary between the API layer and core logic.
- **API Layer**: Provides REST endpoints with input validation.

This structure ensures that the core detection logic is independent of the API framework and can be reused, tested, or extended without modification.

## Data Processing and Smoothing

Raw environmental sensor data often contains short-lived spikes caused by noise, sensor jitter, or transient effects. Before anomaly detection, the system applies Savitzky-Golay smoothing to temperature and smoke signals to reduce the impact of isolated outliers.

**Key design choices:**

- Temperature and smoke signals are smoothed using Savitzky-Golay filter (window=13, polyorder=2) to suppress single-point noise while preserving trends.
- Smoke values are clipped to their physical range [0, 1] after smoothing.
- Wind is intentionally not smoothed:
  - Wind is highly volatile by nature.
  - The data arrives at approximately one-minute resolution.
  - Wind is used only as a contextual signal to amplify risk, not as a primary anomaly source.

## Anomaly Detection Approach

### Statistical Outlier Detection

Anomalies are detected using a statistical approach based on deviations from normal behavior within the analyzed dataset:

- Mean and sample standard deviation (ddof=1) are computed for temperature and smoke.
- Z-scores are converted to severity scores [0, 1] using a one-sided CDF-based mapping.
- Only positive deviations (values above the mean) contribute to anomaly severity.
- Negative deviations are treated as normal behavior and do not increase risk.

This reflects the assumption that wildfire-related events manifest as abnormal increases in temperature and smoke rather than decreases.

### Variance-Aware Damping (Key Design Element)

Pure statistical outlier detection can produce misleading results in environments with very low variance. In such cases, even small absolute changes may appear statistically significant and lead to exaggerated anomaly scores.

To mitigate this, the algorithm applies a dynamic damping mechanism based on the observed standard deviation of each signal using sigmoid functions:

- **Temperature damping**: Sigmoid with pivot=4.0°C, steepness=3.0
- **Smoke damping**: Sigmoid with pivot=0.02, steepness=20.0
- When the variance is extremely low, anomaly contributions are dampened.
- As variability increases, damping is gradually reduced and anomalies regain their full weight.

This mechanism prevents overreaction to minor fluctuations in highly stable environments while preserving sensitivity to meaningful changes. The damping process is a core part of the scoring logic and plays a critical role in reducing false positives.

### Risk Score Computation (0–100)

For each data point, the system computes a risk score in the range 0–100 by combining multiple signals:

- **Temperature anomaly severity** (weight: 60) - primary indicator
- **Smoke anomaly severity** (weight: 60) - primary indicator  
- **Wind contribution** (weight: 15) - contextual amplifier via sigmoid mapping

**Key properties of the score:**

- Deterministic and repeatable
- Explicitly bounded to [0, 100]
- Interpretable, with each signal contributing independently
- Wind alone cannot trigger a high-risk score; it only increases risk when temperature or smoke anomalies are present.

### Suspicious Event Definition

A data point is marked as a suspicious event when:

```
risk_score > 70
```

This threshold represents an operational alert level rather than a definitive classification. It is intended to flag records that warrant further inspection or escalation by downstream systems.

## Configuration

All tunable parameters are centralized in `core/config.py`:

- Data processing: Savitzky-Golay filter parameters
- Event detection: Scoring weights, damping parameters, alert threshold

This allows easy adjustment of algorithm parameters without modifying core logic.

## Assumptions and Limitations

### Limitations (Current Version)

- **No historical context**: All statistical calculations are based solely on the current dataset. Previous days or historical baselines are not considered.
- **Future leakage within a full-day window**: When analyzing a full 24-hour window, statistics are computed using the entire dataset. If a wildfire dominates most of the day, early stages of the event may not appear anomalous because the baseline is already elevated. A future improvement would compute statistics incrementally up to the current timestamp.
- **Fire-at-start scenario**: If data collection begins during an ongoing fire, elevated values may be treated as normal behavior. A future version could incorporate expert-defined rules, external priors, or labeled reference patterns.

### Assumptions

- Sensor data is reasonably calibrated and reliable.
- Measurements arrive at approximately one-minute intervals.
- Wind is intentionally left unsmoothed due to its volatility and contextual role.

## Installation and Usage

### Install Dependencies

Create a virtual environment (optional but recommended), then install dependencies:

```bash
pip install -r requirements.txt
```

### Run the Service

Start the API using FastAPI:

```bash
uvicorn main:app --reload
```

The service will be available at:

```
http://localhost:8000
```

### Endpoints

- **POST /detect** — Analyze sensor data and return detected events
  - Input: Array of `DataPoint` objects (timestamp, temperature, smoke, wind)
  - Output: `EventsSummary` with events list, count, and max_score
  - Validation: Returns 422 error for empty input

- **GET /health** — Health check endpoint

### Example Request

```bash
curl -X POST "http://localhost:8000/detect" \
  -H "Content-Type: application/json" \
  -d @data/sample_input1Spike.json
```

### Run Tests

Run the full test suite using pytest:

```bash
pytest
```

Tests cover:

- Data processing and smoothing behavior
- Anomaly detection and scoring logic
- API-level integration scenarios

## Summary

This project provides a clear, explainable baseline for wildfire-related anomaly detection using statistical methods. Its focus is on interpretability, deterministic behavior, and safety, with variance-aware damping used to reduce false positives in stable environments.

The architecture and logic are intentionally designed to support future improvements such as online statistics, historical baselines, or expert-driven rules.
