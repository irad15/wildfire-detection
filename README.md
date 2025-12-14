# Wildfire Detection – Statistical Anomaly-Based Analysis

## Project Overview

This project implements a small, explainable analysis pipeline for detecting suspicious wildfire-related events from environmental sensor data. The system processes time-series measurements of temperature, smoke, and wind, computes a risk score (0–100) for each data point, and marks high-risk records as suspicious events.

## System Architecture

The solution is organized using a clean, modular design that separates responsibilities:

- **DataProcessor**: Handles input ordering and signal smoothing using Savitzky-Golay filter to reduce sensor noise.
- **EventDetector**: Performs anomaly detection and computes a risk score per data point using variance-aware scoring.
- **DetectionService**: Orchestrates the full pipeline and acts as a boundary between the API layer and core logic.
- **API Layer**: Provides REST endpoints with input validation.

## Data Processing and Smoothing

- Input data is first sorted by timestamps to ensure correct time order.
- Temperature and smoke data are smoothed using a Savitzky-Golay filter to reduce noise and suppress outliers.
- Wind data is left unsmoothed, since wind is naturally volatile and is used only to provide context to the risk score (not for anomaly detection).

## Anomaly Detection Approach

### Statistical Outlier Detection

Anomalies are detected statistically by analyzing deviations from normal behavior:

- For temperature and smoke, mean and sample standard deviation are computed.
- Positive z-score deviations (values above the mean) are mapped to severity scores [0, 1] with a one-sided CDF.
- Only positive deviations raise risk; negative ones are ignored.

### Variance-Aware Damping (Key Design Element)

In low-variance data, small changes can appear much more significant than they are. To address this, the algorithm dampens anomaly scores when variance is very low, using a sigmoid-based scaling:

- Low variance: anomaly scores are reduced.
- Higher variance: scores approach full strength.

This reduces false positives from minor fluctuations while staying responsive to real anomalies.

### Risk Score Computation (0–100)

For each data point, the system computes a risk score ranging from 0 to 100 by combining several signals:

- **Temperature anomaly severity** – primary indicator
- **Smoke anomaly severity** – primary indicator  
- **Wind contribution** – secondary indicator

**Key properties of the score:**

- Deterministic and repeatable
- Strictly bounded within [0, 100]
- Designed such that no single primary indicator can trigger an alert. Elevated risk only occurs when multiple signals together indicate abnormal behavior.


A data point is marked as a suspicious event when:

```
risk_score > 70
```

## Configuration

All tunable parameters are centralized in `core/config.py`:

- Data processing: Savitzky-Golay filter parameters
- Event detection: Scoring weights, damping parameters, alert threshold

This allows easy adjustment of algorithm parameters without modifying core logic.

## Assumptions and Limitations

### Limitations (Current Version)

- **Future leakage within a full-day window**: When analyzing a complete 24-hour window, statistics are computed using the entire dataset. If a wildfire dominates most of the day, early stages of the event may not appear anomalous because the baseline is already elevated, potentially leading to missed alerts. A future improvement would compute statistics incrementally up to the current timestamp.

- **Fire-at-start scenario**: If data collection begins during an ongoing fire, elevated values may be treated as normal behavior since they define the dataset’s baseline. As a result, the system may fail to flag the event. A future version could address this by incorporating expert-defined rules.

- **No historical context**: All statistical calculations are performed on the current dataset only. Historical baselines from previous days are not used.

### Assumptions

- Sensor data is reasonably calibrated and reliable.
- Measurements arrive at approximately one-minute intervals.
- The detection logic operates on a finite batch of data points (e.g., a full time window) rather than on a continuous real-time stream.
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

- API request validation (empty/missing/invalid inputs)
- Data processing and smoothing behavior
- Anomaly detection and scoring logic

## Summary

This project provides a clear, explainable baseline for wildfire-related anomaly detection using statistical methods. Its focus is on interpretability, deterministic behavior, and safety, with variance-aware damping used to reduce false positives in stable environments.

The architecture and logic are intentionally designed to support future improvements such as historical baselines, or expert-driven rules.
