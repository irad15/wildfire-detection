# Wildfire Detection – Statistical Anomaly-Based Analysis

## 1. Part 1 — Algorithmic Core Logic

### 1a. Data Processing and Smoothing

- Input data is first sorted by timestamps to ensure correct time order.
- Temperature and smoke data are smoothed using a Savitzky-Golay filter to reduce noise and suppress outliers.
- V2 optionally suppresses isolated spikes (temperature/smoke) before smoothing to avoid single-sample noise driving alerts.
- Wind data is left unsmoothed, since wind is naturally volatile and is used only to provide context to the risk score (not for anomaly detection).

### 1b. Anomaly Detection Approach

#### 1b.i Statistical Outlier Detection

Anomalies are detected statistically by analyzing deviations from normal behavior:

- Formulas:
```bash
mean = sum(x_i) / n
std  = sqrt( sum((x_i - mean)^2) / (n - 1) )
z    = max(0, (x - mean) / std)   # positive-only
severity = 2 * (0.5 * (1 + erf(z / sqrt(2))) - 0.5)  # one-sided CDF in [0,1]
```

- For temperature and smoke, mean and sample standard deviation are computed.
- Positive z-score deviations (values above the mean) are mapped to severity scores [0, 1] with a one-sided CDF.
- Only positive deviations raise risk; negative ones are ignored.

#### 1b.ii Variance-Aware Damping (Key Design Element):

- Formula:
```bash
damping(std) = 1 / (1 + exp(steepness * (pivot - std)))
```

In low-variance data, small changes can appear much more significant than they are. To address this, the algorithm dampens anomaly scores when variance is very low, using a sigmoid-based scaling:

- Low variance: anomaly scores are reduced.
- Higher variance: scores approach full strength.

This reduces false positives from minor fluctuations while staying responsive to real anomalies.

#### 1b.iii Risk Score Computation (0–100):

- Formula:
```bash
risk = (TEMP_WEIGHT  * severity_temp * damping_temp) +
       (SMOKE_WEIGHT * severity_smoke * damping_smoke) +
       (WIND_BASE_WEIGHT * wind_score)
risk = clamp(risk, 0, 100)
```

For each data point, the system computes a risk score ranging from 0 to 100 by combining several signals:

- **Temperature anomaly severity** – primary indicator
- **Smoke anomaly severity** – primary indicator  
- **Wind contribution** – secondary indicator
- Optional V2 hysteresis: alerts fire once per incident until risk cools below a reset threshold.

**Key properties of the score:**

- Deterministic and repeatable
- Strictly bounded within [0, 100]
- Designed such that no single primary indicator can trigger an event/alert. Elevated risk only occurs when multiple signals together indicate abnormal behavior.


A data point is marked as a suspicious event when:

```
risk_score > 70
```

All tunable parameters are centralized in `core/config.py`:

- Data processing: Savitzky-Golay filter parameters
- V2 spike suppression: temperature/smoke spike thresholds
- Event detection: Scoring weights, damping parameters, alert threshold
- V2 hysteresis: reset threshold to re-arm alerts

This allows easy adjustment of algorithm parameters without modifying core logic.

### 1c. Assumptions and Limitations

#### Limitations:

- **Future leakage within a full-day window**: When analyzing a complete 24-hour window, statistics are computed using the entire dataset. If a wildfire dominates most of the day, early stages of the event may not appear anomalous because the baseline is already elevated, potentially leading to missed alerts. A future improvement would compute statistics incrementally up to the current timestamp.

- **Fire-at-start scenario**: If data collection begins during an ongoing fire, elevated values may be treated as normal behavior since they define the dataset’s baseline. As a result, the system may fail to flag the event. A future version could address this by incorporating expert-defined rules.

- **No historical context**: All statistical calculations are performed on the current dataset only. Historical baselines from previous days are not used.

#### Assumptions:

- Sensor data is reasonably calibrated and reliable.
- Measurements arrive at approximately one-minute intervals.
- The detection logic operates on a finite batch of data points (e.g., a full time window) rather than on a continuous real-time stream.
- Wind is intentionally left unsmoothed due to its volatility and contextual role.

## 2. Part 2 — Microservice API

### Install Dependencies

Create a virtual environment (optional but recommended), then install dependencies:

```bash
pip install -r requirements.txt
```

### Run the Service

Start the API using FastAPI:

```bash
uvicorn main:app --reload --port 8000
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
  -d @data/sample_input_1_spike.json
```

You can also explore and try the API interactively via Swagger UI at:
```
http://127.0.0.1:8000/docs
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

### Optional: Run V2 and Benchmarks
- Switch to V2 locally by enabling `process_v2` and `detect_v2` in `core/detection_service.py`.
- Another option is to run benchmarks comparing V1 vs V2 on sample data:
  ```bash
  python3 -m benchmarks.benchmark_detection
  ```
- In addition to the benchmark summary, this test prints detailed output for both versions, showing per-data-point statistics, risk scores, and alert decisions.