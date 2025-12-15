import json
import time
from pathlib import Path

from core.data_processor import DataProcessor
from core.event_detector import EventDetector
from core.models import DataPoint


DATA_PATH = Path("data/sample_input_flat_rise_twice.json")  # scenario with two gradual rises


def load_data(path: Path):
    """Load raw data from a JSON file and convert to DataPoint objects."""
    with open(path, "r") as f:
        raw = json.load(f)

    return [DataPoint(**item) for item in raw]


def benchmark(data, process_func, detect_func):
    """Generic benchmark utility: run a full pipeline and measure runtime + event count."""
    start = time.perf_counter()
    processed = process_func(data)
    result = detect_func(processed)
    duration = time.perf_counter() - start
    return duration, result.event_count


if __name__ == "__main__":
    data = load_data(DATA_PATH)

    # Benchmark for V1 (original)
    v1_time, v1_events = benchmark(data, DataProcessor.process, EventDetector.detect)

    # Benchmark for V2 (improved version with spike suppression + hysteresis)
    v2_time, v2_events = benchmark(data, DataProcessor.process_v2, EventDetector.detect_v2)

    # Display results
    print("=== Benchmark Results (sample_input_flat_rise_twice) ===")
    print(f"V1 runtime: {v1_time:.6f} seconds")
    print(f"V2 runtime: {v2_time:.6f} seconds")
    print(f"V1 events:  {v1_events}")
    print(f"V2 events:  {v2_events}")
