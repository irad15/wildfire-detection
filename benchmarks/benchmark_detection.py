import json
import time
from pathlib import Path

from core.data_processor import DataProcessor
from core.event_detector import EventDetector
from core.models import DataPoint


DATASETS = {
    "flat_rise_twice": Path("data/sample_input_flat_rise_twice.json"),
    "slow_rise_windy": Path("data/sample_input_slow_rise_windy.json"),
}


def load_data(path: Path):
    """Load raw data from a JSON file and convert to DataPoint objects."""
    with open(path, "r") as f:
        raw = json.load(f)

    return [DataPoint(**item) for item in raw]


def benchmark(data, process_func, detect_func):
    """
    Benchmark utility: measure processor time, detector time, and total time.
    Returns: (processor_time, detector_time, total_time, event_count)
    """
    t0 = time.perf_counter()
    processed = process_func(data)
    t1 = time.perf_counter()
    result = detect_func(processed)
    t2 = time.perf_counter()

    processor_time = t1 - t0
    detector_time = t2 - t1
    total_time = t2 - t0

    return processor_time, detector_time, total_time, result.event_count


if __name__ == "__main__":
    results = []

    for label, path in DATASETS.items():
        data = load_data(path)

        # Benchmark V1 (baseline)
        v1_proc, v1_det, v1_total, v1_events = benchmark(
            data, DataProcessor.process, EventDetector.detect
        )

        # Benchmark V2 (spike suppression + hysteresis)
        v2_proc, v2_det, v2_total, v2_events = benchmark(
            data, DataProcessor.process_v2, EventDetector.detect_v2
        )

        results.append(
            {
                "label": label,
                "v1": {"proc": v1_proc, "det": v1_det, "total": v1_total, "events": v1_events},
                "v2": {"proc": v2_proc, "det": v2_det, "total": v2_total, "events": v2_events},
            }
        )

    # Print all results together
    for entry in results:
        label = entry["label"]
        v1 = entry["v1"]
        v2 = entry["v2"]
        total_delta_pct = ((v2["total"] - v1["total"]) / v1["total"]) * 100 if v1["total"] else 0.0

        print(f"\n=== Benchmark Results ({label}) ===")
        print("V1 (baseline):")
        print(f"  Processor time : {v1['proc']:.6f} seconds")
        print(f"  Detector time  : {v1['det']:.6f} seconds")
        print(f"  Total time     : {v1['total']:.6f} seconds")
        print(f"  Events detected: {v1['events']}")
        print()

        print("V2 (spike suppression + hysteresis):")
        print(f"  Processor time : {v2['proc']:.6f} seconds")
        print(f"  Detector time  : {v2['det']:.6f} seconds")
        print(f"  Total time     : {v2['total']:.6f} seconds ({total_delta_pct:+.2f}% vs V1)")
        print(f"  Events detected: {v2['events']}")
        print()

