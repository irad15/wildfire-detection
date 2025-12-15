"""
Configuration constants for wildfire detection system.

This module contains all tunable parameters for data processing and event detection.
"""

# ============================================================================
# Data Processing Configuration
# ============================================================================

# Savitzky-Golay filter parameters
SAVITZKY_GOLAY_POLYORDER = 2
SAVITZKY_GOLAY_WINDOW = 13  # Must be odd and greater than polyorder

# Spike suppression parameters (DataProcessor V2)
TEMP_SPIKE_THRESHOLD = 10.0  # Threshold for Temperature spike suppression
SMOKE_SPIKE_THRESHOLD = 0.6  # Threshold for Smoke spike suppression

# ============================================================================
# Event Detection Configuration
# ============================================================================

# Risk scoring weights
TEMP_WEIGHT = 60
SMOKE_WEIGHT = 60
WIND_BASE_WEIGHT = 15

# Alert threshold
ALERT_THRESHOLD = 70

# Temperature damping parameters
TEMP_PIVOT = 4.0  # °C
TEMP_STEEPNESS = 3.0

# Smoke damping parameters
SMOKE_PIVOT = 0.02  # tuned to typical smoke std
SMOKE_STEEPNESS = 20.0  # sharper transition for small std

# Wind scoring parameters
WIND_PIVOT = 6.0  # m/s
WIND_STEEPNESS = 0.8

# Alert hysteresis (EventDetector V2)
HYSTERESIS_RESET_THRESHOLD = 65  # Threshold to re-arm alerting after firing

