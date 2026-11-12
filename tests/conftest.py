from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def regular_frame() -> pd.DataFrame:
    timestamps = pd.date_range("2024-01-01", periods=500, freq="10min", tz="UTC")
    rows = []
    for building in ("A", "B"):
        for index, timestamp in enumerate(timestamps):
            occupancy = float((index % 50) + (10 if building == "B" else 0))
            rows.append(
                {
                    "timestamp": timestamp,
                    "building_id": building,
                    "energy_wh": 100 + 2 * occupancy + np.sin(index / 10),
                    "occupancy": occupancy,
                    "temperature_c": 25 + np.sin(index / 30),
                    "humidity_pct": 55,
                    "working_day": float(timestamp.dayofweek < 5),
                    "high_activity": float(timestamp.dayofweek < 5),
                }
            )
    return pd.DataFrame(rows)
