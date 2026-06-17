from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    output_path = Path("data/raw/example_timeseries.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    frame = build_example_frame()
    frame.to_csv(output_path, index=False)
    print(f"Wrote {output_path} with {len(frame)} rows")


def build_example_frame(n_timesteps: int = 720, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    time = np.arange(n_timesteps, dtype=np.float64)
    timestamp = pd.date_range("2024-01-01", periods=n_timesteps, freq="h")

    daily = np.sin(2.0 * np.pi * time / 24.0)
    weekly = np.cos(2.0 * np.pi * time / 168.0)
    trend = 0.01 * time
    temperature = 55.0 + 8.0 * daily + 4.0 * weekly + rng.normal(0.0, 0.35, n_timesteps)
    promo_index = 0.4 + 0.25 * np.sin(2.0 * np.pi * time / 96.0 + 0.4)
    promo_index += rng.normal(0.0, 0.03, n_timesteps)
    demand_driver = 12.0 + 0.18 * temperature + 1.8 * promo_index + 0.15 * trend
    demand_driver += rng.normal(0.0, 0.2, n_timesteps)

    lagged_driver = np.roll(demand_driver, 3)
    lagged_driver[:3] = demand_driver[0]
    value = (
        20.0
        + trend
        + 0.65 * temperature
        + 4.0 * promo_index
        + 0.35 * lagged_driver
        + 2.5 * daily
        + rng.normal(0.0, 0.45, n_timesteps)
    )

    return pd.DataFrame(
        {
            "timestamp": timestamp,
            "value": value,
            "temperature": temperature,
            "promo_index": promo_index,
            "demand_driver": demand_driver,
            "lagged_driver": lagged_driver,
        }
    )


if __name__ == "__main__":
    main()
