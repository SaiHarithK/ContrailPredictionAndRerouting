import xarray as xr
import numpy as np
from datetime import datetime

print("Loading ERA5 full year dataset...")
ds = xr.open_dataset("era5_2023_full.nc")
print(f"Loaded: {len(ds.valid_time)} timesteps "
      f"({str(ds.valid_time.values[0])[:10]} → "
      f"{str(ds.valid_time.values[-1])[:10]})")


def get_weather(latitude, longitude, pressure_level, valid_time=None):
    """
    Returns ERA5 weather interpolated to the given location and pressure.
    
    Time matching: finds the nearest timestep by month and day,
    using 2023 data as a climatological proxy for any year.
    """

    # --- Match to nearest timestep by month ---
    if valid_time is not None:
        target_month = valid_time.month
        target_day   = valid_time.day
        target_hour  = valid_time.hour
    else:
        target_month, target_day, target_hour = 1, 1, 0

    # Find all timesteps in the same month
    times = ds.valid_time.values
    months = [int(str(t)[5:7]) for t in times]
    days   = [int(str(t)[8:10]) for t in times]
    hours  = [int(str(t)[11:13]) for t in times]

    # Score each timestep by proximity to target month/day/hour
    scores = [
        abs(months[i] - target_month) * 10000 +
        abs(days[i]   - target_day)   * 100   +
        abs(hours[i]  - target_hour)
        for i in range(len(times))
    ]
    best_idx = int(np.argmin(scores))

    # --- Clamp coordinates to dataset bounds ---
    lat = float(np.clip(latitude,
                        float(ds.latitude.min()),
                        float(ds.latitude.max())))
    lon = float(np.clip(longitude,
                        float(ds.longitude.min()),
                        float(ds.longitude.max())))
    plev = float(np.clip(pressure_level,
                         float(ds.pressure_level.min()),
                         float(ds.pressure_level.max())))

    # --- Interpolate spatially at selected timestep ---
    point = ds.isel(valid_time=best_idx).interp(
        latitude=lat,
        longitude=lon,
        pressure_level=plev,
    )

    return {
        "temperature":       float(point["t"].values),
        "humidity":          float(point["r"].values),
        "u_wind":            float(point["u"].values),
        "v_wind":            float(point["v"].values),
        "vertical_velocity": float(point["w"].values),
        "geopotential":      float(point["z"].values),
    }


if __name__ == "__main__":
    print("\nSeasonal test at (50N, 5E, 250 hPa):")
    print(f"{'Month':<8} {'Temp (K)':<12} {'RHi (%)':<12} {'U wind':<10}")
    print("-" * 45)

    test_points = [
        (1,  "Jan"),
        (4,  "Apr"),
        (7,  "Jul"),
        (10, "Oct"),
    ]

    for month, label in test_points:
        ts = datetime(2023, month, 15, 12)
        w  = get_weather(50.0, 5.0, 250, valid_time=ts)
        print(f"{label:<8} {w['temperature']:<12.1f} "
              f"{w['humidity']:<12.1f} {w['u_wind']:<10.1f}")