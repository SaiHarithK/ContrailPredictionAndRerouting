import xarray as xr

# --------------------------------------------------
# Load ERA5 dataset (loaded only once)
# --------------------------------------------------

DATASET_PATH = "era5_data_small.nc"

ds = xr.open_dataset(DATASET_PATH)


# --------------------------------------------------
# Weather Query Function
# --------------------------------------------------

def get_weather(
    latitude,
    longitude,
    pressure_level,
    valid_time=None
):
    """
    Returns interpolated ERA5 weather at any location.

    Parameters
    ----------
    latitude : float
    longitude : float
    pressure_level : float
        Pressure level in hPa (e.g. 250)
    valid_time : datetime64 (optional)

    Returns
    -------
    dict
        Weather variables.
    """

    # Use first timestamp if none supplied
    if valid_time is None:
        valid_time = ds.valid_time.values[0]

    point = ds.interp(
        latitude=latitude,
        longitude=longitude,
        pressure_level=pressure_level,
        valid_time=valid_time
    )

    weather = {
        "temperature": float(point.t.values),
        "humidity": float(point.r.values),
        "u_wind": float(point.u.values),
        "v_wind": float(point.v.values),
        "vertical_velocity": float(point.w.values),
        "geopotential": float(point.z.values)
    }

    return weather


# --------------------------------------------------
# Testing
# --------------------------------------------------

if __name__ == "__main__":

    weather1 = get_weather(
        latitude=50.00,
        longitude=5.00,
        pressure_level=250
    )

    weather2 = get_weather(
        latitude=50.05,
        longitude=5.05,
        pressure_level=250
    )

    print("\nWeather at Point 1\n")
    print(weather1)

    print("\nWeather at Point 2\n")
    print(weather2)