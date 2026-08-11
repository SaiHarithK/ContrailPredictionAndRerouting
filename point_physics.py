import math

# ==========================================================
# Constants
# ==========================================================

GRAVITY = 9.81

# ==========================================================
# Relative Humidity with respect to Ice
# Murphy & Koop (2005)
# ==========================================================

def calculate_rhi(temperature, relative_humidity):
    """
    Parameters
    ----------
    temperature : float (Kelvin)
    relative_humidity : float (%)

    Returns
    -------
    float
        Relative Humidity with respect to Ice (%)
    """

    relative_humidity = max(0, min(relative_humidity, 100))

    es_ice = math.exp(
        9.550426
        - (5723.265 / temperature)
        + 3.53068 * math.log(temperature)
        - 0.00728332 * temperature
    )

    es_water = math.exp(
        54.842763
        - (6763.22 / temperature)
        - 4.210 * math.log(temperature)
        + 0.000367 * temperature
        + math.tanh(0.0415 * (temperature - 218.8))
        * (
            53.878
            - (1331.22 / temperature)
            - 9.44523 * math.log(temperature)
            + 0.014025 * temperature
        )
    )

    rhi = relative_humidity * (es_water / es_ice)

    return rhi


# ==========================================================
# Critical Temperature
# ==========================================================

def calculate_tcritical(pressure_hpa):
    """
    Parameters
    ----------
    pressure_hpa : float

    Returns
    -------
    float
        Critical temperature (Kelvin)
    """

    return 233 - 0.0065 * (250 - pressure_hpa)


# ==========================================================
# Delta T
# ==========================================================

def calculate_delta_t(temperature, tcritical):

    return temperature - tcritical


# ==========================================================
# Wind Speed
# ==========================================================

def calculate_wind_speed(u, v):

    return math.sqrt(u * u + v * v)


# ==========================================================
# Placeholder Wind Shear
# ==========================================================

def calculate_wind_shear():
    """
    Wind shear cannot be calculated from a single point.

    Placeholder until neighbouring pressure levels
    are queried from ERA5.
    """

    return 0.0


# ==========================================================
# Placeholder Static Stability
# ==========================================================

def calculate_static_stability():
    """
    Static stability requires temperature at multiple
    pressure levels.

    Placeholder for now.
    """

    return 0.0


# ==========================================================
# ISSR Depth
# ==========================================================

def calculate_issr_depth(rhi):

    if rhi > 100:
        return 1

    return 0


# ==========================================================
# Build Feature Vector
# ==========================================================

def build_features(weather, pressure_level):
    """
    Parameters
    ----------
    weather : dict
        Output from weather_query.get_weather()

    pressure_level : float

    Returns
    -------
    dict
        Physics features
    """

    temperature = weather["temperature"]
    humidity = weather["humidity"]

    u = weather["u_wind"]
    v = weather["v_wind"]

    rhi = calculate_rhi(
        temperature,
        humidity
    )

    tcritical = calculate_tcritical(
        pressure_level
    )

    delta_t = calculate_delta_t(
        temperature,
        tcritical
    )

    wind_speed = calculate_wind_speed(
        u,
        v
    )

    wind_shear = calculate_wind_shear()

    static_stability = calculate_static_stability()

    issr_depth = calculate_issr_depth(
        rhi
    )

    return {
        "temperature": temperature,
        "pressure_level": pressure_level,
        "rhi": rhi,
        "delta_t": delta_t,
        "wind_speed": wind_speed,
        "wind_shear": wind_shear,
        "static_stability": static_stability,
        "issr_depth": issr_depth
    }


# ==========================================================
# Testing
# ==========================================================

if __name__ == "__main__":

    sample_weather = {
        "temperature": 218.4,
        "humidity": 99.7,
        "u_wind": 30.3,
        "v_wind": 19.8,
        "vertical_velocity": 0.08,
        "geopotential": 102330
    }

    features = build_features(
        sample_weather,
        250
    )

    print(features)