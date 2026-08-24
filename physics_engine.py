import numpy as np

# ==========================================================
# Constants
# ==========================================================

GRAVITY = 9.81           # m/s²
ENGINE_EFFICIENCY = 0.33 # Representative turbofan
CP = 1004                # J/(kg·K)
R = 287                  # J/(kg·K)

# ==========================================================
# Murphy & Koop (2005)
# Relative Humidity with respect to Ice
# ==========================================================

def calculate_rhi(temperature, relative_humidity):
    """
    temperature : Kelvin
    relative_humidity : RH with respect to water (%)

    Returns:
        Relative Humidity with respect to Ice (%)
    """
    relative_humidity = np.clip(relative_humidity, 0, 100)
    T = temperature

    # Saturation vapor pressure over ICE (Pa)
    es_ice = np.exp(
        9.550426
        - (5723.265 / T)
        + 3.53068 * np.log(T)
        - 0.00728332 * T
    )

    # Saturation vapor pressure over LIQUID WATER (Pa)
    es_water = np.exp(
        54.842763
        - (6763.22 / T)
        - 4.210 * np.log(T)
        + 0.000367 * T
        + np.tanh(0.0415 * (T - 218.8))
        * (
            53.878
            - (1331.22 / T)
            - 9.44523 * np.log(T)
            + 0.014025 * T
        )
    )

    # Convert RH(water) -> RH(ice)
    rhi = relative_humidity * (es_water / es_ice)

    return rhi


# ==========================================================
# Schumann Critical Temperature (Simplified)
# ==========================================================

def calculate_tcritical(pressure_hpa):
    """
    Returns Critical Temperature (Kelvin)
    """

    return 233 - 0.0065 * (250 - pressure_hpa)
    

# ==========================================================
# Delta T
# ==========================================================
def calculate_delta_t(temperature, tcritical):

    tcritical = tcritical.reshape(1, -1, 1, 1)

    return temperature - tcritical

# ==========================================================
# Wind Speed
# ==========================================================

def calculate_wind_speed(u, v):

    return np.sqrt(u**2 + v**2)


# ==========================================================
# Wind Shear
# ==========================================================

def calculate_wind_shear(u, v, pressure_levels):
    """
    Approximate vertical wind shear
    """

    du = np.gradient(u, axis=1)
    dv = np.gradient(v, axis=1)

    dp = np.gradient(pressure_levels)

    shear = np.sqrt(
        (du / dp.reshape(1, -1, 1, 1))**2 +
        (dv / dp.reshape(1, -1, 1, 1))**2
    )

    return shear


# ==========================================================
# Static Stability
# ==========================================================

def calculate_static_stability(temperature, geopotential, pressure_levels):

    height = geopotential / GRAVITY

    theta = temperature * (1000 / pressure_levels.reshape(1, -1, 1, 1)) ** 0.286

    dtheta = np.gradient(theta, axis=1)
    dz = np.gradient(height, axis=1)

    stability = (GRAVITY / theta) * (dtheta / dz)

    return stability


# ==========================================================
# ISSR Depth
# ==========================================================

def calculate_issr_depth(rhi):

    issr = rhi > 100

    depth = np.sum(issr, axis=1)

    return depth