"""
route_pipeline.py  —  Revised (patched)
========================================
Key fixes from original:
  1. Queries MULTIPLE pressure levels per waypoint (not just 250 hPa)
     so wind_shear and static_stability can be computed as real
     vertical gradients, not single-level constants.
  2. Adds omega (vertical velocity) at inference time.
  3. Adds latitude, sin_month, cos_month, tropopause_dist features.
  4. Uses XGBRegressor (continuous probability) for route scoring.
  5. Estimates actual aircraft pressure level from baro_altitude
     instead of hardcoding 250 hPa.
  6. Prints feature values at every waypoint so flat-output
     debugging is straightforward.

PATCHES APPLIED (this version):
  P1. Cruise-altitude filter raised 8000m -> 9500m. Below ~9500m
      (~FL310) aircraft are typically still climbing, and the
      resulting pressure (~340-380 hPa) falls OUTSIDE the ERA5
      domain (150-300 hPa) the training data and QUERY_LEVELS cover.
  P2. static_stability now has one-sided edge-level fallbacks
      (previously silently returned 0.0 whenever target_level was
      the first/last entry in QUERY_LEVELS — which was ALWAYS the
      case for real aircraft, since their true pressure sits above
      300 hPa and gets clamped to the 300 hPa edge).
  P3. static_stability now includes the (GRAVITY / theta) factor to
      match physics_engine.py exactly. Without it, route-computed
      static_stability was ~10-40x larger than the values the model
      was actually trained on (3e-3 at inference vs 1e-4 in training).
  P4. wind_shear no longer converts dP to Pa (removed the "* 100").
      physics_engine.py computes dp = np.gradient(pressure_levels) in
      hPa with no conversion, so route_pipeline must match — the Pa
      conversion made route wind_shear ~100x smaller than training.
  P5. Flight selection now accepts a callsign/flight number OR a
      departure/arrival airport pair (as promised in project reports),
      resolved via OpenSky's /flights endpoints. Wherever possible it
      prefers an ALREADY-LANDED flight and pulls its full track via
      /tracks/all using a timestamp from within that flight — not
      time=0, which only returns "track so far" and truncates
      mid-flight for aircraft still airborne (this was why waypoints
      were clustering: the live aircraft had JUST reached cruise when
      the API was called, so no varied cruise-phase history existed
      yet to sample from).
"""

import numpy as np
import pandas as pd
import joblib
from datetime import datetime, timezone

from opensky_service import (
    get_live_aircraft,
    get_track,
    search_flight_by_callsign,
    search_flight_by_route,
    get_full_track,
)
from weather_query import get_weather          # you may need to extend this
from point_physics import build_features       # unused directly, kept for parity

# ============================================================
# CONFIG
# ============================================================

MODEL_FILE = "contrail_model_reg.pkl"   # regressor gives continuous prob

FEATURES = [
    "temperature",
    "pressure_level",
    "wind_speed",
    "wind_shear",
    "static_stability",
    "omega",
    "latitude",
    "sin_month",
    "cos_month",
    "tropopause_dist",
]

GRAVITY = 9.80665

# Pressure levels to query for vertical gradient computation.
# wind_shear and static_stability REQUIRE at least two levels.
QUERY_LEVELS = [200, 225, 250, 275, 300]   # hPa
# NOTE: confirm these match era5_2023_full.nc's actual pressure_level
# values exactly (run `print(ds.pressure_level.values)` in
# weather_query.py). If the real file has 150/175 hPa levels too,
# add them here so aircraft near the top of QUERY_LEVELS also get
# an interior (non-edge) gradient.


# ============================================================
# HELPER: Convert altitude to pressure level
# ============================================================

def altitude_to_pressure(altitude_m: float) -> float:
    """
    ISA standard atmosphere: convert geometric altitude (metres)
    to approximate pressure (hPa).

    Used to pick the closest ERA5 pressure level to the aircraft.
    """
    # International Standard Atmosphere formula
    T0, L, R, g, M = 288.15, 0.0065, 8.314, 9.80665, 0.029
    if altitude_m <= 11000:
        p = 1013.25 * (1 - L * altitude_m / T0) ** (g * M / (R * L))
    else:
        # Stratosphere approximation
        p = 226.32 * np.exp(-g * M * (altitude_m - 11000) / (R * 216.65))
    return float(p)


def closest_level(pressure_hpa: float, levels=QUERY_LEVELS) -> int:
    return min(levels, key=lambda x: abs(x - pressure_hpa))


# ============================================================
# HELPER: Tropopause distance
# ============================================================

def approx_tropopause_pressure(lat: float) -> float:
    abs_lat = abs(lat)
    if abs_lat <= 30:
        return 100 + (abs_lat / 30) * 50
    elif abs_lat <= 60:
        return 150 + ((abs_lat - 30) / 30) * 100
    else:
        return 250 + ((abs_lat - 60) / 30) * 50


# ============================================================
# CORE INFERENCE FUNCTION
# ============================================================

def predict_waypoint(
    lat: float,
    lon: float,
    altitude_m: float,
    timestamp: datetime,
    model,
) -> dict:
    """
    Given a single waypoint, fetch multi-level weather,
    compute physics features, and return contrail probability.

    Parameters
    ----------
    lat, lon     : degrees
    altitude_m   : geometric altitude in metres (from ADS-B)
    timestamp    : UTC datetime of the waypoint
    model        : loaded XGBRegressor

    Returns
    -------
    dict with all feature values and final probability
    """
    # --- Determine aircraft pressure level ---
    aircraft_p = altitude_to_pressure(altitude_m)
    target_level = closest_level(aircraft_p)

    # --- Fetch weather at MULTIPLE levels ---
    # This is the key fix: wind_shear and static_stability
    # need at least two adjacent levels to be meaningful.
    level_data = {}
    for level in QUERY_LEVELS:
        level_data[level] = get_weather(
            latitude=lat,
            longitude=lon,
            pressure_level=level,
            valid_time=timestamp,
        )

    # --- Build physics features using multi-level data ---
    features = build_features_multilevel(
        level_data=level_data,
        target_level=target_level,
    )

    # --- Temporal features ---
    month = timestamp.month
    sin_month = np.sin(2 * np.pi * month / 12)
    cos_month = np.cos(2 * np.pi * month / 12)

    # --- Tropopause distance ---
    trop_p = approx_tropopause_pressure(lat)
    tropopause_dist = target_level - trop_p

    # --- Assemble feature row ---
    X = pd.DataFrame([{
        "temperature":     features["temperature"],
        "pressure_level":  float(target_level),
        "wind_speed":      features["wind_speed"],
        "wind_shear":      features["wind_shear"],
        "static_stability": features["static_stability"],
        "omega":           features.get("omega", 0.0),
        "latitude":        lat,
        "sin_month":       sin_month,
        "cos_month":       cos_month,
        "tropopause_dist": tropopause_dist,
    }])

    # --- Predict ---
    probability = float(model.predict(X)[0])
    probability = np.clip(probability, 0.0, 1.0)

    return {
        "latitude":         lat,
        "longitude":        lon,
        "altitude_m":       altitude_m,
        "pressure_level":   target_level,
        "temperature":      features["temperature"],
        "wind_speed":       features["wind_speed"],
        "wind_shear":       features["wind_shear"],
        "static_stability": features["static_stability"],
        "omega":            features.get("omega", 0.0),
        "tropopause_dist":  tropopause_dist,
        "sin_month":        sin_month,
        "p_contrail":       probability,
        "prediction":       int(probability > 0.5),
    }


# ============================================================
# build_features_multilevel  (PATCHED: P2, P3, P4)
# ============================================================

def build_features_multilevel(level_data: dict, target_level: int) -> dict:
    """
    Compute physics features from multi-level weather data.

    Key mapping from get_weather() response:
        'u_wind'           -> u component (m/s)
        'v_wind'           -> v component (m/s)
        'vertical_velocity'-> omega (Pa/s)
        'geopotential'     -> geopotential (m^2/s^2)
        'temperature'      -> T (K)

    Units are matched to physics_engine.py (used at training time):
      - wind_shear:      d(wind)/dp, with dp in hPa (NOT Pa)
      - static_stability: (g / theta) * d(theta)/dz
    """
    levels = sorted(level_data.keys())
    t_idx  = levels.index(target_level)

    tgt = level_data[target_level]

    # --- Basic fields at target level ---
    T   = tgt["temperature"]
    u   = tgt["u_wind"]
    v   = tgt["v_wind"]
    om  = tgt.get("vertical_velocity", 0.0)

    wind_speed = np.sqrt(u**2 + v**2)

    # --- Wind shear across adjacent levels ---
    # P4: dP kept in hPa to match physics_engine.py's
    # dp = np.gradient(pressure_levels) (no *100 Pa conversion).
    if t_idx > 0 and t_idx < len(levels) - 1:
        above = level_data[levels[t_idx - 1]]
        below = level_data[levels[t_idx + 1]]
        dP    = (levels[t_idx + 1] - levels[t_idx - 1])   # hPa
        du    = below["u_wind"] - above["u_wind"]
        dv    = below["v_wind"] - above["v_wind"]
        wind_shear = np.sqrt((du / dP)**2 + (dv / dP)**2)

    elif t_idx > 0:
        above = level_data[levels[t_idx - 1]]
        dP    = (target_level - levels[t_idx - 1])        # hPa
        du    = u - above["u_wind"]
        dv    = v - above["v_wind"]
        wind_shear = np.sqrt((du / dP)**2 + (dv / dP)**2)

    elif t_idx < len(levels) - 1:
        below = level_data[levels[t_idx + 1]]
        dP    = (levels[t_idx + 1] - target_level)         # hPa
        du    = below["u_wind"] - u
        dv    = below["v_wind"] - v
        wind_shear = np.sqrt((du / dP)**2 + (dv / dP)**2)

    else:
        wind_shear = 0.0

    # --- Static stability (potential temperature gradient) ---
    # P2: edge levels now get a one-sided fallback instead of 0.0.
    # P3: multiplied by (GRAVITY / theta_at_target) to match
    #     physics_engine.py's stability = (GRAVITY / theta) * (dtheta/dz).
    def theta(T_k, p_hpa):
        return T_k * (1000.0 / p_hpa) ** 0.286

    th_tgt = theta(T, target_level)

    if t_idx > 0 and t_idx < len(levels) - 1:
        above = level_data[levels[t_idx - 1]]
        below = level_data[levels[t_idx + 1]]

        th_above = theta(above["temperature"], levels[t_idx - 1])
        th_below = theta(below["temperature"], levels[t_idx + 1])

        z_above  = above["geopotential"] / GRAVITY
        z_below  = below["geopotential"] / GRAVITY
        dz       = z_above - z_below

        static_stability = (GRAVITY / th_tgt) * ((th_above - th_below) / dz) \
            if abs(dz) > 1 else 0.0

    elif t_idx > 0:
        above = level_data[levels[t_idx - 1]]
        th_above = theta(above["temperature"], levels[t_idx - 1])
        z_above  = above["geopotential"] / GRAVITY
        z_tgt    = tgt["geopotential"] / GRAVITY
        dz       = z_above - z_tgt
        static_stability = (GRAVITY / th_tgt) * ((th_above - th_tgt) / dz) \
            if abs(dz) > 1 else 0.0

    elif t_idx < len(levels) - 1:
        below = level_data[levels[t_idx + 1]]
        th_below = theta(below["temperature"], levels[t_idx + 1])
        z_below  = below["geopotential"] / GRAVITY
        z_tgt    = tgt["geopotential"] / GRAVITY
        dz       = z_tgt - z_below
        static_stability = (GRAVITY / th_tgt) * ((th_tgt - th_below) / dz) \
            if abs(dz) > 1 else 0.0

    else:
        static_stability = 0.0

    return {
        "temperature":      T,
        "wind_speed":       wind_speed,
        "wind_shear":       wind_shear,
        "static_stability": static_stability,
        "omega":            om,
    }


# ============================================================
# MAIN PIPELINE
# ============================================================

def resolve_flight(callsign=None, dep_icao=None, arr_icao=None, hours_back=None):
    """
    Resolve a callsign/flight number OR a departure->arrival airport pair
    to an icao24 + a full track, preferring an already-LANDED flight so
    /tracks/all returns the complete climb -> cruise -> descent path.

    hours_back: how far back to search. Defaults to 48h for callsign
    searches, 720h (30 days) for route searches. NOTE: OpenSky's free
    tier may not actually retain/serve flight history that far back
    regardless of what you request — if a wide search still returns
    nothing, that's a data-availability limit, not a code bug.

    Returns (icao24, callsign, track) or raises if nothing found.
    """
    import time as _time
    now = int(_time.time())

    candidates = []
    if callsign:
        hb = hours_back if hours_back is not None else 48
        print(f"Searching flights for callsign '{callsign}' "
              f"(last {hb}h)...")
        candidates = search_flight_by_callsign(callsign, hours_back=hb)
    elif dep_icao and arr_icao:
        hb = hours_back if hours_back is not None else 168
        print(f"Searching flights {dep_icao} -> {arr_icao} "
              f"(last {hb}h)...")
        candidates = search_flight_by_route(dep_icao, arr_icao, hours_back=hb)

    if not candidates:
        raise Exception(
            "No matching flights found for the given callsign/route in "
            "the searched window. Try a different flight number, widen "
            "hours_back, or confirm the ICAO airport codes are correct."
        )

    # Prefer flights that have already landed (lastSeen well in the past)
    # so the full track is available, not a still-in-progress one.
    landed = [f for f in candidates if f.get("lastSeen", 0) < now - 300]
    pick_from = landed if landed else candidates

    for f in pick_from:
        icao24 = f["icao24"]
        flight_time = f.get("lastSeen") or f.get("firstSeen")

        try:
            track = get_full_track(icao24, flight_time)
        except Exception as e:
            print(f"  Could not fetch full track for {icao24} "
                  f"({e}). Trying live track (time=0) as fallback...")
            try:
                track = get_track(icao24, time=0)
                print(f"  WARNING: using LIVE track (time=0), not the "
                      f"full landed-flight track. This may be a short, "
                      f"in-progress snapshot rather than the complete "
                      f"climb->cruise->descent path.")
            except Exception as e2:
                print(f"  Live track fallback also failed for {icao24}: "
                      f"{e2}. Trying next candidate...")
                continue

        if track:
            max_alt = max((p.get("baro_altitude") or 0) for p in track)
            if max_alt >= 9500:
                return icao24, f.get("callsign", "").strip(), track

    raise Exception(
        "Matching flights were found, but none had a track reaching "
        "cruise altitude (>=9500m). The flight may still be too short "
        "or track data may be incomplete for this aircraft."
    )


if __name__ == "__main__":

    import sys

    print("Loading model...")
    model = joblib.load(MODEL_FILE)

    # --- Flight selection ---
    # Usage:
    #   python route_pipeline.py CALLSIGN
    #   python route_pipeline.py DEP_ICAO ARR_ICAO
    #   python route_pipeline.py                    (falls back to live traffic)
    selected_callsign = None

    if len(sys.argv) == 2:
        icao24, selected_callsign, track = resolve_flight(callsign=sys.argv[1])
        print(f"\nSelected: {selected_callsign} ({icao24}) — "
              f"track max altitude: "
              f"{max((p.get('baro_altitude') or 0) for p in track):.0f} m")

    elif len(sys.argv) == 3:
        icao24, selected_callsign, track = resolve_flight(
            dep_icao=sys.argv[1], arr_icao=sys.argv[2]
        )
        print(f"\nSelected: {selected_callsign} ({icao24}) — "
              f"{sys.argv[1]} -> {sys.argv[2]} — track max altitude: "
              f"{max((p.get('baro_altitude') or 0) for p in track):.0f} m")

    else:
        print("No flight number or route given — falling back to live "
              "traffic search. (Usage: python route_pipeline.py CALLSIGN "
              "  |  python route_pipeline.py DEP_ICAO ARR_ICAO)")
        print("Fetching live aircraft...")
        aircraft = get_live_aircraft(100)

        selected = None
        track = None
        for a in aircraft:
            current_alt = a.get("baro_altitude") or 0
            if current_alt < 9500:
                continue
            candidate_track = get_track(a["icao24"])
            if not candidate_track:
                continue
            max_track_alt = max((p.get("baro_altitude") or 0) for p in candidate_track)
            if max_track_alt >= 9500:
                selected = a
                track = candidate_track
                break

        if selected is None:
            raise Exception(
                "No aircraft with track waypoints above 9500m found in "
                "this batch. Try again shortly or increase "
                "get_live_aircraft(100) to a larger candidate pool."
            )

        icao24 = selected["icao24"]
        print(f"\nSelected: {icao24} at {selected['baro_altitude']:.0f} m "
              f"(track max: "
              f"{max((p.get('baro_altitude') or 0) for p in track):.0f} m)")

    print(f"Total waypoints: {len(track)}")

    results = []

    for i, point in enumerate(track):
        if point.get("latitude") is None:
            continue

        alt = point.get("baro_altitude") or 0

        # P1: Skip takeoff, climb, and descent waypoints.
        # Raised from 8000m -> 9500m: below ~9500m (~FL310) aircraft are
        # typically still climbing, and the resulting pressure
        # (~340-380 hPa) falls OUTSIDE the ERA5 domain (150-300 hPa)
        # the training data and QUERY_LEVELS cover.
        if alt < 9500:
            print(f"Waypoint {i+1}: skipping (altitude {alt:.0f}m — below cruise)")
            continue

        # Parse timestamp
        if "time" in point and point["time"]:
            ts = datetime.fromtimestamp(point["time"], tz=timezone.utc).replace(tzinfo=None)
        else:
            ts = datetime.now(tz=timezone.utc).replace(tzinfo=None)

        try:
            result = predict_waypoint(
                lat=point["latitude"],
                lon=point["longitude"],
                altitude_m=alt,
                timestamp=ts,
                model=model,
            )
            results.append(result)

            print(f"\n{'='*50}")
            print(f"Waypoint {i+1:2d} | "
                  f"({result['latitude']:7.3f}, {result['longitude']:8.3f}) | "
                  f"{alt:.0f} m → {result['pressure_level']} hPa")
            print(f"  Temperature    : {result['temperature']:.1f} K")
            print(f"  Wind Speed     : {result['wind_speed']:.1f} m/s")
            print(f"  Wind Shear     : {result['wind_shear']:.2e} s⁻¹")
            print(f"  Static Stab    : {result['static_stability']:.2e}")
            print(f"  Omega          : {result['omega']:.4f} Pa/s")
            print(f"  Trop Distance  : {result['tropopause_dist']:.1f} hPa")
            print(f"  ── P_contrail  : {result['p_contrail']:.4f}  "
                  f"→ {'CONTRAIL' if result['prediction'] else 'clear'}")

        except Exception as e:
            print(f"Waypoint {i+1}: ERROR — {e}")
            continue

    # Summary table
    if results:
        df_results = pd.DataFrame(results)
        print(f"\n{'='*50}")
        print("ROUTE SUMMARY")
        print(f"{'='*50}")
        print(f"Mean P_contrail  : {df_results['p_contrail'].mean():.4f}")
        print(f"Max  P_contrail  : {df_results['p_contrail'].max():.4f}")
        print(f"Min  P_contrail  : {df_results['p_contrail'].min():.4f}")
        print(f"Std  P_contrail  : {df_results['p_contrail'].std():.4f}")
        print(f"\nContrail waypoints: "
              f"{df_results['prediction'].sum()} / {len(df_results)}")

        df_results.to_csv("route_results.csv", index=False)
        print("\nSaved route_results.csv")