"""
route_optimizer.py
===================
Ties together:
  - candidate_routes.generate_candidates()  -> lateral/altitude options
  - fuel_cost.compute_fuel_cost()           -> fuel penalty per candidate
  - route_pipeline.predict_waypoint()       -> p_contrail per candidate

For each segment of a real flight track (from route_pipeline's flight
lookup), generates candidates around the current waypoint, scores each
by a weighted combination of contrail probability and fuel cost, and
greedily picks the lowest-cost candidate. Reports the optimized path
alongside the original track for comparison.

NOTE ON METHOD: this is a GREEDY per-segment optimizer — it picks the
locally-best candidate at each step without looking ahead. It does not
guarantee a globally optimal path (a full dynamic-programming or A*
formulation over the whole candidate graph would), but it's simple,
fast, and good enough to demonstrate meaningful contrail-vs-fuel
trade-offs across a route. Mention this as a known limitation / future
work if asked at viva.
"""

import numpy as np
import pandas as pd
import joblib
from datetime import datetime, timezone

from route_pipeline import predict_waypoint, MODEL_FILE
from candidate_generator import generate_candidates
from fuel_cost import compute_fuel_cost

# ============================================================
# CONFIG
# ============================================================

# Weight trading off contrail avoidance vs fuel cost.
# cost = CONTRAIL_WEIGHT * p_contrail + FUEL_WEIGHT * fuel_cost_kg
#
# p_contrail is in [0, 1]; fuel_cost is in kg-equivalent (can be tens to
# hundreds). CONTRAIL_WEIGHT is scaled up so contrail avoidance actually
# matters in the comparison rather than being drowned out by fuel units.
CONTRAIL_WEIGHT = 200.0   # "kg-equivalent" penalty per unit of p_contrail
FUEL_WEIGHT = 1.0


# ============================================================
# Score one candidate
# ============================================================

def score_candidate(current_state, candidate, timestamp, model):
    """
    current_state : dict with 'lat', 'lon', 'pressure' (hPa) — the
                    waypoint this candidate is relative to.
    candidate     : dict from generate_candidates() with 'name', 'lat',
                    'lon', 'pressure'.
    timestamp     : datetime for weather lookup.
    model         : loaded contrail regressor.

    Returns dict with p_contrail, fuel_cost, total_cost, and the
    original candidate fields.
    """
    # predict_waypoint expects geometric altitude in metres, not
    # pressure in hPa — convert candidate pressure back to an
    # approximate altitude so it queries the right ERA5 level.
    altitude_m = pressure_to_altitude(candidate["pressure"])

    result = predict_waypoint(
        lat=candidate["lat"],
        lon=candidate["lon"],
        altitude_m=altitude_m,
        timestamp=timestamp,
        model=model,
    )

    fuel = compute_fuel_cost(current_state, candidate)
    p_contrail = result["p_contrail"]

    total_cost = CONTRAIL_WEIGHT * p_contrail + FUEL_WEIGHT * fuel

    return {
        **candidate,
        "p_contrail": p_contrail,
        "fuel_cost": fuel,
        "total_cost": total_cost,
        "temperature": result["temperature"],
    }


# ============================================================
# Pressure <-> altitude helper (inverse of route_pipeline's
# altitude_to_pressure, needed because candidates are generated
# in pressure space but predict_waypoint takes geometric altitude)
# ============================================================

def pressure_to_altitude(pressure_hpa):
    """
    Inverse ISA formula: approximate geometric altitude (m) from
    pressure (hPa). Mirrors route_pipeline.altitude_to_pressure().
    """
    T0, L, R, g, M = 288.15, 0.0065, 8.314, 9.80665, 0.029
    p0 = 1013.25

    if pressure_hpa > 226.32:
        # Troposphere branch
        exponent = (R * L) / (g * M)
        altitude = (T0 / L) * (1 - (pressure_hpa / p0) ** exponent)
    else:
        # Stratosphere approximation
        altitude = 11000 - (R * 216.65 / (g * M)) * np.log(pressure_hpa / 226.32)

    return float(altitude)


# ============================================================
# Optimize one segment (current waypoint -> next waypoint)
# ============================================================

def optimize_segment(current_lat, current_lon, current_pressure,
                      next_lat, next_lon, timestamp, model):
    """
    Generates candidates around the current waypoint (aimed toward the
    next one), scores each, and returns the best (lowest total_cost)
    candidate plus the full scored list for inspection/plotting.
    """
    current_state = {
        "lat": current_lat, "lon": current_lon, "pressure": current_pressure
    }

    candidates = generate_candidates(
        current_lat=current_lat,
        current_lon=current_lon,
        current_alt=current_pressure,
        next_lat=next_lat,
        next_lon=next_lon,
    )

    scored = [
        score_candidate(current_state, c, timestamp, model)
        for c in candidates
    ]

    best = min(scored, key=lambda c: c["total_cost"])
    return best, scored


# ============================================================
# Optimize a full track
# ============================================================

def optimize_route(track, model, altitude_to_pressure_fn, n_segments=None):
    """
    track : list of waypoint dicts with 'latitude', 'longitude',
            'baro_altitude' (m), 'timestamp' (unix seconds) — e.g. the
            output of route_pipeline's flight lookup, already filtered
            to cruise-altitude points.
    model : loaded contrail regressor.
    altitude_to_pressure_fn : route_pipeline.altitude_to_pressure
    n_segments : optionally cap the number of segments processed
                 (useful for quick tests on long tracks).

    Returns a DataFrame with one row per segment: the original
    waypoint, the chosen candidate, its p_contrail/fuel_cost, and the
    full candidate comparison for that segment.
    """
    points = track if n_segments is None else track[:n_segments + 1]

    results = []

    for i in range(len(points) - 1):
        cur = points[i]
        nxt = points[i + 1]

        cur_pressure = altitude_to_pressure_fn(cur["baro_altitude"] or 0)
        ts = datetime.fromtimestamp(
            cur.get("timestamp", 0) or 0, tz=timezone.utc
        ).replace(tzinfo=None) if cur.get("timestamp") else datetime.now()

        best, scored = optimize_segment(
            current_lat=cur["latitude"],
            current_lon=cur["longitude"],
            current_pressure=cur_pressure,
            next_lat=nxt["latitude"],
            next_lon=nxt["longitude"],
            timestamp=ts,
            model=model,
        )

        print(f"\nSegment {i+1}/{len(points)-1} | "
              f"({cur['latitude']:.3f}, {cur['longitude']:.3f}) @ "
              f"{cur_pressure:.0f} hPa")
        for c in sorted(scored, key=lambda x: x["total_cost"]):
            marker = "  <-- CHOSEN" if c["name"] == best["name"] else ""
            print(f"  {c['name']:16s} p_contrail={c['p_contrail']:.3f}  "
                  f"fuel={c['fuel_cost']:7.2f}kg  "
                  f"total_cost={c['total_cost']:8.2f}{marker}")

        results.append({
            "segment": i + 1,
            "orig_lat": cur["latitude"],
            "orig_lon": cur["longitude"],
            "orig_pressure": cur_pressure,
            "chosen_option": best["name"],
            "chosen_lat": best["lat"],
            "chosen_lon": best["lon"],
            "chosen_pressure": best["pressure"],
            "p_contrail": best["p_contrail"],
            "fuel_cost": best["fuel_cost"],
            "total_cost": best["total_cost"],
        })

    return pd.DataFrame(results)


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    import sys
    from route_pipeline import (
        resolve_flight, get_live_aircraft, get_track, altitude_to_pressure,
    )

    print("Loading model...")
    model = joblib.load(MODEL_FILE)

    if len(sys.argv) == 2:
        icao24, callsign, track = resolve_flight(callsign=sys.argv[1])
    elif len(sys.argv) == 3:
        icao24, callsign, track = resolve_flight(
            dep_icao=sys.argv[1], arr_icao=sys.argv[2]
        )
    else:
        raise SystemExit(
            "Usage: python route_optimizer.py CALLSIGN\n"
            "   or: python route_optimizer.py DEP_ICAO ARR_ICAO"
        )

    # Keep only cruise-altitude points, matching route_pipeline's filter.
    cruise_points = [p for p in track if (p.get("baro_altitude") or 0) >= 9500]

    if len(cruise_points) < 2:
        raise SystemExit("Not enough cruise-altitude waypoints to optimize.")

    print(f"\nOptimizing {len(cruise_points) - 1} segments...")

    df_results = optimize_route(
        cruise_points, model, altitude_to_pressure
    )

    print("\n" + "=" * 60)
    print("OPTIMIZATION SUMMARY")
    print("=" * 60)
    print(f"Segments optimized     : {len(df_results)}")
    print(f"Mean p_contrail (opt)  : {df_results['p_contrail'].mean():.4f}")
    print(f"Total fuel cost (opt)  : {df_results['fuel_cost'].sum():.1f} kg-equiv")
    print(f"Options chosen         :")
    print(df_results['chosen_option'].value_counts().to_string())

    df_results.to_csv("optimized_route.csv", index=False)
    print("\nSaved optimized_route.csv")