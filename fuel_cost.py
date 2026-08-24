"""
fuel_cost.py
============
Simple heuristic fuel cost model for comparing route candidates.

Two components:
  1. Altitude change cost — climbing burns meaningfully more fuel than
     staying level; descending is cheaper (idle/reduced thrust), so the
     two are NOT symmetric even though candidate_routes.py generates
     +/-25 hPa candidates symmetrically.
  2. Lateral diversion cost — any lateral (Left/Right) offset adds extra
     great-circle distance to the segment, burned at a cruise fuel rate.

These are deliberately simple, tunable heuristics (kg-of-fuel-equivalent
units), not a full performance model — good enough to trade off against
p_contrail in a combined cost function. Swap in a real fuel-burn model
later (e.g. BADA) without changing the optimizer's interface.
"""

import math

# ============================================================
# TUNABLE CONSTANTS (kg fuel equivalent — adjust to taste)
# ============================================================

# Climbing burns more fuel per hPa of pressure decrease than staying
# level costs. Values are illustrative, not from a real performance model.
CLIMB_COST_PER_HPA = 8.0     # kg fuel per hPa climbed
DESCEND_COST_PER_HPA = 2.0   # kg fuel per hPa descended (cheaper: reduced thrust)

# Extra lateral distance burned at a typical cruise fuel flow rate.
CRUISE_FUEL_PER_KM = 3.0     # kg fuel per km of extra distance flown

EARTH_RADIUS_KM = 6371.0


# ============================================================
# Great-circle distance (Haversine)
# ============================================================

def haversine_km(lat1, lon1, lat2, lon2):
    lat1_r, lon1_r = math.radians(lat1), math.radians(lon1)
    lat2_r, lon2_r = math.radians(lat2), math.radians(lon2)

    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r

    a = (math.sin(dlat / 2) ** 2
         + math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2) ** 2)
    c = 2 * math.asin(math.sqrt(a))

    return EARTH_RADIUS_KM * c


# ============================================================
# Altitude change cost
# ============================================================

def altitude_cost(current_pressure_hpa, candidate_pressure_hpa):
    """
    Pressure DECREASE = climb (higher altitude).
    Pressure INCREASE = descend (lower altitude).
    """
    delta = candidate_pressure_hpa - current_pressure_hpa

    if delta < 0:
        # Climbing (pressure went down)
        return CLIMB_COST_PER_HPA * abs(delta)
    elif delta > 0:
        # Descending (pressure went up)
        return DESCEND_COST_PER_HPA * delta
    else:
        return 0.0


# ============================================================
# Lateral diversion cost
# ============================================================

def lateral_cost(current_lat, current_lon, candidate_lat, candidate_lon):
    """
    Extra distance from diverting laterally, relative to going straight
    from the current point. Approximated as the direct distance from the
    current point to the candidate point — a reasonable proxy for extra
    track-miles when offsets are small (as they are here, ~0.10 deg).
    """
    extra_km = haversine_km(current_lat, current_lon,
                             candidate_lat, candidate_lon)
    return CRUISE_FUEL_PER_KM * extra_km


# ============================================================
# Combined fuel cost for one candidate
# ============================================================

def compute_fuel_cost(current, candidate):
    """
    current   : dict with 'lat', 'lon', 'pressure' (current waypoint state)
    candidate : dict with 'lat', 'lon', 'pressure' (from generate_candidates)

    Returns fuel cost in kg-equivalent units (relative, for comparison
    between candidates — not a calibrated absolute fuel burn figure).
    """
    alt_cost = altitude_cost(current["pressure"], candidate["pressure"])
    lat_cost = lateral_cost(
        current["lat"], current["lon"],
        candidate["lat"], candidate["lon"]
    )
    return alt_cost + lat_cost


# ============================================================
# Testing
# ============================================================

if __name__ == "__main__":
    current = {"lat": 50.0, "lon": 5.0, "pressure": 250}

    test_candidates = [
        {"name": "Stay",   "lat": 50.0,  "lon": 5.0,  "pressure": 250},
        {"name": "Climb",  "lat": 50.0,  "lon": 5.0,  "pressure": 225},
        {"name": "Descend","lat": 50.0,  "lon": 5.0,  "pressure": 275},
        {"name": "Left",   "lat": 50.07, "lon": 4.93, "pressure": 250},
    ]

    for c in test_candidates:
        cost = compute_fuel_cost(current, c)
        print(f"{c['name']:10s}  fuel_cost = {cost:.2f} kg-equiv")