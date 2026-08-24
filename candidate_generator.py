import math


# ----------------------------------------
# Normalize a vector
# ----------------------------------------

def normalize(x, y):

    magnitude = math.sqrt(x * x + y * y)

    if magnitude == 0:
        return 0, 0

    return x / magnitude, y / magnitude


# ----------------------------------------
# Generate candidate waypoints
# ----------------------------------------

def generate_candidates(
    current_lat,
    current_lon,
    current_alt,
    next_lat,
    next_lon
):
    """
    Generate candidate points around a waypoint.

    Altitude is in hPa.
    """

    # Direction of travel

    dx = next_lon - current_lon
    dy = next_lat - current_lat

    dx, dy = normalize(dx, dy)

    # Left vector (90° rotation)

    left_x = -dy
    left_y = dx

    # Right vector

    right_x = dy
    right_y = -dx

    # Amount of lateral movement

    OFFSET = 0.10

    candidates = [

        {
            "name": "Stay",
            "lat": current_lat,
            "lon": current_lon,
            "pressure": current_alt
        },

        {
            "name": "Left",
            "lat": current_lat + OFFSET * left_y,
            "lon": current_lon + OFFSET * left_x,
            "pressure": current_alt
        },

        {
            "name": "Right",
            "lat": current_lat + OFFSET * right_y,
            "lon": current_lon + OFFSET * right_x,
            "pressure": current_alt
        },

        {
            "name": "Climb",
            "lat": current_lat,
            "lon": current_lon,
            "pressure": current_alt - 25
        },

        {
            "name": "Descend",
            "lat": current_lat,
            "lon": current_lon,
            "pressure": current_alt + 25
        },

        {
            "name": "Left + Climb",
            "lat": current_lat + OFFSET * left_y,
            "lon": current_lon + OFFSET * left_x,
            "pressure": current_alt - 25
        },

        {
            "name": "Right + Climb",
            "lat": current_lat + OFFSET * right_y,
            "lon": current_lon + OFFSET * right_x,
            "pressure": current_alt - 25
        },

        {
            "name": "Left + Descend",
            "lat": current_lat + OFFSET * left_y,
            "lon": current_lon + OFFSET * left_x,
            "pressure": current_alt + 25
        },

        {
            "name": "Right + Descend",
            "lat": current_lat + OFFSET * right_y,
            "lon": current_lon + OFFSET * right_x,
            "pressure": current_alt + 25
        }

    ]

    return candidates


# ----------------------------------------
# Testing
# ----------------------------------------

if __name__ == "__main__":

    candidates = generate_candidates(

        current_lat=50.0,
        current_lon=5.0,
        current_alt=250,

        next_lat=50.5,
        next_lon=5.8

    )

    for candidate in candidates:

        print(candidate)