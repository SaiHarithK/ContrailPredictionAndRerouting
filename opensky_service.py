import json
import os
import time as _time
import requests

# ==========================================================
# CONFIG
# ==========================================================

TOKEN_URL = (
    "https://auth.opensky-network.org/auth/realms/"
    "opensky-network/protocol/openid-connect/token"
)

BASE_URL = "https://opensky-network.org/api"

# Path to the credentials.json you downloaded
CREDENTIALS_FILE = "opensky/credentials.json"

# ==========================================================
# LOCAL CACHE (avoids re-hitting the live API during dev/testing,
# and doubles as an offline fallback if OpenSky is rate-limited or
# unavailable right before a demo)
# ==========================================================

CACHE_DIR = "opensky_cache"
CACHE_TTL_SECONDS = 24 * 3600  # cached results are reused for 24h

os.makedirs(CACHE_DIR, exist_ok=True)


def _cache_path(key):
    safe_key = "".join(c if c.isalnum() or c in "-_" else "_" for c in key)
    return os.path.join(CACHE_DIR, f"{safe_key}.json")


def _cache_get(key):
    path = _cache_path(key)
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r") as f:
            entry = json.load(f)
        if _time.time() - entry["cached_at"] > CACHE_TTL_SECONDS:
            return None
        return entry["data"]
    except (json.JSONDecodeError, KeyError):
        return None


def _cache_set(key, data):
    path = _cache_path(key)
    with open(path, "w") as f:
        json.dump({"cached_at": _time.time(), "data": data}, f)


def cached_or_fetch(cache_key, fetch_fn, use_cache=True):
    """
    Returns cached data for cache_key if present and fresh; otherwise
    calls fetch_fn(), caches the result, and returns it.
    Set use_cache=False to force a live fetch (still updates the cache).
    """
    if use_cache:
        cached = _cache_get(cache_key)
        if cached is not None:
            print(f"  [cache hit] {cache_key}")
            return cached

    result = fetch_fn()
    _cache_set(cache_key, result)
    return result


# ==========================================================
# LOAD CREDENTIALS
# ==========================================================

with open(CREDENTIALS_FILE, "r") as f:
    creds = json.load(f)

CLIENT_ID = creds["clientId"]
CLIENT_SECRET = creds["clientSecret"]


# ==========================================================
# GET ACCESS TOKEN (cached in-memory — was being re-fetched on
# EVERY call, which contributed to hitting the rate limit)
# ==========================================================

_token_cache = {"token": None, "expires_at": 0}


def get_access_token():
    now = _time.time()

    if _token_cache["token"] and now < _token_cache["expires_at"]:
        return _token_cache["token"]

    response = requests.post(
        TOKEN_URL,
        data={
            "grant_type": "client_credentials",
            "client_id": CLIENT_ID,
            "client_secret": CLIENT_SECRET,
        },
        timeout=15,
    )

    response.raise_for_status()

    data = response.json()
    token = data["access_token"]
    expires_in = data.get("expires_in", 1800)
    _token_cache["token"] = token
    _token_cache["expires_at"] = now + max(expires_in - 60, 60)

    return token


# ==========================================================
# GET LIVE AIRCRAFT STATES
# ==========================================================

def get_live_aircraft(limit=10):

    token = get_access_token()

    response = requests.get(
        f"{BASE_URL}/states/all",
        headers={
            "Authorization": f"Bearer {token}"
        },
        timeout=20,
    )

    response.raise_for_status()

    data = response.json()

    states = data["states"]

    europe = []

    for s in states:

        lat = s[6]
        lon = s[5]

        if lat is None or lon is None:
            continue

        if 35 <= lat <= 60 and -10 <= lon <= 30:
            europe.append(s)

    states = europe

    aircraft = []

    for s in states:

        aircraft.append({
            "icao24": s[0],
            "callsign": s[1].strip() if s[1] else "",
            "country": s[2],
            "longitude": s[5],
            "latitude": s[6],
            "baro_altitude": s[7],
            "velocity": s[9],
            "heading": s[10],
        })

    return aircraft

# ==========================================================
# GET TRAJECTORY
# ==========================================================

def get_track(icao24, time=0, max_retries=4, use_cache=True):

    def _fetch():
        token = get_access_token()

        for attempt in range(max_retries):
            response = requests.get(
                f"{BASE_URL}/tracks/all",
                params={
                    "icao24": icao24,
                    "time": time
                },
                headers={
                    "Authorization": f"Bearer {token}"
                },
                timeout=20,
            )

            if response.status_code == 429:
                wait = 2 ** attempt  # 1, 2, 4, 8 seconds
                print(f"  Rate limited (429) on /tracks/all — "
                      f"retrying in {wait}s (attempt {attempt+1}/{max_retries})...")
                _time.sleep(wait)
                continue

            response.raise_for_status()
            data = response.json()

            path = []
            for p in data["path"]:
                path.append({
                    "timestamp": p[0],
                    "latitude": p[1],
                    "longitude": p[2],
                    "baro_altitude": p[3],
                    "true_track": p[4],
                    "on_ground": p[5]
                })

            return path

        raise Exception(
            f"/tracks/all rate limited after {max_retries} retries for "
            f"icao24={icao24}. Wait a bit before trying again."
        )

    cache_key = f"track_{icao24}_{time}"
    return cached_or_fetch(cache_key, _fetch, use_cache=use_cache)


# ==========================================================
# SEARCH FLIGHTS BY CALLSIGN (flight number)
# ==========================================================

def search_flight_by_callsign(callsign, hours_back=48, use_cache=True):
    """
    Find recent flights matching a callsign/flight number (e.g. "BAW123").

    Searches /flights/all over the given time window and filters by
    callsign client-side (OpenSky has no direct callsign search endpoint).

    Returns a list of flight dicts with icao24, callsign, firstSeen,
    lastSeen, estDepartureAirport, estArrivalAirport — ordered most
    recent first. Prefer flights where lastSeen is well in the past
    (i.e. the flight has LANDED), so /tracks/all can return the full
    climb -> cruise -> descent path instead of a still-in-progress one.

    Results are cached (refreshed every 6h) to avoid re-hitting the API
    on repeated dev/test runs.
    """
    now = int(_time.time())
    hour_bucket = now // (6 * 3600)
    cache_key = f"callsign_{callsign.strip().upper()}_{hours_back}_{hour_bucket}"

    def _fetch():
        token = get_access_token()
        begin = now - hours_back * 3600

        matches = []
        chunk = 2 * 3600
        t = begin
        while t < now:
            t_end = min(t + chunk, now)
            response = requests.get(
                f"{BASE_URL}/flights/all",
                params={"begin": t, "end": t_end},
                headers={"Authorization": f"Bearer {token}"},
                timeout=20,
            )
            if response.status_code == 200:
                data = response.json()
                for f in data:
                    cs = (f.get("callsign") or "").strip().upper()
                    if cs == callsign.strip().upper():
                        matches.append(f)
            elif response.status_code == 429:
                print(f"  Rate limited (429) on /flights/all — "
                      f"stopping search early with {len(matches)} matches so far.")
                break
            else:
                print(f"  WARNING: /flights/all returned "
                      f"{response.status_code}: {response.text[:200]}")
            t = t_end
            _time.sleep(0.3)

        print(f"  {len(matches)} flights matched callsign '{callsign}'.")
        matches.sort(key=lambda f: f.get("lastSeen", 0), reverse=True)
        return matches

    return cached_or_fetch(cache_key, _fetch, use_cache=use_cache)


# ==========================================================
# SEARCH FLIGHTS BY ROUTE (departure -> arrival airport)
# ==========================================================

def search_flight_by_route(dep_icao, arr_icao, hours_back=168, max_matches=5,
                            use_cache=True):
    """
    Find recent flights from departure airport (ICAO code, e.g. "EGLL")
    to arrival airport (ICAO code, e.g. "LFPG").

    Uses /flights/departure for the departure airport, then filters
    results client-side for matching estArrivalAirport.

    Results are cached (by route + day-bucket) for CACHE_TTL_SECONDS,
    so repeated dev/test runs don't re-hit the API and burn rate limit.
    """
    now = int(_time.time())
    day_bucket = now // (6 * 3600)  # cache refreshes every 6h at most
    cache_key = f"route_{dep_icao}_{arr_icao}_{hours_back}_{day_bucket}"

    def _fetch():
        token = get_access_token()
        begin = now - hours_back * 3600

        matches = []
        total_departures = 0

        DAY = 24 * 3600
        aligned_begin = (begin // DAY) * DAY
        t = aligned_begin
        while t < now:
            t_end = min(t + DAY, now)
            response = requests.get(
                f"{BASE_URL}/flights/departure",
                params={"airport": dep_icao, "begin": t, "end": t_end},
                headers={"Authorization": f"Bearer {token}"},
                timeout=20,
            )
            if response.status_code == 200:
                all_departures = response.json()
                total_departures += len(all_departures)
                print(f"  Window {t}-{t_end}: {len(all_departures)} "
                      f"departures from {dep_icao}.")
                for f in all_departures:
                    if f.get("estArrivalAirport") == arr_icao:
                        matches.append(f)
            elif response.status_code == 429:
                print(f"  Rate limited (429) on /flights/departure — "
                      f"stopping search early with {len(matches)} matches so far.")
                break
            else:
                print(f"  WARNING: /flights/departure returned "
                      f"{response.status_code}: {response.text[:200]}")

            t = t_end

            if len(matches) >= max_matches:
                print(f"  Found {len(matches)} matches — stopping search "
                      f"early (no need to burn through the full window).")
                break

            _time.sleep(0.3)

        print(f"  Total: {total_departures} departures from {dep_icao} found, "
              f"{len(matches)} matched arrival airport {arr_icao}.")
        matches.sort(key=lambda f: f.get("lastSeen", 0), reverse=True)
        return matches

    return cached_or_fetch(cache_key, _fetch, use_cache=use_cache)


# ==========================================================
# GET FULL TRACK FOR A COMPLETED FLIGHT
# ==========================================================

def get_full_track(icao24, flight_time):
    """
    Like get_track(), but pass a timestamp FROM WITHIN a specific
    (ideally already-landed) flight, e.g. flight["lastSeen"] from
    search_flight_by_callsign()/search_flight_by_route(). This returns
    that flight's complete climb -> cruise -> descent path, instead of
    time=0's "most recent track so far" (which truncates mid-flight
    for aircraft that are still airborne).
    """
    return get_track(icao24, time=flight_time)


# ==========================================================
# MAIN
# ==========================================================
if __name__ == "__main__":

    aircraft = get_live_aircraft(5)

    print("Available Aircraft\n")

    for i, a in enumerate(aircraft):
        print(i, a)

    print()

    icao24 = aircraft[0]["icao24"]

    print(f"Fetching trajectory for {icao24}...\n")

    track = get_track(icao24)

    print(f"Total Waypoints: {len(track)}\n")

    for p in track[:10]:
        print(p)