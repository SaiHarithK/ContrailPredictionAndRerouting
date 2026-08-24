import xarray as xr
import numpy as np
import pandas as pd
from physics_engine import (
    calculate_rhi,
    calculate_tcritical,
    calculate_delta_t,
    calculate_wind_speed,
    calculate_wind_shear,
    calculate_static_stability,
    calculate_issr_depth,
)

# ============================================================
# CONFIG
# ============================================================

ERA5_FILE   = "era5_2023_full.nc"
OUTPUT_FILE = "feature_dataset.parquet"
N_SAMPLES   = 600_000
RANDOM_SEED = 42

# Revised fractions — added warm_contrail stratum
# All five must sum to <= 1.0 (remainder goes to random)
STRAT_CONFIG = {
    "issr_fraction":          0.25,  # RHi > 100%
    "formation_fraction":     0.20,  # DeltaT < 0
    "joint_fraction":         0.10,  # DeltaT < 0 AND RHi > 100
    "warm_contrail_fraction": 0.15,  # T > 224K AND RHi > 110 AND DeltaT < 0
    # remaining 0.30 = pure random background
}


# ============================================================
# STEP 1: LOAD DATA
# ============================================================

print("=" * 55)
print("LOADING ERA5 DATASET")
print("=" * 55)

ds = xr.open_dataset(ERA5_FILE)
print(f"Dataset dimensions: {dict(ds.dims)}")

temperature  = ds["t"].values.astype(np.float32)
humidity     = ds["r"].values.astype(np.float32)
u            = ds["u"].values.astype(np.float32)
v            = ds["v"].values.astype(np.float32)
omega        = ds["w"].values.astype(np.float32)
geopotential = ds["z"].values.astype(np.float32)
pressure     = ds["pressure_level"].values.astype(np.float32)
latitudes    = ds["latitude"].values.astype(np.float32)
longitudes   = ds["longitude"].values.astype(np.float32)

ds.close()

t_dim, p_dim, lat_dim, lon_dim = temperature.shape
total_points = t_dim * p_dim * lat_dim * lon_dim

print(f"\nArray shape : {temperature.shape}")
print(f"Total points: {total_points:,}  (~{total_points/1e6:.1f}M)")
print(f"Per-array RAM (float32): {temperature.nbytes / 1e9:.2f} GB")


# ============================================================
# STEP 2: COMPUTE PHYSICS FEATURES
# ============================================================

print("\n" + "=" * 55)
print("COMPUTING PHYSICS FEATURES")
print("=" * 55)

rhi        = calculate_rhi(temperature, humidity)
tcritical  = calculate_tcritical(pressure)
delta_t    = calculate_delta_t(temperature, tcritical)
wind_speed = calculate_wind_speed(u, v)
wind_shear = calculate_wind_shear(u, v, pressure)
stability  = calculate_static_stability(temperature, geopotential, pressure)
issr_depth = calculate_issr_depth(rhi)

del humidity, u, v, omega, geopotential
import gc; gc.collect()

print("Physics features computed.")
print(f"  temperature  : {temperature.shape}")
print(f"  rhi          : {rhi.shape}")
print(f"  delta_t      : {delta_t.shape}")
print(f"  wind_speed   : {wind_speed.shape}")
print(f"  wind_shear   : {wind_shear.shape}")
print(f"  stability    : {stability.shape}")
print(f"  issr_depth   : {issr_depth.shape}  (3D — no pressure dim)")


# ============================================================
# STEP 3: STRATIFIED INDEX SAMPLING
# ============================================================

print("\n" + "=" * 55)
print("STRATIFIED INDEX SAMPLING")
print("=" * 55)

rng = np.random.default_rng(RANDOM_SEED)


def _sample_indices(mask_4d, n, rng):
    valid = np.argwhere(mask_4d)
    if len(valid) == 0:
        return np.empty((0, 4), dtype=np.intp)
    n = min(n, len(valid))
    chosen = rng.choice(len(valid), size=n, replace=False)
    return valid[chosen]


def _random_indices(n, shape, rng):
    t   = rng.integers(0, shape[0], size=n)
    p   = rng.integers(0, shape[1], size=n)
    lat = rng.integers(0, shape[2], size=n)
    lon = rng.integers(0, shape[3], size=n)
    return np.column_stack([t, p, lat, lon])


n_issr          = int(N_SAMPLES * STRAT_CONFIG["issr_fraction"])
n_formation     = int(N_SAMPLES * STRAT_CONFIG["formation_fraction"])
n_joint         = int(N_SAMPLES * STRAT_CONFIG["joint_fraction"])
n_warm_contrail = int(N_SAMPLES * STRAT_CONFIG["warm_contrail_fraction"])
n_random        = N_SAMPLES - n_issr - n_formation - n_joint - n_warm_contrail

print(f"  ISSR stratum          : {n_issr:,} samples  (RHi > 100%)")
print(f"  Formation stratum     : {n_formation:,} samples  (DeltaT < 0)")
print(f"  Joint stratum         : {n_joint:,} samples  (both)")
print(f"  Warm contrail stratum : {n_warm_contrail:,} samples  "
      f"(T > 224K, RHi > 110%, DeltaT < 0)")
print(f"  Random background     : {n_random:,} samples")

mask_issr          = rhi > 100.0
mask_formation     = delta_t < 0.0
mask_joint         = mask_issr & mask_formation
mask_warm_contrail = (
    (temperature > 224.0) &
    (rhi > 110.0) &
    (delta_t < 0.0)
)

print(f"\n  Valid ISSR points          : {mask_issr.sum():,}  "
      f"({100*mask_issr.mean():.1f}%)")
print(f"  Valid formation points     : {mask_formation.sum():,}  "
      f"({100*mask_formation.mean():.1f}%)")
print(f"  Valid joint points         : {mask_joint.sum():,}  "
      f"({100*mask_joint.mean():.1f}%)")
print(f"  Valid warm contrail points : {mask_warm_contrail.sum():,}  "
      f"({100*mask_warm_contrail.mean():.1f}%)")

idx_issr          = _sample_indices(mask_issr,          n_issr,          rng)
idx_formation     = _sample_indices(mask_formation,     n_formation,     rng)
idx_joint         = _sample_indices(mask_joint,         n_joint,         rng)
idx_warm_contrail = _sample_indices(mask_warm_contrail, n_warm_contrail, rng)
idx_random        = _random_indices(n_random, temperature.shape, rng)

del mask_issr, mask_formation, mask_joint, mask_warm_contrail
gc.collect()

all_idx = np.concatenate(
    [idx_issr, idx_formation, idx_joint, idx_warm_contrail, idx_random],
    axis=0
)

idx_struct = np.ascontiguousarray(all_idx).view(
    np.dtype((np.void, all_idx.dtype.itemsize * all_idx.shape[1])))
_, unique_pos = np.unique(idx_struct, return_index=True)
all_idx = all_idx[unique_pos]

shortfall = N_SAMPLES - len(all_idx)
if shortfall > 0:
    print(f"  Topping up {shortfall:,} rows lost to deduplication...")
    extra = _random_indices(shortfall * 2, temperature.shape, rng)
    all_idx = np.concatenate([all_idx, extra], axis=0)[:N_SAMPLES]

rng.shuffle(all_idx)

ti, pi, lati, loni = all_idx[:, 0], all_idx[:, 1], all_idx[:, 2], all_idx[:, 3]

print(f"\n  Final sample count: {len(ti):,}")


# ============================================================
# STEP 4: EXTRACT VALUES AT SAMPLED INDICES
# ============================================================

print("\n" + "=" * 55)
print("EXTRACTING FEATURE VALUES AT SAMPLED INDICES")
print("=" * 55)

pressure_sampled = pressure[pi]
issr_sampled     = issr_depth[ti, lati, loni]

df = pd.DataFrame({
    "temperature":      temperature [ti, pi, lati, loni],
    "pressure_level":   pressure_sampled,
    "rhi":              rhi         [ti, pi, lati, loni],
    "delta_t":          delta_t     [ti, pi, lati, loni],
    "wind_speed":       wind_speed  [ti, pi, lati, loni],
    "wind_shear":       wind_shear  [ti, pi, lati, loni],
    "static_stability": stability   [ti, pi, lati, loni],
    "issr_depth":       issr_sampled,
    "latitude":         latitudes[lati],
    "longitude":        longitudes[loni],
    "t_index":          ti,
    "p_index":          pi,
})

print(f"DataFrame shape : {df.shape}")
print(f"DataFrame RAM   : {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")


# ============================================================
# STEP 5: SANITY CHECKS
# ============================================================

print("\n" + "=" * 55)
print("SANITY CHECKS")
print("=" * 55)

print("\nFeature statistics:")
print(df[["temperature", "rhi", "delta_t", "wind_speed",
          "wind_shear", "static_stability", "issr_depth"]].describe().round(3))

print(f"\nRHi > 100% rows  : {(df['rhi'] > 100).sum():,}  "
      f"({100*(df['rhi']>100).mean():.1f}%)")
print(f"DeltaT < 0 rows  : {(df['delta_t'] < 0).sum():,}  "
      f"({100*(df['delta_t']<0).mean():.1f}%)")
print(f"Both conditions  : {((df['rhi']>100)&(df['delta_t']<0)).sum():,}  "
      f"({100*((df['rhi']>100)&(df['delta_t']<0)).mean():.1f}%)")

warm_mask = (df['temperature'] > 224) & (df['rhi'] > 110) & (df['delta_t'] < 0)
print(f"Warm contrail rows (T>224, RHi>110, dT<0): "
      f"{warm_mask.sum():,}  ({100*warm_mask.mean():.1f}%)")

nan_counts = df.isnull().sum()
if nan_counts.any():
    print(f"\nWARNING: NaN values found:\n{nan_counts[nan_counts > 0]}")
else:
    print("\nNo NaN values found. ✓")


# ============================================================
# STEP 6: SAVE
# ============================================================

print("\n" + "=" * 55)
print("SAVING DATASET")
print("=" * 55)

df.to_parquet(OUTPUT_FILE, index=False, compression="snappy")
print(f"Saved to: {OUTPUT_FILE}")

import os
size_mb = os.path.getsize(OUTPUT_FILE) / 1e6
print(f"File size: {size_mb:.1f} MB")

print("\nDone. Feed this file into label_generation.py next.")
print("=" * 55)