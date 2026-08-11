"""
feature_engineering.py
=======================
Memory-efficient feature extraction from ERA5 data.

Key design:
  - NEVER flattens full arrays into 1D
  - Samples multi-dimensional indices directly
  - Uses stratified sampling to preserve rare contrail conditions
  - Saves to Parquet (not CSV) for ~5x smaller files and faster I/O

Memory budget (M2 Mac, 16GB):
  - Peak usage here: ~1.5–2 GB (vs ~5+ GB with flatten approach)
  - Safe for 500K–1M samples across 8 feature arrays

Output:
  feature_dataset.parquet
  → feed directly into label_generation.py
"""

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
N_SAMPLES   = 600_000    # total rows in final dataset
RANDOM_SEED = 42

# Stratification: what fraction of samples to GUARANTEE
# come from physically interesting regions.
# The remainder (1 - sum of fractions) is drawn randomly.
STRAT_CONFIG = {
    # Fraction reserved for ice-supersaturated points (RHi > 100%)
    # These are rare but critical — contrails only persist here.
    "issr_fraction":      0.30,   # 30% of samples from RHi > 100 regions

    # Fraction reserved for SAC-favourable formation conditions
    # (DeltaT < 0, i.e. temperature below T_critical)
    "formation_fraction": 0.25,   # 25% from DeltaT < 0 regions

    # Fraction reserved for the joint "high probability" region
    # (both formation AND persistence conditions met)
    "joint_fraction":     0.15,   # 15% from DeltaT<0 AND RHi>100
}
# Remaining 30% = pure random sample for background distribution


# ============================================================
# STEP 1: LOAD DATA
# ============================================================

print("=" * 55)
print("LOADING ERA5 DATASET")
print("=" * 55)

# open_dataset with chunks=None loads eagerly but uses xarray's
# optimised memory layout. For this use-case this is fine since
# we need random access across all dimensions anyway.
ds = xr.open_dataset(ERA5_FILE)

print(f"Dataset dimensions: {dict(ds.dims)}")

# Extract as numpy arrays (float32 to halve memory vs float64)
temperature  = ds["t"].values.astype(np.float32)
humidity     = ds["r"].values.astype(np.float32)
u            = ds["u"].values.astype(np.float32)
v            = ds["v"].values.astype(np.float32)
omega        = ds["w"].values.astype(np.float32)
geopotential = ds["z"].values.astype(np.float32)

pressure     = ds["pressure_level"].values.astype(np.float32)

# Coordinate arrays (for optional output columns)
latitudes    = ds["latitude"].values.astype(np.float32)
longitudes   = ds["longitude"].values.astype(np.float32)

ds.close()  # free the xarray object; we have numpy arrays now

t_dim, p_dim, lat_dim, lon_dim = temperature.shape
total_points = t_dim * p_dim * lat_dim * lon_dim

print(f"\nArray shape : {temperature.shape}")
print(f"Total points: {total_points:,}  (~{total_points/1e6:.1f}M)")
print(f"Per-array RAM (float32): {temperature.nbytes / 1e9:.2f} GB")


# ============================================================
# STEP 2: COMPUTE PHYSICS FEATURES (IN-PLACE WHERE POSSIBLE)
# ============================================================

print("\n" + "=" * 55)
print("COMPUTING PHYSICS FEATURES")
print("=" * 55)

rhi       = calculate_rhi(temperature, humidity)
tcritical = calculate_tcritical(pressure)          # shape (p_dim,)
delta_t   = calculate_delta_t(temperature, tcritical)
wind_speed = calculate_wind_speed(u, v)
wind_shear = calculate_wind_shear(u, v, pressure)
stability  = calculate_static_stability(temperature, geopotential, pressure)
issr_depth = calculate_issr_depth(rhi)             # shape (t, lat, lon)

# Free raw variables we no longer need to reduce RAM
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
#
# We sample (t_idx, p_idx, lat_idx, lon_idx) tuples directly.
# No array is ever flattened.
# ============================================================

print("\n" + "=" * 55)
print("STRATIFIED INDEX SAMPLING")
print("=" * 55)

rng = np.random.default_rng(RANDOM_SEED)

def _sample_indices(mask_4d, n, rng):
    """
    Given a boolean 4D mask, return n random (t, p, lat, lon)
    index tuples drawn only from positions where mask is True.
    If fewer than n valid positions exist, returns all of them.
    """
    valid = np.argwhere(mask_4d)   # shape (N_valid, 4)
    if len(valid) == 0:
        return np.empty((0, 4), dtype=np.intp)
    n = min(n, len(valid))
    chosen = rng.choice(len(valid), size=n, replace=False)
    return valid[chosen]           # shape (n, 4)


def _random_indices(n, shape, rng):
    """
    Return n random (t, p, lat, lon) index tuples from
    the full 4D space, with no mask constraint.
    """
    t   = rng.integers(0, shape[0], size=n)
    p   = rng.integers(0, shape[1], size=n)
    lat = rng.integers(0, shape[2], size=n)
    lon = rng.integers(0, shape[3], size=n)
    return np.column_stack([t, p, lat, lon])


# Expand issr_depth to 4D for masking (no data copy — broadcast view)
# We need it 4D only for building masks; we don't permanently expand it.
issr_4d_view = np.broadcast_to(
    issr_depth[:, np.newaxis, :, :],
    temperature.shape)

n_issr      = int(N_SAMPLES * STRAT_CONFIG["issr_fraction"])
n_formation = int(N_SAMPLES * STRAT_CONFIG["formation_fraction"])
n_joint     = int(N_SAMPLES * STRAT_CONFIG["joint_fraction"])
n_random    = N_SAMPLES - n_issr - n_formation - n_joint

print(f"  ISSR stratum      : {n_issr:,} samples  (RHi > 100%)")
print(f"  Formation stratum : {n_formation:,} samples  (DeltaT < 0)")
print(f"  Joint stratum     : {n_joint:,} samples  (both)")
print(f"  Random background : {n_random:,} samples")

# Build masks (boolean arrays, 1 byte each → 142M bytes = 142 MB each, acceptable)
mask_issr      = rhi > 100.0
mask_formation = delta_t < 0.0
mask_joint     = mask_issr & mask_formation

print(f"\n  Valid ISSR points      : {mask_issr.sum():,}  "
      f"({100*mask_issr.mean():.1f}% of total)")
print(f"  Valid formation points : {mask_formation.sum():,}  "
      f"({100*mask_formation.mean():.1f}% of total)")
print(f"  Valid joint points     : {mask_joint.sum():,}  "
      f"({100*mask_joint.mean():.1f}% of total)")

idx_issr      = _sample_indices(mask_issr,      n_issr,      rng)
idx_formation = _sample_indices(mask_formation, n_formation, rng)
idx_joint     = _sample_indices(mask_joint,     n_joint,     rng)
idx_random    = _random_indices(n_random, temperature.shape,  rng)

# Free masks
del mask_issr, mask_formation, mask_joint
gc.collect()

# Concatenate all index sets and deduplicate
all_idx = np.concatenate(
    [idx_issr, idx_formation, idx_joint, idx_random], axis=0)

# Deduplicate: convert to structured array for fast unique
idx_struct = np.ascontiguousarray(all_idx).view(
    np.dtype((np.void, all_idx.dtype.itemsize * all_idx.shape[1])))
_, unique_pos = np.unique(idx_struct, return_index=True)
all_idx = all_idx[unique_pos]

# If deduplication reduced count, top up with random samples
shortfall = N_SAMPLES - len(all_idx)
if shortfall > 0:
    extra = _random_indices(shortfall * 2, temperature.shape, rng)
    all_idx = np.concatenate([all_idx, extra], axis=0)[:N_SAMPLES]

# Shuffle so strata are not block-ordered (important for mini-batch training)
rng.shuffle(all_idx)

ti, pi, lati, loni = all_idx[:, 0], all_idx[:, 1], all_idx[:, 2], all_idx[:, 3]

print(f"\n  Final sample count: {len(ti):,}")


# ============================================================
# STEP 4: EXTRACT VALUES AT SAMPLED INDICES
#
# Direct fancy indexing — touches only N_SAMPLES elements.
# Each line below uses ~2.4 MB of RAM (500K × float32).
# Total peak here: ~25 MB. Compare with flatten: ~570 MB each.
# ============================================================

print("\n" + "=" * 55)
print("EXTRACTING FEATURE VALUES AT SAMPLED INDICES")
print("=" * 55)

# pressure_level: 1D array, indexed by pi only
pressure_sampled = pressure[pi]

# issr_depth: 3D (t, lat, lon), indexed by (ti, lati, loni)
issr_sampled = issr_depth[ti, lati, loni]

# All 4D arrays: indexed by (ti, pi, lati, loni)
df = pd.DataFrame({
    "temperature":    temperature [ti, pi, lati, loni],
    "pressure_level": pressure_sampled,
    "rhi":            rhi         [ti, pi, lati, loni],
    "delta_t":        delta_t     [ti, pi, lati, loni],
    "wind_speed":     wind_speed  [ti, pi, lati, loni],
    "wind_shear":     wind_shear  [ti, pi, lati, loni],
    "static_stability": stability [ti, pi, lati, loni],
    "issr_depth":     issr_sampled,
    # Optional coordinate columns (useful for spatial diagnostics)
    "latitude":       latitudes[lati],
    "longitude":      longitudes[loni],
    "t_index":        ti,
    "p_index":        pi,
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

print(f"\nRHi > 100% rows : {(df['rhi'] > 100).sum():,}  "
      f"({100*(df['rhi']>100).mean():.1f}%)")
print(f"DeltaT < 0 rows : {(df['delta_t'] < 0).sum():,}  "
      f"({100*(df['delta_t']<0).mean():.1f}%)")
print(f"Both conditions : {((df['rhi']>100)&(df['delta_t']<0)).sum():,}  "
      f"({100*((df['rhi']>100)&(df['delta_t']<0)).mean():.1f}%)")

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