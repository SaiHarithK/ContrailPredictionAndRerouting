import joblib
import pandas as pd
import numpy as np

model = joblib.load("contrail_model_reg.pkl")
df = pd.read_parquet("labeled_dataset.parquet")

# Use realistic feature values, not arbitrary constants.
# Fix pressure_level at 250 hPa and pull the real mean wind_shear/
# static_stability for rows near that pressure level.
near_250 = df[(df["pressure_level"] > 235) & (df["pressure_level"] < 265)]
real_wind_shear = near_250["wind_shear"].mean()
real_static_stability = near_250["static_stability"].mean()
real_tropopause_dist = 250.0 - 175.0  # or compute properly for your test latitude

print(f"Using wind_shear={real_wind_shear:.5f}, "
      f"static_stability={real_static_stability:.6f}")

results = []
for T in np.arange(215, 242, 1.0):
    X = pd.DataFrame([{
        "temperature":      T,
        "pressure_level":   250.0,
        "wind_speed":       20.0,
        "wind_shear":       real_wind_shear,
        "static_stability": real_static_stability,
        "omega":            0.0,     # matches training — omega was always 0
        "latitude":         44.0,
        "sin_month":        np.sin(2 * np.pi * 8 / 12),
        "cos_month":        np.cos(2 * np.pi * 8 / 12),
        "tropopause_dist":  real_tropopause_dist,
    }])
    p = float(model.predict(X)[0])
    results.append((T, p))

print(f"{'Temp (K)':<12} {'P_contrail':<12} {'Prediction'}")
print("-" * 35)
for T, p in results:
    flag = "← CONTRAIL" if p > 0.5 else ""
    print(f"{T:<12.1f} {p:<12.4f} {flag}")