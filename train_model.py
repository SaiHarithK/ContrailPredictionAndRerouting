"""
train_model.py  —  Revised
===========================
Key changes from original:
  1. Features are DISJOINT from label generation inputs.
     Labels were built from delta_t and rhi (thermodynamic SAC).
     Model features are synoptic/dynamical state variables only.
  2. Temporal train/test split (not random) to prevent
     autocorrelation leakage between adjacent timesteps.
  3. Added latitude, month, omega, tropopause_distance features.
  4. XGBRegressor on p_contrail (continuous) in addition to
     XGBClassifier on binary label.
  5. scale_pos_weight set correctly for class imbalance.
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    brier_score_loss,
)
from xgboost import XGBClassifier, XGBRegressor

# ============================================================
# CONFIG
# ============================================================

DATA_FILE = "labeled_dataset.parquet"

# Features the MODEL sees — none of these are inputs to
# the label generation function (delta_t, rhi, issr_depth).
# These are synoptic/dynamical atmospheric state variables.
FEATURES = [
    "temperature",        # raw T from ERA5 — not delta_t
    "pressure_level",     # altitude proxy
    "wind_speed",         # horizontal wind magnitude
    "wind_shear",         # vertical wind shear
    "static_stability",   # Brunt-Väisälä related
    "omega",              # vertical velocity (ERA5 'w') — new
    "latitude",           # geographic signal
    "sin_month",          # seasonal signal (sine component)
    "cos_month",          # seasonal signal (cosine component)
    "tropopause_dist",    # distance below tropopause in hPa — new
]

# Regression target: continuous probability (richer signal)
PROB_TARGET  = "p_contrail"

# Classification target: binary label
CLASS_TARGET = "label"

# ============================================================
# LOAD AND PREPARE DATA
# ============================================================

print("Loading dataset...")
df = pd.read_parquet(DATA_FILE)
print(f"Shape: {df.shape}")

# --- Derived temporal features ---
# Assumes 't_index' maps to a datetime; adjust if your dataset
# stores timestamps differently.
if "timestamp" in df.columns:
    df["month"] = pd.to_datetime(df["timestamp"]).dt.month
elif "t_index" in df.columns:
    # t_index 0-1459 over 2023 (4 per day) → day of year
    df["month"] = ((df["t_index"] // 4) // 30).clip(0, 11) + 1

df["sin_month"] = np.sin(2 * np.pi * df["month"] / 12)
df["cos_month"] = np.cos(2 * np.pi * df["month"] / 12)

# --- Tropopause distance ---
# Approximate WMO thermal tropopause pressure by latitude.
# At viva: "We use a latitude-dependent climatological tropopause
# pressure as a first approximation; a better version would
# compute it from the ERA5 temperature lapse rate profile."
def approx_tropopause_pressure(lat):
    """
    Returns approximate tropopause pressure (hPa) as a function
    of latitude. Tropics ~100 hPa, poles ~300 hPa.
    Linear interpolation between observed climatological values.
    """
    abs_lat = np.abs(lat)
    # Simple piecewise: equator=100 hPa, 60°=200 hPa, 90°=300 hPa
    trop_p = np.where(
        abs_lat <= 30,
        100 + (abs_lat / 30) * 50,           # 100→150 hPa
        np.where(
            abs_lat <= 60,
            150 + ((abs_lat - 30) / 30) * 100, # 150→250 hPa
            250 + ((abs_lat - 60) / 30) * 50   # 250→300 hPa
        )
    )
    return trop_p

df["tropopause_p"] = approx_tropopause_pressure(df["latitude"].values)
df["tropopause_dist"] = df["pressure_level"] - df["tropopause_p"]
# Positive = below tropopause (troposphere), negative = above

# --- Check omega column exists ---
if "omega" not in df.columns:
    print("WARNING: 'omega' column not found. Setting to 0.")
    print("  Re-run feature_engineering.py with omega included.")
    df["omega"] = 0.0

# --- Verify all required features exist ---
missing = [f for f in FEATURES if f not in df.columns]
if missing:
    raise ValueError(f"Missing features: {missing}")

X = df[FEATURES]
y_class = df[CLASS_TARGET]
y_prob  = df[PROB_TARGET]

# ============================================================
# TEMPORAL TRAIN / TEST SPLIT
# ============================================================
# Sort by time index so train = earlier period, test = later.
# This prevents autocorrelation leakage between adjacent
# timesteps that random splitting would allow.

print("\nApplying temporal train/test split...")

if "t_index" in df.columns:
    sort_col = "t_index"
elif "timestamp" in df.columns:
    sort_col = "timestamp"
else:
    raise ValueError("No time column found for temporal split.")

df_sorted = df.sort_values(sort_col).reset_index(drop=True)
X_sorted = df_sorted[FEATURES]
y_class_sorted = df_sorted[CLASS_TARGET]
y_prob_sorted  = df_sorted[PROB_TARGET]

split = int(0.8 * len(df_sorted))
X_train, X_test = X_sorted.iloc[:split], X_sorted.iloc[split:]
y_train_c, y_test_c = y_class_sorted.iloc[:split], y_class_sorted.iloc[split:]
y_train_p, y_test_p = y_prob_sorted.iloc[:split],  y_prob_sorted.iloc[split:]

print(f"  Train: {len(X_train):,} rows  "
      f"(label=1: {y_train_c.mean()*100:.1f}%)")
print(f"  Test:  {len(X_test):,} rows  "
      f"(label=1: {y_test_c.mean()*100:.1f}%)")

# ============================================================
# LOGISTIC REGRESSION (BASELINE)
# ============================================================

print("\nTraining Logistic Regression (baseline)...")

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

lr = LogisticRegression(max_iter=1000, class_weight="balanced")
lr.fit(X_train_sc, y_train_c)

pred_lr = lr.predict(X_test_sc)
prob_lr = lr.predict_proba(X_test_sc)[:, 1]

print("\n===== LOGISTIC REGRESSION =====")
print(classification_report(y_test_c, pred_lr))
print(f"ROC AUC : {roc_auc_score(y_test_c, prob_lr):.4f}")
print(f"Brier   : {brier_score_loss(y_test_c, prob_lr):.4f}")

print("\nFeature coefficients (|coef| → importance):")
for f, c in sorted(
    zip(FEATURES, lr.coef_[0]),
    key=lambda x: abs(x[1]), reverse=True
):
    print(f"  {f:25s}  {c:+.4f}")

# ============================================================
# XGBOOST CLASSIFIER
# ============================================================

print("\nTraining XGBoost Classifier...")

pos_weight = (y_train_c == 0).sum() / (y_train_c == 1).sum()
print(f"  scale_pos_weight = {pos_weight:.2f}")

xgb_clf = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=pos_weight,
    random_state=42,
    eval_metric="logloss",
    early_stopping_rounds=20,
)

xgb_clf.fit(
    X_train, y_train_c,
    eval_set=[(X_test, y_test_c)],
    verbose=50,
)

pred_xgb = xgb_clf.predict(X_test)
prob_xgb = xgb_clf.predict_proba(X_test)[:, 1]

print("\n===== XGBOOST CLASSIFIER =====")
print(classification_report(y_test_c, pred_xgb))
print(f"ROC AUC : {roc_auc_score(y_test_c, prob_xgb):.4f}")
print(f"Brier   : {brier_score_loss(y_test_c, prob_xgb):.4f}")

print("\nFeature Importances (gain):")
importances = xgb_clf.get_booster().get_score(importance_type="gain")
for f in FEATURES:
    imp = importances.get(f, 0.0)
    print(f"  {f:25s}  {imp:.2f}")

# ============================================================
# XGBOOST REGRESSOR (on continuous p_contrail)
# ============================================================
# Training on the continuous probability target provides a
# richer gradient signal than binary cross-entropy.
# At inference, the output IS the contrail probability directly.

print("\nTraining XGBoost Regressor (on p_contrail)...")

xgb_reg = XGBRegressor(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="reg:logistic",   # forces output to [0,1]
    random_state=42,
    eval_metric="rmse",
    early_stopping_rounds=20,
)

xgb_reg.fit(
    X_train, y_train_p,
    eval_set=[(X_test, y_test_p)],
    verbose=50,
)

pred_reg = xgb_reg.predict(X_test).clip(0, 1)
# Evaluate regression output as classifier at threshold 0.5
pred_reg_binary = (pred_reg > 0.5).astype(int)

print("\n===== XGBOOST REGRESSOR =====")
print(classification_report(y_test_c, pred_reg_binary))
print(f"ROC AUC (reg→class) : {roc_auc_score(y_test_c, pred_reg):.4f}")
print(f"Brier Score         : {brier_score_loss(y_test_c, pred_reg):.4f}")

# ============================================================
# SAVE ARTEFACTS
# ============================================================

joblib.dump(xgb_clf, "contrail_model_clf.pkl")
joblib.dump(xgb_reg, "contrail_model_reg.pkl")
joblib.dump(scaler,  "feature_scaler.pkl")

print("\nSaved:")
print("  contrail_model_clf.pkl  — XGBoost classifier")
print("  contrail_model_reg.pkl  — XGBoost regressor (recommended for route)")
print("  feature_scaler.pkl      — StandardScaler for Logistic Regression")
print(f"\nModel features: {FEATURES}")