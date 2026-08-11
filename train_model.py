import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score
)

from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

print("Loading dataset...")

df = pd.read_parquet(
    "labeled_dataset.parquet"
)

FEATURES = [
    "temperature",
    "pressure_level",
    "wind_speed",
    "wind_shear",
    "static_stability"
]

TARGET = "label"

X = df[FEATURES]
y = df[TARGET]

print("Splitting data...")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# ==========================
# Logistic Regression
# ==========================

print("\nTraining Logistic Regression...")

scaler = StandardScaler()

X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

lr = LogisticRegression(
    max_iter=1000
)

lr.fit(X_train_sc, y_train)

pred_lr = lr.predict(X_test_sc)
prob_lr = lr.predict_proba(X_test_sc)[:,1]

print("\n===== LOGISTIC REGRESSION =====")

print(classification_report(
    y_test,
    pred_lr
))

print(
    "ROC AUC:",
    roc_auc_score(
        y_test,
        prob_lr
    )
)

# ==========================
# XGBoost
# ==========================

print("\nTraining XGBoost...")

xgb = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric="logloss"
)

xgb.fit(
    X_train,
    y_train
)

pred_xgb = xgb.predict(X_test)
prob_xgb = xgb.predict_proba(X_test)[:,1]

print("\n===== XGBOOST =====")

print(classification_report(
    y_test,
    pred_xgb
))

print(
    "ROC AUC:",
    roc_auc_score(
        y_test,
        prob_xgb
    )
)

print("\nFeature Importance:")

for f, imp in sorted(
    zip(FEATURES, xgb.feature_importances_),
    key=lambda x: x[1],
    reverse=True
):
    print(f"{f:20s} {imp:.4f}")