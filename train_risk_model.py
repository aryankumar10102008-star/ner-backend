"""
NER Smart Logistics — Route Risk Prediction
Training script

Generates a realistic synthetic dataset (swap this for real historical
data later) and trains a RandomForest classifier to predict the
probability that a route segment gets disrupted.

Run:
    pip install scikit-learn pandas numpy joblib
    python train_risk_model.py
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import joblib

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

FEATURE_COLUMNS = [
    "rainfall_last_24h_mm",
    "rainfall_forecast_6h_mm",
    "wind_speed_kmh",
    "elevation_m",
    "slope_deg",
    "landslide_prone_zone",   # 0/1
    "past_closures_90d",      # count of closures in last 90 days on this segment
    "road_type_highway",      # 1 = highway, 0 = rural/district road
    "is_monsoon_month",       # 0/1 (Jun-Sep for NER)
]

TARGET_COLUMN = "disrupted"


def generate_synthetic_dataset(n_samples: int = 4000) -> pd.DataFrame:
    """
    Creates a labeled dataset that mimics realistic NER conditions.
    Replace this with real historical route/weather/closure data once
    you have it — the training code below doesn't need to change.
    """
    rainfall_24h = np.random.gamma(shape=2.0, scale=15, size=n_samples)          # mm
    rainfall_fc_6h = np.random.gamma(shape=1.5, scale=10, size=n_samples)        # mm
    wind_speed = np.random.normal(20, 8, n_samples).clip(0)                     # km/h
    elevation = np.random.uniform(50, 2200, n_samples)                          # m
    slope = np.random.uniform(0, 45, n_samples)                                 # degrees
    landslide_zone = np.random.binomial(1, 0.25, n_samples)
    past_closures = np.random.poisson(1.2, n_samples)
    road_type_highway = np.random.binomial(1, 0.4, n_samples)
    is_monsoon = np.random.binomial(1, 0.35, n_samples)

    # ground-truth risk signal used to generate labels (kept nonlinear/interaction-heavy
    # so the model actually has something meaningful to learn)
    risk_signal = (
        0.035 * rainfall_24h
        + 0.045 * rainfall_fc_6h
        + 0.02 * wind_speed
        + 0.06 * slope
        + 1.8 * landslide_zone
        + 0.5 * past_closures
        + 1.4 * is_monsoon
        - 0.6 * road_type_highway          # highways are more resilient
        + 0.0009 * elevation
        + np.random.normal(0, 1.0, n_samples)   # noise
    )

    # convert to probability then sample a binary label
    prob_disrupted = 1 / (1 + np.exp(-(risk_signal - 6)))
    disrupted = np.random.binomial(1, prob_disrupted)

    df = pd.DataFrame({
        "rainfall_last_24h_mm": rainfall_24h,
        "rainfall_forecast_6h_mm": rainfall_fc_6h,
        "wind_speed_kmh": wind_speed,
        "elevation_m": elevation,
        "slope_deg": slope,
        "landslide_prone_zone": landslide_zone,
        "past_closures_90d": past_closures,
        "road_type_highway": road_type_highway,
        "is_monsoon_month": is_monsoon,
        "disrupted": disrupted,
    })
    return df


def train_and_save_model(output_path: str = "risk_model.pkl"):
    df = generate_synthetic_dataset()

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # Evaluation
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    print("=== Evaluation on held-out test set ===")
    print(classification_report(y_test, y_pred, target_names=["No disruption", "Disruption"]))
    print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.3f}")

    print("\n=== Feature importance (what drives the risk score) ===")
    importances = pd.Series(model.feature_importances_, index=FEATURE_COLUMNS)
    print(importances.sort_values(ascending=False).round(3))

    joblib.dump({"model": model, "features": FEATURE_COLUMNS}, output_path)
    print(f"\nModel saved to {output_path}")


if __name__ == "__main__":
    train_and_save_model()
