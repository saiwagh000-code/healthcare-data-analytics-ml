"""
Healthcare Enrollment Resolution Time Prediction
Beginner-friendly training script.
Target: Resolution Time (days), using only features available at submission.
Note: this is a learning project, not a production healthcare decision system.
"""
from pathlib import Path
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

BASE = Path(__file__).resolve().parent
DATA = BASE / "healthcare_enrollment_cleaned.csv"
MODEL_PATH = BASE / "resolution_time_model.joblib"
METRICS_PATH = BASE / "model_metrics.json"
PREDICTIONS_PATH = BASE / "test_set_predictions.csv"

df = pd.read_csv(DATA, parse_dates=["Date"])
df["MonthNumber"] = df["Date"].dt.month
df["DayOfWeek"] = df["Date"].dt.dayofweek

# A missing resolution time is not a zero. Exclude those rows from model training.
model_df = df.dropna(subset=["Resolution Time (days)"]).copy()
target = "Resolution Time (days)"
features = ["Enrollment Type", "Plan Type", "Submitted By", "MonthNumber", "DayOfWeek"]
X = model_df[features]
y = model_df[target]

categorical = ["Enrollment Type", "Plan Type", "Submitted By"]
numeric = ["MonthNumber", "DayOfWeek"]
preprocess = ColumnTransformer([
    ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical),
    ("numeric", SimpleImputer(strategy="median"), numeric),
])
pipeline = Pipeline([
    ("preprocess", preprocess),
    ("model", RandomForestRegressor(n_estimators=200, random_state=42, min_samples_leaf=3)),
])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)
pipeline.fit(X_train, y_train)
pred = pipeline.predict(X_test)

metrics = {
    "target": target,
    "training_rows": int(len(X_train)),
    "test_rows": int(len(X_test)),
    "MAE_days": round(float(mean_absolute_error(y_test, pred)), 3),
    "RMSE_days": round(float(mean_squared_error(y_test, pred) ** 0.5), 3),
    "R2": round(float(r2_score(y_test, pred)), 3),
    "note": "Evaluate these metrics before interpreting predictions; performance is dataset-specific."
}
joblib.dump(pipeline, MODEL_PATH)
with open(METRICS_PATH, "w", encoding="utf-8") as f:
    json.dump(metrics, f, indent=2)
results = X_test.copy()
results["Actual Resolution Time (days)"] = y_test.values
results["Predicted Resolution Time (days)"] = pred.round(2)
results.to_csv(PREDICTIONS_PATH, index=False)

print("Model training completed.")
print(json.dumps(metrics, indent=2))
print(f"Saved model: {MODEL_PATH}")
print(f"Saved test predictions: {PREDICTIONS_PATH}")
