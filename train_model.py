"""
train_model.py
--------------
Rebuilds the AQI model with the rigor a evaluator will actually probe:

  1. Model comparison   -> Linear Regression vs Random Forest vs HistGradientBoosting,
                           not just "I picked Random Forest".
  2. Cross-validation    -> 5-fold CV metrics (R², MAE, RMSE), not a single lucky split.
  3. Leakage check        -> flags if the target looks like it was computed FROM the
                           features (common with AQI datasets), so you can address it
                           head-on in your report instead of getting caught off guard.
  4. Temporal split       -> if a date/time column exists, splits chronologically
                           instead of randomly shuffling — avoids the "future leaking
                           into training" problem examiners look for.
  5. Prediction intervals -> uses per-tree variance in the Random Forest to report an
                           uncertainty band, not just a bare point prediction.

Usage:
    python train_model.py --data data/your_dataset.csv --target AQI

If you don't pass --target, it tries to auto-detect a column that looks like AQI.
If your data has no date/timestamp column, it falls back to a random split and
prints an explicit warning explaining the tradeoff (put that warning in your report).

Output:
    model/aqi_model.pkl        -> drop-in replacement, same format your app.py expects
    model_comparison.json      -> full metrics table, for your report/viva prep
    PROJECT_JUSTIFICATION.md   -> auto-generated paragraph-level writeup of the above
"""

import argparse
import json
import os
import pickle
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, TimeSeriesSplit, cross_validate

FEATURE_ORDER = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]
TARGET_CANDIDATES = ["AQI", "aqi", "AQI_Value", "aqi_value", "Aqi"]
TIME_COL_HINTS = ["date", "time", "timestamp", "datetime"]


def find_target_column(df, explicit=None):
    if explicit:
        if explicit not in df.columns:
            sys.exit(f"--target '{explicit}' not found in columns: {list(df.columns)}")
        return explicit
    for c in TARGET_CANDIDATES:
        if c in df.columns:
            return c
    sys.exit(
        "Could not auto-detect the AQI target column. "
        f"Pass it explicitly: --target <column_name>. Available columns: {list(df.columns)}"
    )


def find_time_column(df):
    for c in df.columns:
        if any(hint in c.lower() for hint in TIME_COL_HINTS):
            return c
    return None


def check_leakage(df, features, target):
    """
    Many public AQI datasets compute the AQI column directly FROM the pollutant
    readings via a fixed sub-index formula. If that's true here, a high R² doesn't
    mean the model learned something hard — it means it recovered a known formula.
    A simple linear fit catches this: if plain Linear Regression already gets a
    near-perfect R², the relationship is likely deterministic, not statistical.
    """
    lr = LinearRegression()
    lr.fit(df[features], df[target])
    r2 = r2_score(df[target], lr.predict(df[features]))
    suspicious = r2 > 0.97
    return r2, suspicious


def build_models():
    return {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
        n_estimators=150, max_depth=14, min_samples_leaf=4, random_state=42, n_jobs=-1
          ),
        "Hist Gradient Boosting": HistGradientBoostingRegressor(random_state=42),
    }


def run_cv(models, X, y, cv):
    scoring = {
        "r2": "r2",
        "mae": "neg_mean_absolute_error",
        "rmse": "neg_root_mean_squared_error",
    }
    results = {}
    for name, model in models.items():
        scores = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=-1)
        results[name] = {
            "r2_mean": float(np.mean(scores["test_r2"])),
            "r2_std": float(np.std(scores["test_r2"])),
            "mae_mean": float(-np.mean(scores["test_mae"])),
            "rmse_mean": float(-np.mean(scores["test_rmse"])),
        }
    return results


def prediction_interval(rf_model, X_row, percentile=90):
    """Uses the spread across individual trees in the forest as an uncertainty band."""
    tree_preds = np.array([t.predict(X_row)[0] for t in rf_model.estimators_])
    lower = np.percentile(tree_preds, (100 - percentile) / 2)
    upper = np.percentile(tree_preds, 100 - (100 - percentile) / 2)
    return float(lower), float(upper)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True, help="Path to your training CSV")
    parser.add_argument("--target", default=None, help="Name of the AQI target column")
    parser.add_argument("--model-out", default="model/aqi_model.pkl")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    df.columns = [c.strip() for c in df.columns]

    missing_features = [f for f in FEATURE_ORDER if f not in df.columns]
    if missing_features:
        sys.exit(
            f"Dataset is missing expected feature columns: {missing_features}. "
            f"Found columns: {list(df.columns)}. "
            "Rename your columns to match, or edit FEATURE_ORDER at the top of this script."
        )

    target = find_target_column(df, args.target)
    time_col = find_time_column(df)

    df = df.dropna(subset=FEATURE_ORDER + [target]).reset_index(drop=True)

    print(f"\nRows after dropping missing values: {len(df)}")
    print(f"Target column: {target}")
    print(f"Time column detected: {time_col if time_col else 'NONE — see warning below'}\n")

    # ---- 3. Leakage check ----
    r2_linear, suspicious = check_leakage(df, FEATURE_ORDER, target)
    print(f"[Leakage check] Plain Linear Regression R² on full data: {r2_linear:.4f}")
    if suspicious:
        print(
            "  WARNING: This is very high for a linear fit. It suggests the AQI "
            "target may be computed directly from these pollutant readings via a "
            "known sub-index formula, rather than being an independently measured "
            "quantity. Say this explicitly in your report — do not present the R² "
            "as evidence of a hard prediction problem without this caveat.\n"
        )
    else:
        print("  Looks like a genuinely learned relationship, not a restated formula.\n")

    # ---- 4. Temporal split (or explicit fallback) ----
    if time_col:
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
        df = df.dropna(subset=[time_col]).sort_values(time_col).reset_index(drop=True)
        split_idx = int(len(df) * 0.8)
        train_df, test_df = df.iloc[:split_idx], df.iloc[split_idx:]
        cv = TimeSeriesSplit(n_splits=5)
        split_note = (
            f"Chronological split on '{time_col}': trained on the earliest 80% "
            f"({train_df[time_col].min()} to {train_df[time_col].max()}), tested on "
            f"the most recent 20% ({test_df[time_col].min()} to {test_df[time_col].max()})."
        )
    else:
        warnings.warn(
            "No date/timestamp column found — falling back to a random 80/20 split. "
            "Air quality is inherently time-dependent (seasonality, daily cycles), so "
            "a random split risks leaking future patterns into training. State this "
            "limitation explicitly in your report."
        )
        train_df = df.sample(frac=0.8, random_state=42)
        test_df = df.drop(train_df.index)
        cv = KFold(n_splits=5, shuffle=True, random_state=42)
        split_note = (
            "Random 80/20 split (no timestamp column available in the dataset). "
            "This is a known limitation: a time-based split is preferable for "
            "air quality data and should be used if timestamps become available."
        )

    print(f"[Split] {split_note}\n")

    X_train, y_train = train_df[FEATURE_ORDER], train_df[target]
    X_test, y_test = test_df[FEATURE_ORDER], test_df[target]

    # ---- 1 & 2. Model comparison via cross-validation ----
    models = build_models()
    print("Running 5-fold cross-validation across candidate models...\n")
    cv_results = run_cv(models, X_train, y_train, cv)

    print(f"{'Model':<24}{'CV R2':>10}{'CV MAE':>12}{'CV RMSE':>12}")
    for name, m in cv_results.items():
        print(f"{name:<24}{m['r2_mean']:>10.3f}{m['mae_mean']:>12.3f}{m['rmse_mean']:>12.3f}")

    best_name = min(cv_results, key=lambda n: cv_results[n]["mae_mean"])
    print(f"\nSelected model (lowest CV MAE): {best_name}\n")

    best_model = models[best_name]
    best_model.fit(X_train, y_train)

    test_preds = best_model.predict(X_test)
    test_metrics = {
        "r2": float(r2_score(y_test, test_preds)),
        "mae": float(mean_absolute_error(y_test, test_preds)),
        "rmse": float(np.sqrt(mean_squared_error(y_test, test_preds))),
    }
    print(f"Held-out test performance ({best_name}):")
    print(f"  R2   = {test_metrics['r2']:.3f}")
    print(f"  MAE  = {test_metrics['mae']:.3f}")
    print(f"  RMSE = {test_metrics['rmse']:.3f}\n")

    # ---- 5. Prediction interval demo (only meaningful for the Random Forest) ----
    interval_note = None
    if best_name == "Random Forest":
        sample_row = X_test.iloc[[0]]
        lo, hi = prediction_interval(best_model, sample_row)
        point = best_model.predict(sample_row)[0]
        interval_note = f"Example: point prediction {point:.1f}, 90% interval [{lo:.1f}, {hi:.1f}]"
        print(f"[Prediction interval] {interval_note}\n")
    elif "Random Forest" in models:
        rf_fallback = models["Random Forest"]
        rf_fallback.fit(X_train, y_train)
        sample_row = X_test.iloc[[0]]
        lo, hi = prediction_interval(rf_fallback, sample_row)
        point = rf_fallback.predict(sample_row)[0]
        interval_note = (
            f"(Random Forest wasn't the best model here, but for reference — "
            f"point prediction {point:.1f}, 90% interval [{lo:.1f}, {hi:.1f}])"
        )
        print(f"[Prediction interval] {interval_note}\n")

    # ---- Save the model exactly how app.py expects it ----
    os.makedirs(os.path.dirname(args.model_out), exist_ok=True)
    with open(args.model_out, "wb") as f:
        pickle.dump(best_model, f)
    print(f"Saved model to {args.model_out} — feature_names_in_ = {list(best_model.feature_names_in_)}\n")

    # ---- Save full comparison for your report/viva ----
    report = {
        "leakage_check": {"linear_r2_full_data": r2_linear, "suspicious": suspicious},
        "split_strategy": split_note,
        "cv_results": cv_results,
        "selected_model": best_name,
        "test_metrics": test_metrics,
        "prediction_interval_example": interval_note,
    }
    with open("model_comparison.json", "w") as f:
        json.dump(report, f, indent=2)
    print("Saved full metrics to model_comparison.json")

    write_justification_md(report)
    print("Saved PROJECT_JUSTIFICATION.md — paste sections directly into your report.")


def write_justification_md(report):
    lines = []
    lines.append("# Model Justification\n")
    lines.append("## Model selection\n")
    lines.append(
        "Three candidates were compared using 5-fold cross-validation rather than "
        "a single train/test split: Linear Regression (baseline), Random Forest "
        "Regressor, and Histogram-based Gradient Boosting. Cross-validated MAE was "
        "used as the selection criterion since it is directly interpretable in AQI units.\n"
    )
    lines.append("| Model | CV R² | CV MAE | CV RMSE |")
    lines.append("|---|---|---|---|")
    for name, m in report["cv_results"].items():
        lines.append(f"| {name} | {m['r2_mean']:.3f} | {m['mae_mean']:.3f} | {m['rmse_mean']:.3f} |")
    lines.append(f"\n**Selected:** {report['selected_model']}, based on lowest cross-validated MAE.\n")

    lines.append("## Data leakage check\n")
    r2l = report["leakage_check"]["linear_r2_full_data"]
    if report["leakage_check"]["suspicious"]:
        lines.append(
            f"A plain Linear Regression on the full feature set already achieves an "
            f"R² of {r2l:.3f}. This is high enough to suggest the AQI target in this "
            f"dataset may be computed directly from these pollutant readings via a "
            f"known sub-index formula, rather than being an independently measured "
            f"quantity. This is disclosed here rather than presenting the model's R² "
            f"as evidence of a difficult, purely statistical prediction problem.\n"
        )
    else:
        lines.append(
            f"A plain Linear Regression on the full feature set achieves an R² of "
            f"{r2l:.3f}, which is not high enough to suggest the target is a direct "
            f"restatement of the inputs — the relationship the model is learning "
            f"appears genuinely non-linear rather than a hidden formula.\n"
        )

    lines.append("## Train/test split strategy\n")
    lines.append(report["split_strategy"] + "\n")

    lines.append("## Held-out test performance\n")
    tm = report["test_metrics"]
    lines.append(f"R² = {tm['r2']:.3f}, MAE = {tm['mae']:.3f}, RMSE = {tm['rmse']:.3f}\n")

    if report["prediction_interval_example"]:
        lines.append("## Prediction uncertainty\n")
        lines.append(
            "Rather than reporting a single point prediction, the variance across "
            "individual trees in the Random Forest can be used to construct an "
            "uncertainty interval:\n"
        )
        lines.append(f"{report['prediction_interval_example']}\n")

    with open("PROJECT_JUSTIFICATION.md", "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
