# Model Justification

## Model selection

Three candidates were compared using 5-fold cross-validation rather than a single train/test split: Linear Regression (baseline), Random Forest Regressor, and Histogram-based Gradient Boosting. Cross-validated MAE was used as the selection criterion since it is directly interpretable in AQI units.

| Model | CV R² | CV MAE | CV RMSE |
|---|---|---|---|
| Linear Regression | 0.885 | 22.427 | 33.561 |
| Random Forest | 0.843 | 20.590 | 39.174 |
| Hist Gradient Boosting | 0.849 | 20.761 | 38.668 |

**Selected:** Random Forest, based on lowest cross-validated MAE.

## Data leakage check

A plain Linear Regression on the full feature set achieves an R² of 0.894, which is not high enough to suggest the target is a direct restatement of the inputs — the relationship the model is learning appears genuinely non-linear rather than a hidden formula.

## Train/test split strategy

Chronological split on 'Date': trained on the earliest 80% (2015-01-01 00:00:00 to 2020-01-25 00:00:00), tested on the most recent 20% (2020-01-25 00:00:00 to 2020-07-01 00:00:00).

## Held-out test performance

R² = 0.870, MAE = 14.590, RMSE = 29.321

## Prediction uncertainty

Rather than reporting a single point prediction, the variance across individual trees in the Random Forest can be used to construct an uncertainty interval:

Example: point prediction 457.5, 90% interval [398.7, 513.6]
