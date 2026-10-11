
import os
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load dataset
data = fetch_california_housing()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# Split dataset: 80% training, 20% validation
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# MLflow settings
mlflow.set_experiment("California Housing Experiments")

experiments = [
    {"max_depth": 3, "learning_rate": 0.1},
    {"max_depth": 5, "learning_rate": 0.05},
    {"max_depth": 7, "learning_rate": 0.01},
]

results = []

for i, params in enumerate(experiments, start=1):
    with mlflow.start_run(run_name=f"Run {i}"):
        model = GradientBoostingRegressor(
            max_depth=params["max_depth"],
            learning_rate=params["learning_rate"],
            random_state=42
        )

        model.fit(X_train, y_train)
        predictions = model.predict(X_val)

        rmse = np.sqrt(mean_squared_error(y_val, predictions))
        mae = mean_absolute_error(y_val, predictions)
        r2 = r2_score(y_val, predictions)

        mlflow.log_param("max_depth", params["max_depth"])
        mlflow.log_param("learning_rate", params["learning_rate"])

        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("R2", r2)

        mlflow.sklearn.log_model(
            sk_model=model,
            name="model",
            serialization_format="pickle"
        )

        results.append({
            "Run": f"Run {i}",
            "Max Depth": params["max_depth"],
            "Learning Rate": params["learning_rate"],
            "RMSE": rmse,
            "MAE": mae,
            "R2": r2
        })

        print(f"Run {i}: RMSE={rmse:.4f}, MAE={mae:.4f}, R2={r2:.4f}")

# Compare experiments using RMSE
results_df = pd.DataFrame(results).sort_values("RMSE")
results_df.to_csv("comparison_results.csv", index=False)

print("\nExperiment comparison:")
print(results_df.to_string(index=False))

best = results_df.iloc[0]
print(f"\nBest model: {best['Run']}")
print(f"Best RMSE: {best['RMSE']:.4f}")
