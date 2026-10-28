import pandas as pd
from sklearn.model_selection import KFold

from model_training import DATA_DIR, create_random_forest_model, split_data


OUTPUT_PATH = DATA_DIR / "validation_prediction_errors.csv"


def main():
    cars = pd.read_csv(DATA_DIR / "cars_cleaned.csv")
    X_train, _, y_train, _ = split_data(cars)
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    results = X_train.copy()
    # CSV row numbers include the header, so the first car is on row 2.
    results.insert(0, "source_row", results.index + 2)
    results.insert(1, "fold", 0)
    results["actual_price"] = y_train
    results["predicted_price"] = 0.0

    for fold, (fit_rows, validation_rows) in enumerate(cv.split(X_train), start=1):
        print(f"Training and predicting validation fold {fold}/5...", flush=True)
        model = create_random_forest_model()
        model.fit(X_train.iloc[fit_rows], y_train.iloc[fit_rows])
        predictions = model.predict(X_train.iloc[validation_rows])

        validation_indices = X_train.index[validation_rows]
        results.loc[validation_indices, "predicted_price"] = predictions
        results.loc[validation_indices, "fold"] = fold

    results["prediction_error"] = results["predicted_price"] - results["actual_price"]
    results["absolute_error"] = results["prediction_error"].abs()
    results = results.sort_values("absolute_error", ascending=False, kind="stable")
    results.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")

    print(f"\nSaved validation errors for {len(results)} cars to {OUTPUT_PATH}")
    print(f"Validation MAE: {results['absolute_error'].mean():,.2f} USD")
    print("\nTen largest errors (prices and errors in USD):")
    columns = [
        "source_row", "car", "model", "year", "mileage",
        "actual_price", "predicted_price", "absolute_error",
    ]
    print(results[columns].head(10).round(2).to_string(index=False))


if __name__ == "__main__":
    main()
