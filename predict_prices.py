"""Fill missing car prices using the saved pipeline, with prices in USD."""

import joblib
import pandas as pd

from model_training import DATA_DIR, MODEL_PATH


OUTPUT_PATH = DATA_DIR / "cars_with_predicted_prices.csv"


def main():
    cars = pd.read_csv(DATA_DIR / "cars_without_prices.csv")
    pipeline = joblib.load(MODEL_PATH)

    # Keep the same cars and columns, replacing unusable prices with estimates.
    cars["price"] = pipeline.predict(cars.drop(columns="price")).round(2)
    cars.to_csv(OUTPUT_PATH, index=False, encoding="utf-8")
    print(f"Saved {len(cars)} cars with predicted prices in USD to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
