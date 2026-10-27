from pathlib import Path

import numpy as np
import pandas as pd


DATA_DIR = Path(__file__).resolve().parent / "data"


def translate_cyrillic_to_english(df):
    """Translate known Cyrillic labels without changing the input dataframe."""

    df = df.copy()

    # Translate Cyrillic model names
    model_translations = {
        "1103 Славута": "1103 Slavuta",
        "1102 Таврия": "1102 Tavria",
        "Патриот": "Patriot",
        "969М": "969M",
        "2705 GAZель": "2705 GAZelle",
        "Таврия-Нова": "Tavria-Nova",
        "3302 GAZель": "3302 GAZelle",
        "2752 Соболь": "2752 Sobol",
        "469Б": "469B",
        "Приора": "Priora",
        "969 Волынь": "969 Volyn",
        "3221 GAZель": "3221 GAZelle",
        "2190 Гранта": "2190 Granta",
        "2109 (Балтика)": "2109 (Baltika)",
        "Классические": "Classic"
    }

    df["model"] = df["model"].replace(model_translations)

    # Translate Cyrillic descriptions names
    description_translations = {
        "груз.": "Cargo",
        "груз": "Cargo",
        "пасс.": "Passenger",
        "пасс": "Passenger"
    }

    for original, english in description_translations.items():
        df["model"] = df["model"].str.replace(
            original,
            english,
            regex=False
        )

    # Translate Cyrillic car manufacturer names
    df["car"] = df["car"].replace({
        "ЛUAZ": "LuAZ"
    })

    # Verify that no Cyrillic characters remain
    remaining = df.astype("string").apply(
        lambda col: col.str.contains(
            r"[\u0400-\u052F]",
            na=False
        )
    )

    count = int(remaining.sum().sum())

    print(f"Cyrillic cells remaining: {count}")

    return df

def remove_duplicates(df):
    """Keep the first occurrence of each exact full-row duplicate."""

    before = len(df)
    df = df.drop_duplicates().copy()

    removed = before - len(df)
    print(f"Removed {removed} duplicate rows")

    return df


def remove_engine_volume_column(df):
    """Count missing engine volumes, then remove the column."""
    print(f"Rows missing engV: {df['engV'].isna().sum()}")
    df = df.drop(columns=["engV"])
    return df


def fill_missing_drive(df):
    """Replace missing drive values with the category unknown."""
    df = df.copy()
    missing = df["drive"].isna().sum()
    df["drive"] = df["drive"].fillna("unknown")
    print(f"Filled {missing} missing drive values with unknown")
    return df


def valid_price_mask(df):
    """A usable observed price must be numeric, finite, and greater than zero."""
    prices = pd.to_numeric(df["price"], errors="coerce")
    return prices.notna() & np.isfinite(prices) & prices.gt(0)


def remove_invalid_prices(df):
    """Exclude unusable target prices from data intended for model development."""
    bad_rows = df[~valid_price_mask(df)].index
    cleaned = df.drop(bad_rows)
    cleaned["price"] = pd.to_numeric(cleaned["price"])
    print(f"Excluded {len(df) - len(cleaned)} rows without a valid price")
    return cleaned


def main():
    """Save translated data, valid-price data, and rows for later prediction."""
    df = pd.read_csv(
        DATA_DIR / "original_cars.csv",
        encoding="cp1251")

    df = translate_cyrillic_to_english(df)

    df = remove_engine_volume_column(df)

    df = remove_duplicates(df)

    df = fill_missing_drive(df)

    df.to_csv(
        DATA_DIR / "cars_translated.csv",
        index=False,
        encoding="utf-8")

    # These rows have no usable target. Keep them for later predictions,
    # separate from model training and evaluation.
    without_prices = df.loc[~valid_price_mask(df)].copy()
    without_prices.to_csv(
        DATA_DIR / "cars_without_prices.csv", index=False, encoding="utf-8")

    cleaned = remove_invalid_prices(df)
    cleaned.to_csv(
        DATA_DIR / "cars_cleaned.csv", index=False, encoding="utf-8")

    print(f"Saved {len(df)} translated, deduplicated rows")
    print(f"Saved {len(cleaned)} rows with valid prices")
    print(f"Saved {len(without_prices)} rows for later prediction")


if __name__ == "__main__":
    main()
