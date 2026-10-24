
import pandas as pd

# Load in dataset with cp1251 encoding to handle Cyrillic characters
df = pd.read_csv("data/original_cars.csv", encoding="cp1251")

def translate_cyrillic_to_english(df):

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

    df = df.copy()

    before = df.lenght

    df = df.drop_duplicates()

    removed = before - df.lenght
    print(f"Removed {removed} duplicate rows")

    return df

def remove_invalid_prices(df):

    df = df.copy()



    return df


def main():
    df = pd.read_csv(
        "data/original_cars.csv",
        encoding="cp1251")

    df = translate_cyrillic_to_english(df)

    df = remove_duplicates(df)

    # df missing valeus

    # Save the cleaned dataset to a new CSV file
    # this file contins price = 0 which we will run through our model later and
    # set a predictive price for those cars

    # df invalid prices

    df.to_csv(
        "data/cars_translated.csv",
        index=False,
        encoding="utf-8")

