import pandas as pd

df = pd.read_csv("data/cars_cleaned.csv")

duplicates = df.duplicated().sum()
print(f"number of duplicates: {duplicates}")