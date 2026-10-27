# UsedCarPricePredicator - Dat158

Used car price prediction project for DAT158.

The project currently contains car datasets and Python data preparation code using pandas.

- `data/original_cars.csv`: original car dataset.
- `data/cars_translated.csv`: translated data with the `engV` column and exact duplicate rows removed.
- `data/cars_cleaned.csv`: deduplicated data with valid, positive prices for later model development, without `engV`.
- `data/cars_without_prices.csv`: rows without usable prices, kept for later prediction.
- `data_cleaning.py`: reusable translation, engine-volume column removal, duplicate removal, missing-drive filling, and price validation functions.
- `model_training.py`: compares three regression models, tunes random forest, and evaluates the selected pipeline on held-out test rows.
- `notebooks/data_exploration.ipynb`: runnable cleaning, training-data plots, feature preparation, and regression model comparisons.
- `models/car_price_pipeline.joblib`: saved selected random forest pipeline, including fitted preprocessing.

## Setup

```sh
python -m venv .venv
source .venv/bin/activate
pip install pandas numpy scikit-learn matplotlib jupyter joblib
```

Regenerate the prepared CSV files from the original data:

```sh
python data_cleaning.py
```

Open `notebooks/data_exploration.ipynb` and run all cells to see the same steps
and their results. The notebook can start from the project folder or `notebooks/`.
The notebook shows the missing `engV` count before dropping that column.
Rows are no longer removed based on engine volume. Duplicates are checked after
dropping `engV`, because removing a column can make additional rows identical.
Missing `drive` values are filled with `unknown` in all prepared CSV files.
Section 7 uses the same split as `model_training.py` to plot price distribution,
price against year, and price against mileage using training rows only.
Section 8 shows how mileage and year are scaled and text categories become 0/1
columns. The preparation learns only from training data. Its code matches
`create_preprocessor()` in `model_training.py`.
Section 9 fits a linear regression pipeline and shows training predictions in
a table and plot. Section 10 validates linear regression using five folds and
shows training/validation MAE and RMSE in tables.
Section 11 trains a decision tree, shows example predictions, and compares its
cross-validation errors with linear regression. Validation results use tables.
Section 12 trains a random forest and adds its validation errors to the comparison.
Section 13 searches nine combinations of forest depth and minimum leaf size,
shows their errors, and compares the selected settings with the initial forest.
Section 14 evaluates the selected pipeline on the held-out test rows, reports
MAE and RMSE, and shows five example test predictions. The pipeline is not
refitted on test data, and its settings are not changed based on these results.
Section 15 saves the fitted pipeline using `joblib`.

Compare models, tune random forest, and evaluate the selected model:

```sh
python model_training.py
```

`split_data(df)` returns `X_train`, `X_test`, `y_train`, and `y_test`.
The split uses `random_state=42`, giving the same split when the input data is
unchanged. Only `cars_cleaned.csv` is used; rows without usable prices remain
separate. Preprocessing is fitted using training data, with test data reserved for final
evaluation. `StandardScaler` standardizes mileage and year; `OneHotEncoder`
creates a 0/1 column for each category. Unseen categories are handled with
`handle_unknown="ignore"`. `create_model()` combines this preparation with
`LinearRegression`. Running the script fits the pipeline on training rows, prints
five training predictions, and reports cross-validation errors. Each fold fits
a fresh pipeline, including preprocessing, using only that fold's training rows.
`KFold(n_splits=5, shuffle=True, random_state=42)` makes the folds repeatable.
The held-out test set is reserved for final evaluation.
`create_decision_tree_model()` uses `DecisionTreeRegressor(random_state=42)`
to create a regression tree pipeline.
All models use `validate_model()` with the same folds and metrics. The first
tree uses default complexity settings; limiting its depth or leaf size is left
for tuning. `create_random_forest_model()` combines trees
with randomized samples and feature choices. It starts with 100 trees,
`max_features="sqrt"`, and `random_state=42`.

`tune_random_forest()` uses `GridSearchCV`
to try `max_depth` values of unlimited (`None`), 20, and 40, and
`min_samples_leaf` values of 1, 2, and 5. The grid includes the initial settings
and keeps the tree count at 100. These nine combinations are project choices.
It uses the same five folds, refits preparation within each fold, and selects
the lowest mean validation MAE while also reporting RMSE. `refit="MAE"`
automatically fits the selected pipeline on all training rows.
The tuning scores are used to choose settings. `evaluate_final_model()` then
predicts the held-out test prices and calculates MAE and RMSE.
The test results are for final reporting, rather than further setting selection.

`save_pipeline(best_model)` saves the fitted preprocessing and selected forest
together to `models/car_price_pipeline.joblib`. The script saves it after final
evaluation. The file was created using scikit-learn 1.6.1; use the project's
existing `.venv` to load it. Input mileage must remain in thousands: 68 represents
68,000. Input columns are `car`, `body`, `mileage`, `engType`, `registration`,
`year`, `model`, and `drive`.

Load the saved pipeline without retraining:

```python
import joblib
from model_training import MODEL_PATH

pipeline = joblib.load(MODEL_PATH)
```


##Lenke for dokumentet: 
https://hvl365-my.sharepoint.com/:w:/r/personal/184906_stud_hvl_no/Documents/DAT158_ML_sp.docx?d=waa390cb1d4b94d99932ba5aa8b4de27a&csf=1&web=1&e=UReRgJ
