# Used Car Price Prediction - DAT158

A machine learning project that estimates used car prices. We compare linear
regression, a decision tree, and previous and tuned random forests with engine volume.

## Setup

Use Python 3.12. In PyCharm, select a project interpreter with a virtual
environment (`.venv`), then install the packages in its terminal:

```sh
python -m pip install -r requirements.txt
```

The saved model uses scikit-learn 1.6.1.

## Run the project

```sh
python -m streamlit run app.py
```

The app runs at **http://localhost:8501**. Press **Ctrl + C** in the terminal to stop.

To regenerate the cleaned data and retrain the model, run:

```sh
python data_cleaning.py
python model_training.py
```

The first command cleans the original data and saves two files:

- `data/cars_cleaned.csv`: cleaned rows with valid prices for training.
- `data/cars_without_prices.csv`: cleaned rows without usable prices for later prediction.

The second command compares models, tunes the random forest, reports MAE and
RMSE on held-out test data, and saves the fitted pipeline to
`models/car_price_pipeline.joblib`.

For plots and a step-by-step walkthrough, open
`notebooks/data_exploration.ipynb` and run all cells.

Generate a separate CSV of the cars without prices, filled with model estimates in USD:

```sh
python predict_prices.py
```

Generate validation predictions and sort the largest errors first (prices in USD):

```sh
python analyze_validation_errors.py
```

The output includes all training cars and their details, actual prices, predictions,
and errors. `source_row` identifies the row in `cars_cleaned.csv`, including its header.
Each car is predicted while excluded from training; the final test set stays separate.

Cleaning retains engine volume (`engV`) and removes rows above 8 litres. Missing
volumes remain in the CSV and are filled with the training median inside each
pipeline. The normal training script compares both forest versions, searches 13
configurations by validation MAE, evaluates the selected model, and saves it.
It also reports the percentage of predictions within 20% of actual price.
The app accepts engine volume or lets the saved pipeline fill an unknown value.

## Important files

- `app.py`: simple Streamlit website for predicting a car's price.
- `requirements.txt`: package versions for installation.
- `data_cleaning.py`: cleans the data and separates rows by price validity.
- `model_training.py`: trains, compares, evaluates, and saves the model.
- `predict_prices.py`: fills missing prices using the saved model.
- `analyze_validation_errors.py`: generates five-fold validation predictions for error analysis.
- `notebooks/data_exploration.ipynb`: data exploration, plots, and model results.
- `data/original_cars.csv`: original dataset.
- `data/cars_cleaned.csv`: cleaned data for training.
- `data/cars_without_prices.csv`: cleaned rows for later prediction.
- `data/cars_with_predicted_prices.csv`: the same cars with estimated prices in USD, kept separate from training data.
- `data/validation_prediction_errors.csv`: car details and validation errors, largest first.
- `models/car_price_pipeline.joblib`: saved model and preprocessing.

For the final submission, deploy the website or record a demonstration video
and add its link here.

## Data and report

- Dataset: [Car Sale Advertisements on Kaggle](https://www.kaggle.com/datasets/antfarol/car-sale-advertisements).
- Original data: `data/original_cars.csv`.
- Project report: [DAT158_ML_sp.pdf](https://github.com/user-attachments/files/34191864/DAT158_ML_sp.pdf)
.
- Website framework: [Streamlit documentation](https://docs.streamlit.io/).
- Gjennomgang nettside :

https://github.com/user-attachments/assets/fb909e5c-3316-4a5f-9e7a-173c898fcd27



Mileage is recorded in thousands: `68` means 68,000.
Dataset prices are in USD. The website displays NOK using a fixed conversion rate from 22 October 2026.
