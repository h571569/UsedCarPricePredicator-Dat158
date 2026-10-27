# Used Car Price Prediction - DAT158

A machine learning project that estimates used car prices. We compare linear
regression, decision tree, and random forest models.

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

## Important files

- `app.py`: simple Streamlit website for predicting a car's price.
- `requirements.txt`: package versions for installation.
- `data_cleaning.py`: cleans the data and separates rows by price validity.
- `model_training.py`: trains, compares, evaluates, and saves the model.
- `notebooks/data_exploration.ipynb`: data exploration, plots, and model results.
- `testing.py`: checks the cleaned training data for duplicates.
- `data/original_cars.csv`: original dataset.
- `data/cars_cleaned.csv`: cleaned data for training.
- `data/cars_without_prices.csv`: cleaned rows for later prediction.
- `models/car_price_pipeline.joblib`: saved model and preprocessing.

For the final submission, deploy the website or record a demonstration video
and add its link here.

## Data and report

- Dataset: [Car Sale Advertisements on Kaggle](https://www.kaggle.com/datasets/antfarol/car-sale-advertisements).
- Original data: `data/original_cars.csv`.
- [Project report (shared document)](https://hvl365-my.sharepoint.com/:w:/r/personal/184906_stud_hvl_no/Documents/DAT158_ML_sp.docx?d=waa390cb1d4b94d99932ba5aa8b4de27a&csf=1&web=1&e=UReRgJ).
- Website framework: [Streamlit documentation](https://docs.streamlit.io/).

Mileage is recorded in thousands: `68` means 68,000.
