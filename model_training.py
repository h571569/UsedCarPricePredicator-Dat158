from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import make_scorer, mean_absolute_error, root_mean_squared_error
from sklearn.model_selection import GridSearchCV, KFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor


DATA_DIR = Path(__file__).resolve().parent / "data"
MODEL_PATH = Path(__file__).resolve().parent / "models" / "car_price_pipeline.joblib"


def split_data(df):
    """Separate the car details from price and make an 80/20 split."""
    X = df.drop(columns=["price"])
    y = df["price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test


def create_preprocessor():
    """Set up training-only median filling, scaling, and category encoding."""
    numeric_features = ["mileage", "year", "engV"]
    categorical_features = ["car", "body", "engType", "registration", "model", "drive"]

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_features),
        ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical_features),
    ])

    return preprocessor


def create_model():
    """Combine preparation and linear regression."""
    model = Pipeline([
        ("preprocessor", create_preprocessor()),
        ("regressor", LinearRegression()),
    ])
    return model


def create_decision_tree_model():
    """Combine preparation and a regression tree."""
    model = Pipeline([
        ("preprocessor", create_preprocessor()),
        ("regressor", DecisionTreeRegressor(random_state=42)),
    ])
    return model


def create_random_forest_model():
    """Combine preparation and a random forest."""
    model = Pipeline([
        ("preprocessor", create_preprocessor()),
        ("regressor", RandomForestRegressor(
            n_estimators=100, max_features="sqrt", random_state=42
        )),
    ])
    return model


def create_tuned_random_forest_model():
    """Use the settings selected in the earlier engV tuning experiment."""
    return create_random_forest_model().set_params(
        regressor__n_estimators=300,
        regressor__max_features=0.5,
        regressor__max_depth=30,
        regressor__min_samples_leaf=1,
    )


def within_twenty_percent(actual, predicted):
    """Fraction of estimates within 20% of the observed positive price."""
    return np.mean(np.abs(predicted - actual) <= 0.2 * actual)


SCORING = {
    "MAE": "neg_mean_absolute_error",
    "RMSE": "neg_root_mean_squared_error",
    "within20": make_scorer(within_twenty_percent),
}


def validate_model(model, X_train, y_train):
    """Evaluate a pipeline on the same five folds of training data."""
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_validate(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring=SCORING,
        return_train_score=True,
        error_score="raise",
    )

    fold_results = pd.DataFrame({
        "fold": range(1, 6),
        "training_MAE": -scores["train_MAE"],
        "validation_MAE": -scores["test_MAE"],
        "training_RMSE": -scores["train_RMSE"],
        "validation_RMSE": -scores["test_RMSE"],
        "validation_within20_percent": 100 * scores["test_within20"],
    })
    return fold_results


def validate_linear_regression(X_train, y_train):
    """Validate linear regression without using the held-out test set."""
    return validate_model(create_model(), X_train, y_train)


def validate_decision_tree(X_train, y_train):
    """Validate the decision tree without using the held-out test set."""
    return validate_model(create_decision_tree_model(), X_train, y_train)


def validate_random_forest(X_train, y_train):
    """Validate the random forest without using the held-out test set."""
    return validate_model(create_random_forest_model(), X_train, y_train)


def tune_random_forest(X_train, y_train):
    """Search tree settings using training data."""
    # Twelve structures plus the 300-tree candidate from the earlier experiment.
    param_grid = [
        {
            "regressor__n_estimators": [100],
            "regressor__max_features": ["sqrt", 0.5, 1.0],
            "regressor__max_depth": [None, 30],
            "regressor__min_samples_leaf": [1, 2],
        },
        {
            "regressor__n_estimators": [300],
            "regressor__max_features": [0.5],
            "regressor__max_depth": [30],
            "regressor__min_samples_leaf": [1],
        },
    ]
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        create_random_forest_model(),
        param_grid=param_grid,
        cv=cv,
        scoring=SCORING,
        refit="MAE",
        return_train_score=True,
        error_score="raise",
        n_jobs=2,
        pre_dispatch=2,
    )

    grid_search.fit(X_train, y_train)
    return grid_search


def summarize_tuning(grid_search):
    """Show each setting combination and its mean training/validation errors."""
    scores = grid_search.cv_results_
    grid_results = pd.DataFrame({
        "n_estimators": scores["param_regressor__n_estimators"],
        "max_features": scores["param_regressor__max_features"],
        "max_depth": pd.Series(scores["param_regressor__max_depth"], dtype=object),
        "min_samples_leaf": scores["param_regressor__min_samples_leaf"],
        "training_MAE": -scores["mean_train_MAE"],
        "validation_MAE": -scores["mean_test_MAE"],
        "training_RMSE": -scores["mean_train_RMSE"],
        "validation_RMSE": -scores["mean_test_RMSE"],
        "validation_within20_percent": 100 * scores["mean_test_within20"],
    })
    grid_results["max_depth"] = grid_results["max_depth"].fillna("unlimited")
    return grid_results.sort_values("validation_MAE")


def evaluate_final_model(model, X_test, y_test):
    """Evaluate the already selected, fitted pipeline."""
    test_predictions = model.predict(X_test)
    test_results = pd.DataFrame({
        "MAE": [mean_absolute_error(y_test, test_predictions)],
        "RMSE": [root_mean_squared_error(y_test, test_predictions)],
        "within20_percent": [100 * within_twenty_percent(y_test, test_predictions)],
    }, index=["Held-out test"])
    return test_predictions, test_results


def save_pipeline(model):
    """Save the fitted preparation and regressor together for later predictions."""
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_PATH, compress=3)
    print(f"Saved pipeline to {MODEL_PATH}")
    return MODEL_PATH


def main():
    df = pd.read_csv(DATA_DIR / "cars_cleaned.csv")
    X_train, X_test, y_train, y_test = split_data(df)

    print("Comparing models with five-fold cross-validation...")
    fold_results = validate_linear_regression(X_train, y_train)
    tree_fold_results = validate_decision_tree(X_train, y_train)
    forest_fold_results = validate_random_forest(X_train, y_train)
    tuned_fold_results = validate_model(create_tuned_random_forest_model(), X_train, y_train)

    comparison = pd.DataFrame({
        "Linear regression": fold_results.drop(columns=["fold"]).mean(),
        "Decision tree": tree_fold_results.drop(columns=["fold"]).mean(),
        "Random forest with engV": forest_fold_results.drop(columns=["fold"]).mean(),
        "Tuned random forest with engV": tuned_fold_results.drop(columns=["fold"]).mean(),
    }).T
    print("Mean cross-validation results (price errors in USD, within20 in %):")
    print(comparison.round(2).to_string())

    print("Tuning random forest...")
    grid_search = tune_random_forest(X_train, y_train)
    print(f"Selected settings: {grid_search.best_params_}")

    best_model = grid_search.best_estimator_
    _, test_results = evaluate_final_model(best_model, X_test, y_test)
    print(f"Final evaluation on {len(X_test)} held-out test rows (USD):")
    print(test_results.round(2).to_string())

    save_pipeline(best_model)


if __name__ == "__main__":
    main()
