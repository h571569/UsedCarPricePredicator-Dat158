from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
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

    # Split before fitting preprocessing steps or models.
    # random_state makes the split repeatable for the same dataset.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    return X_train, X_test, y_train, y_test


def create_preprocessor():
    """Set up numeric scaling and categorical encoding."""
    numeric_features = ["mileage", "year"]
    categorical_features = ["car", "body", "engType", "registration", "model", "drive"]

    preprocessor = ColumnTransformer([
        ("numeric", StandardScaler(), numeric_features),
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


def validate_model(model, X_train, y_train):
    """Evaluate a pipeline on the same five folds of training data."""
    # Use five shuffled folds and a fixed seed for repeatable validation.
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_validate(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring={"MAE": "neg_mean_absolute_error", "RMSE": "neg_root_mean_squared_error"},
        return_train_score=True,
        error_score="raise",
    )

    # Scikit-learn returns negative error scores. Convert them to positive errors.
    # Its "test" keys refer to validation folds, not our held-out test set.
    fold_results = pd.DataFrame({
        "fold": range(1, 6),
        "training_MAE": -scores["train_MAE"],
        "validation_MAE": -scores["test_MAE"],
        "training_RMSE": -scores["train_RMSE"],
        "validation_RMSE": -scores["test_RMSE"],
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
    # Try nine combinations of tree depth and minimum leaf size.
    param_grid = {
        "regressor__max_depth": [None, 20, 40],
        "regressor__min_samples_leaf": [1, 2, 5],
    }
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        create_random_forest_model(),
        param_grid=param_grid,
        cv=cv,
        scoring={"MAE": "neg_mean_absolute_error", "RMSE": "neg_root_mean_squared_error"},
        refit="MAE",
        return_train_score=True,
        error_score="raise",
        verbose=1,
    )
    # The complete pipeline is fitted separately within each fold.
    # refit="MAE" then fits the selected pipeline on all training rows.
    grid_search.fit(X_train, y_train)
    return grid_search


def summarize_tuning(grid_search):
    """Show each setting combination and its mean training/validation errors."""
    scores = grid_search.cv_results_
    grid_results = pd.DataFrame({
        "max_depth": pd.Series(scores["param_regressor__max_depth"], dtype=object),
        "min_samples_leaf": scores["param_regressor__min_samples_leaf"],
        "training_MAE": -scores["mean_train_MAE"],
        "validation_MAE": -scores["mean_test_MAE"],
        "training_RMSE": -scores["mean_train_RMSE"],
        "validation_RMSE": -scores["mean_test_RMSE"],
    })
    grid_results["max_depth"] = grid_results["max_depth"].fillna("unlimited")
    return grid_results.sort_values("validation_MAE")


def evaluate_final_model(model, X_test, y_test):
    """Evaluate the already selected, fitted pipeline."""
    test_predictions = model.predict(X_test)
    test_results = pd.DataFrame({
        "MAE": [mean_absolute_error(y_test, test_predictions)],
        "RMSE": [root_mean_squared_error(y_test, test_predictions)],
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

    print(f"Input features: {X_train.columns.tolist()}")
    print(f"Training rows: {len(X_train)}; training prices: {len(y_train)}")
    print(f"Test rows: {len(X_test)}; test prices: {len(y_test)}")

    model = create_model()
    model.fit(X_train, y_train)
    print("Linear regression trained.")

    feature_names = model.named_steps["preprocessor"].get_feature_names_out()
    print(f"Prepared input features: {len(feature_names)}")

    examples = X_train[["car", "model", "year", "mileage"]].head().copy()
    examples["actual_price"] = y_train.head()
    examples["predicted_price"] = model.predict(X_train.head())
    print("Example predictions for training cars (not validation results):")
    print(examples.round(2).to_string(index=False))

    fold_results = validate_linear_regression(X_train, y_train)
    print("Linear regression: five-fold cross-validation on training data")
    print(fold_results.round(2).to_string(index=False))
    print("Mean errors across the five folds:")
    print(fold_results.drop(columns=["fold"]).mean().round(2).to_string())

    tree_fold_results = validate_decision_tree(X_train, y_train)
    print("Decision tree: five-fold cross-validation on training data")
    print(tree_fold_results.round(2).to_string(index=False))

    forest_fold_results = validate_random_forest(X_train, y_train)
    print("Random forest: five-fold cross-validation on training data")
    print(forest_fold_results.round(2).to_string(index=False))

    comparison = pd.DataFrame({
        "Linear regression": fold_results.drop(columns=["fold"]).mean(),
        "Decision tree": tree_fold_results.drop(columns=["fold"]).mean(),
        "Random forest": forest_fold_results.drop(columns=["fold"]).mean(),
    }).T
    print("Mean cross-validation errors by model:")
    print(comparison.round(2).to_string())

    grid_search = tune_random_forest(X_train, y_train)
    grid_results = summarize_tuning(grid_search)
    print("Random forest tuning: mean errors across five folds")
    print(grid_results.round(2).to_string(index=False))
    print(f"Selected settings: {grid_search.best_params_}")
    tuned_errors = grid_results.loc[grid_search.best_index_, comparison.columns].astype(float)
    tuning_comparison = pd.DataFrame({
        "Initial random forest": comparison.loc["Random forest"],
        "Selected random forest": tuned_errors,
    }).T
    print("Initial and selected settings (cross-validation errors):")
    print(tuning_comparison.round(2).to_string())
    print(f"Selected pipeline refitted on all {len(X_train)} training rows.")

    # Model selection is finished before the held-out test data is evaluated.
    best_model = grid_search.best_estimator_
    test_predictions, test_results = evaluate_final_model(best_model, X_test, y_test)
    print(f"Final evaluation on {len(X_test)} held-out test rows:")
    print(test_results.round(2).to_string())

    save_pipeline(best_model)


if __name__ == "__main__":
    main()
