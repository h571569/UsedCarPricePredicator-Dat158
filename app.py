"""Run the prediction website with: python -m streamlit run app.py."""

import joblib
import pandas as pd
import streamlit as st

from model_training import DATA_DIR, MODEL_PATH


# Usd to Nok rate from 22 October 2026.
USD_TO_NOK = 9.3601


@st.cache_resource
def load_model():
    """Load the saved pipeline once, without retraining it."""
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_cars():
    """Read car details for dropdown choices and supported input ranges."""
    return pd.read_csv(DATA_DIR / "cars_cleaned.csv").drop(columns="price")


def main():
    st.set_page_config(page_title="Used Car Price Predictor", layout="centered")
    st.title("Used Car Price Predictor")
    st.write("Enter the car's details to estimate its price.")

    try:
        model = load_model()
        cars = load_cars()
    except FileNotFoundError:
        st.error("The saved model or car data is missing. Follow the setup instructions in the README.")
        st.stop()

    default_car = cars.iloc[0]
    manufacturers = sorted(cars["car"].unique())
    left, right = st.columns(2)
    with left:
        manufacturer = st.selectbox(
            "Manufacturer",
            manufacturers,
            index=manufacturers.index(default_car["car"]),
            key="manufacturer",
        )
    with right:
        models = sorted(cars.loc[cars["car"] == manufacturer, "model"].unique())
        default_model = default_car["model"] if manufacturer == default_car["car"] else models[0]
        car_model = st.selectbox(
            "Model", models, index=models.index(default_model), key=f"model_{manufacturer}"
        )

    with st.form("car_details"):
        left, right = st.columns(2)
        with left:
            year = st.number_input(
                "Year",
                min_value=int(cars["year"].min()),
                max_value=int(cars["year"].max()),
                value=int(default_car["year"]),
                step=1,
                help="Choose a year covered by the source dataset.",
            )
            body = st.selectbox(
                "Body type", sorted(cars["body"].unique()),
                format_func=lambda value: "Wagon" if value == "vagon" else value.capitalize(),
            )
            registration = st.selectbox(
                "Registered", ["yes", "no"], format_func=str.capitalize
            )
        with right:
            mileage_km = st.number_input(
                "Mileage (km)",
                min_value=0,
                max_value=int(cars["mileage"].max() * 1000),
                value=int(default_car["mileage"] * 1000),
                step=1000,
            )
            fuel = st.selectbox("Fuel type", sorted(cars["engType"].unique()))
            drive = st.selectbox(
                "Drive", sorted(cars["drive"].unique()), format_func=str.capitalize
            )
        submitted = st.form_submit_button("Predict price", type="primary")

    if submitted:
        # The training dataset records mileage in thousands of kilometres.
        inputs = pd.DataFrame([{
            "car": manufacturer,
            "body": body,
            "mileage": mileage_km / 1000,
            "engType": fuel,
            "registration": registration,
            "year": year,
            "model": car_model,
            "drive": drive,
        }])
        prediction_usd = float(model.predict(inputs)[0])
        prediction_nok = prediction_usd * USD_TO_NOK
        formatted_price = f"{prediction_nok:,.0f}".replace(",", " ")
        st.metric("Estimated price", f"{formatted_price} NOK")


if __name__ == "__main__":
    main()
