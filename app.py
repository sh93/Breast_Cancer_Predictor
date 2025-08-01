import streamlit as st
import pickle
import os
import pandas as pd


model_path = "final_breast_cancer.pkl"
with open(model_path, "rb") as model_file:
    lg_model = pickle.load(model_file)


feature_names = [
    "mean radius",
    "mean texture",
    "mean perimeter",
    "mean area",
    "mean smoothness",
    "mean compactness",
    "mean concavity",
    "mean concave points",
    "mean symmetry",
    "mean fractal dimension",
    "radius error",
    "texture error",
    "perimeter error",
    "area error",
    "smoothness error",
    "compactness error",
    "concavity error",
    "concave points error",
    "symmetry error",
    "fractal dimension error",
    "worst radius",
    "worst texture",
    "worst perimeter",
    "worst area",
    "worst smoothness",
    "worst compactness",
    "worst concavity",
    "worst concave points",
    "worst symmetry",
    "worst fractal dimension",
]


sample_input = [
    17.99,
    10.38,
    122.80,
    1001.0,
    0.11840,
    0.27760,
    0.3001,
    0.14710,
    0.2419,
    0.07871,
    1.0950,
    0.9053,
    8.589,
    153.40,
    0.006399,
    0.04904,
    0.05373,
    0.01587,
    0.03003,
    0.006193,
    25.38,
    17.33,
    184.60,
    2019.0,
    0.1622,
    0.6656,
    0.7119,
    0.2654,
    0.4601,
    0.11890,
]


def take_inputs(features):
    input_df = pd.DataFrame([features], columns=feature_names)
    prediction = lg_model.predict(input_df)[0]
    return prediction


def main():
    st.title("Breast Cancer Detection")
    st.markdown("Enter the 30 features below or autofill with sample values.")

    autofill = st.checkbox("Use sample input (first row from UCI dataset)")
    inputs = []

    for i, feature in enumerate(feature_names):
        if autofill:
            value = st.number_input(
                f"{feature}", format="%.5f", value=float(sample_input[i])
            )
        else:
            value = st.number_input(f"{feature}", format="%.5f")
        inputs.append(value)

    if st.button("Predict"):
        counter = 30
        for i in inputs:
            if abs(i) == 0.00000:
                counter -= 1
        if counter == 0:
            st.error("Enter the inputs")
        else:
            prediction = take_inputs(inputs)
            result = (
                "Malignant (Cancerous)" if prediction == 1 else "Benign (Non-cancerous)"
            )
            st.success(f"Prediction: {result}")


if __name__ == "__main__":
    main()
