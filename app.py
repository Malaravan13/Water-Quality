import streamlit as st
import pickle
import numpy as np

with open("water_potability_svm.sav", "rb") as f:
    model, scaler = pickle.load(f)

st.title("💧 Water Potability Prediction App")

st.write("Enter the water quality parameters to check if the water is potable or not.")

ph = st.number_input("pH", min_value=0.0, max_value=14.0, value=7.0)
hardness = st.number_input("Hardness", min_value=0.0, value=150.0)
solids = st.number_input("Solids", min_value=0.0, value=20000.0)
chloramines = st.number_input("Chloramines", min_value=0.0, value=7.0)
sulfate = st.number_input("Sulfate", min_value=0.0, value=300.0)
conductivity = st.number_input("Conductivity", min_value=0.0, value=400.0)
organic_carbon = st.number_input("Organic Carbon", min_value=0.0, value=10.0)
trihalomethanes = st.number_input("Trihalomethanes", min_value=0.0, value=50.0)
turbidity = st.number_input("Turbidity", min_value=0.0, value=4.0)

if st.button("Check Potability"):
    features = np.array([[ph, hardness, solids, chloramines, sulfate,
                          conductivity, organic_carbon, trihalomethanes, turbidity]])
    
    features_scaled = scaler.transform(features)

    prediction = model.predict(features_scaled)[0]
    probability = model.predict_proba(features_scaled)[0][1]

    if prediction == 1:
        st.success(f"Can Drink(1).")
    else:
        st.error(f"TCannot Drink(0).")
