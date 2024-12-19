import streamlit as st
import numpy as np
import joblib  # To load saved models

# Load the model once at the start of the app
def load_model():
    try:
        model = joblib.load('best_model.pkl')  # Update this path as needed
        return model
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        return None

# Load the model at the start of the app
model = load_model()

# Check if the model is loaded successfully
if model is None:
    st.error("Model not loaded. Please check the file path and try again.")

# Biomass Composition Inputs
biomass_type = st.text_input("Biomass Type")
C_wt = st.number_input("C (wt%)", min_value=0.0, max_value=100.0, value=50.0)
H_wt = st.number_input("H (wt%)", min_value=0.0, max_value=100.0, value=6.0)
N_wt = st.number_input("N (wt%)", min_value=0.0, max_value=100.0, value=2.0)
O_wt = st.number_input("O (wt%)", min_value=0.0, max_value=100.0, value=40.0)
S_wt = st.number_input("S (wt%)", min_value=0.0, max_value=100.0, value=1.0)
ash = st.number_input("Ash (%)", min_value=0.0, max_value=100.0, value=1.0)
operating_dry_matter = st.number_input("Operating dry matter (wt%)", min_value=0.0, max_value=100.0, value=95.0)

# Reaction Conditions Inputs
temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=1000.0, value=300.0)
residence_time = st.number_input("Residence time (min)", min_value=0.0, max_value=1000.0, value=30.0)
pressure = st.number_input("Pressure (MPa)", min_value=0.0, max_value=100.0, value=5.0)

# Prepare the input data
input_data = np.array([C_wt, H_wt, N_wt, O_wt, S_wt, ash, operating_dry_matter, temperature, residence_time, pressure]).reshape(1, -1)

# Predict the outputs
if st.button('Predict'):
    if model is not None:  # Ensure model is loaded before making prediction
        prediction = model.predict(input_data)
        
        # Debugging: Print out the shape and value of the prediction
        st.write("Prediction shape:", prediction.shape)
        st.write("Prediction output:", prediction)
        
        # Display results if prediction has the correct shape
        if prediction.ndim == 2 and prediction.shape[1] == 10:  # Ensure it's a 2D array with 10 outputs
            st.subheader("Predicted Biocrude Properties:")
            st.write(f"Biocrude Oil Yield (%): {prediction[0][0]:.2f}")
            st.write(f"Aqueous Phase Yield (%): {prediction[0][1]:.2f}")
            st.write(f"Syngas Yield (%): {prediction[0][2]:.2f}")
            st.write(f"Hydrochar Yield (%): {prediction[0][3]:.2f}")
            st.write(f"Biocrude Carbon Content (wt%): {prediction[0][4]:.2f}")
            st.write(f"Biocrude Hydrogen Content (wt%): {prediction[0][5]:.2f}")
            st.write(f"Biocrude Nitrogen Content (wt%): {prediction[0][6]:.2f}")
            st.write(f"Biocrude Oxygen Content (wt%): {prediction[0][7]:.2f}")
            st.write(f"Biocrude Sulfur Content (wt%): {prediction[0][8]:.2f}")
            st.write(f"Biocrude Calorific Value (MJ/kg): {prediction[0][9]:.2f}")
        else:
            st.error("Prediction output has unexpected shape.")
    else:
        st.error("Model is not loaded. Please check the model file.")
