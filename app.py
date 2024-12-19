import streamlit as st
import numpy as np
import joblib  # To load saved models

# Load your model (assumed to be saved as 'htl_model.pkl')
def load_model():
    try:
        model = joblib.load('htl_model.pkl')  # Adjust the path if needed
        return model
    except Exception as e:
        st.error(f"Failed to load model: {str(e)}")
        return None

# Function to make predictions
def predict(biomass_input, reaction_input):
    model = load_model()
    if model is None:
        return None
    
    try:
        # Combine biomass and reaction condition inputs
        user_input = np.array(biomass_input + reaction_input).reshape(1, -1)  # Reshape for prediction
        
        # Make prediction
        prediction = model.predict(user_input)
        
        # Display the results (biocrude yield and quality)
        yield_result = prediction[0][0]  # Adjust indexing depending on your model output
        quality_result = prediction[0][1]  # Adjust indexing depending on your model output
        
        return yield_result, quality_result
    except Exception as e:
        st.error(f"Invalid input: {str(e)}")
        return None, None

# Streamlit Layout
def create_streamlit_ui():
    st.title("HTL Prediction for Biocrude Oil")

    # Biomass Composition Inputs
    moisture = st.number_input("Moisture (%)", min_value=0.0, max_value=100.0, value=10.0)
    protein = st.number_input("Protein (%)", min_value=0.0, max_value=100.0, value=10.0)
    carbohydrates = st.number_input("Carbohydrates (%)", min_value=0.0, max_value=100.0, value=10.0)
    lipids = st.number_input("Lipids (%)", min_value=0.0, max_value=100.0, value=10.0)

    # Reaction Conditions Inputs
    temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=1000.0, value=300.0)
    pressure = st.number_input("Pressure (MPa)", min_value=0.0, max_value=100.0, value=5.0)
    reaction_time = st.number_input("Reaction Time (min)", min_value=0.0, max_value=1000.0, value=30.0)

    # Make Prediction Button
    if st.button("Predict Biocrude Yield and Quality"):
        biomass_input = [moisture, protein, carbohydrates, lipids]
        reaction_input = [temperature, pressure, reaction_time]
        
        yield_result, quality_result = predict(biomass_input, reaction_input)
        
        if yield_result is not None:
            st.write(f"Predicted Biocrude Yield: {yield_result:.2f}")
            st.write(f"Predicted Biocrude Quality: {quality_result:.2f}")

# Run the Streamlit app
create_streamlit_ui()
