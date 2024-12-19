import streamlit as st
import numpy as np
import joblib  # To load saved models

# Load your model 
def load_model():
    try:
        model = joblib.load('best_model.pkl')  
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
        
        # Check if the prediction is a scalar or array
        if isinstance(prediction, np.ndarray):
            if prediction.ndim == 1:
                yield_result = prediction[0]  # For a 1D array, just take the first element
                quality_result = None  # If no second output, set quality_result to None
            else:
                yield_result = prediction[0][0]  # Adjust indexing depending on your model output
                quality_result = prediction[0][1]  # Adjust indexing depending on your model output
        else:
            yield_result = prediction  # If it's a scalar, just return it
            quality_result = None  # If there's no quality result, set to None
        
        return yield_result, quality_result
    except Exception as e:
        st.error(f"Invalid input: {str(e)}")
        return None, None

# Streamlit Layout
def create_streamlit_ui():
    st.title("HTL Prediction for Biocrude Oil")

    # Biomass Composition Inputs
    biomass_type = st.text_input("Biomass Type")  # Text input for biomass type
    C_wt = st.number_input("C (wt%)", min_value=0.0, max_value=100.0, value=50.0)
    H_wt = st.number_input("H (wt%)", min_value=0.0, max_value=100.0, value=6.0)
    N_wt = st.number_input("N (wt%)", min_value=0.0, max_value=100.0, value=2.0)
    O_wt = st.number_input("O (wt%)", min_value=0.0, max_value=100.0, value=40.0)
    S_wt = st.number_input("S (wt%)", min_value=0.0, max_value=100.0, value=1.0)
    ash = st.number_input("Ash (%)", min_value=0.0, max_value=100.0, value=1.0)
    dry_matter = st.number_input("Operating Dry Matter (wt%)", min_value=0.0, max_value=100.0, value=95.0)

    # Reaction Conditions Inputs
    temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=1000.0, value=300.0)
    residence_time = st.number_input("Residence Time (min)", min_value=0.0, max_value=1000.0, value=30.0)
    pressure = st.number_input("Pressure (MPa)", min_value=0.0, max_value=100.0, value=5.0)

    # Make Prediction Button
    if st.button("Predict Biocrude Yield and Quality"):
        biomass_input = [C_wt, H_wt, N_wt, O_wt, S_wt, ash, dry_matter]
        reaction_input = [temperature, residence_time, pressure]
        
        yield_result, quality_result = predict(biomass_input, reaction_input)
        
        if yield_result is not None:
            st.write(f"Predicted Biocrude Yield: {yield_result:.2f}")
            if quality_result is not None:
                st.write(f"Predicted Biocrude Quality: {quality_result:.2f}")
            else:
                st.write("No prediction for quality available.")

# Run the Streamlit app
create_streamlit_ui()
