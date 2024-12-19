import tkinter as tk
from tkinter import messagebox
from sklearn.externals import joblib  # To load saved models
import numpy as np
import pandas as pd

# Load your model (assumed to be saved as 'htl_model.pkl')
def load_model():
    try:
        model = joblib.load('htl_model.pkl')  # Adjust the path if needed
        return model
    except Exception as e:
        messagebox.showerror("Error", f"Failed to load model: {str(e)}")
        return None

# Function to make predictions
def predict():
    model = load_model()
    if model is None:
        return
    
    try:
        # Get input data from the user (biomass composition and reaction conditions)
        biomass_input = [float(entry.get()) for entry in biomass_entries]  # Biomass composition inputs
        reaction_input = [float(entry.get()) for entry in reaction_entries]  # Reaction condition inputs
        
        # Combine biomass and reaction condition inputs
        user_input = np.array(biomass_input + reaction_input).reshape(1, -1)  # Reshape for prediction
        
        # Make prediction
        prediction = model.predict(user_input)
        
        # Display the results (biocrude yield and quality)
        yield_result = prediction[0][0]  # Adjust indexing depending on your model output
        quality_result = prediction[0][1]  # Adjust indexing depending on your model output
        
        result_label.config(text=f"Predicted Biocrude Yield: {yield_result:.2f}\nPredicted Biocrude Quality: {quality_result:.2f}")
    except Exception as e:
        messagebox.showerror("Error", f"Invalid input: {str(e)}")

# GUI Layout
def create_gui():
    # Create the root window
    root = tk.Tk()
    root.title("HTL Prediction for Biocrude Oil")

    # Labels for Biomass Composition Inputs
    biomass_labels = ["Moisture (%)", "Protein (%)", "Carbohydrates (%)", "Lipids (%)"]
    global biomass_entries
    biomass_entries = []

    # Create Entry widgets for biomass composition
    for label in biomass_labels:
        frame = tk.Frame(root)
        frame.pack(pady=5)
        
        label_widget = tk.Label(frame, text=label)
        label_widget.pack(side=tk.LEFT)
        
        entry_widget = tk.Entry(frame)
        entry_widget.pack(side=tk.LEFT)
        biomass_entries.append(entry_widget)

    # Labels for Reaction Condition Inputs
    reaction_labels = ["Temperature (°C)", "Pressure (MPa)", "Reaction Time (min)"]
    global reaction_entries
    reaction_entries = []

    # Create Entry widgets for reaction conditions
    for label in reaction_labels:
        frame = tk.Frame(root)
        frame.pack(pady=5)
        
        label_widget = tk.Label(frame, text=label)
        label_widget.pack(side=tk.LEFT)
        
        entry_widget = tk.Entry(frame)
        entry_widget.pack(side=tk.LEFT)
        reaction_entries.append(entry_widget)

    # Create a button to make predictions
    predict_button = tk.Button(root, text="Predict Biocrude Yield and Quality", command=predict)
    predict_button.pack(pady=10)
    
    # Label to display the result
    global result_label
    result_label = tk.Label(root, text="Predicted Biocrude Yield and Quality: N/A")
    result_label.pack(pady=20)
    
    # Run the GUI loop
    root.mainloop()

# Start the GUI
create_gui()
