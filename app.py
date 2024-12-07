import numpy as np
import streamlit as st
import tensorflow as tf

# Load your model
model = tf.keras.models.load_model("D:\PW Skills\Internship\Regression_technique_for_TurboEngine_RUL-main\Regression_technique_for_TurboEngine_RUL-main\models\model_1.h5")

# Prediction function
def predict(input_data):
    # Reshape the input data to match the model's expected shape (25, 13, 1)
    input_data = np.array(input_data).reshape(1, 25, 13, 1)  # Adjust according to your model's input shape
    prediction = model.predict(input_data)
    return prediction

# Streamlit UI for input
st.title("Turbo Engine RUL Prediction")

# Dataset description for the sidebar
dataset_description = """
### Dataset Description

The dataset consists of multiple multivariate time series data collected from engines. Each engine starts with different initial wear and variation, which is normal and not considered a fault. There are three operational settings affecting engine performance. The data includes sensor noise.

- **Engine Operation**: Each engine starts normally and develops a fault during the series. The training data shows fault growth until failure, while the test data ends before failure.
- **Objective**: The goal is to predict the Remaining Useful Life (RUL) of the engine before failure based on sensor readings and operational settings.
  
**Data Sets:**
- **FD001**: 100 training trajectories, 100 test trajectories, Sea Level conditions, and one fault mode (HPC Degradation).
- **FD002**: 260 training trajectories, 259 test trajectories, six conditions, and one fault mode (HPC Degradation).
- **FD003**: 100 training trajectories, 100 test trajectories, Sea Level conditions, and two fault modes (HPC Degradation, Fan Degradation).
- **FD004**: 248 training trajectories, 249 test trajectories, six conditions, and two fault modes (HPC Degradation, Fan Degradation).

The data consists of 26 columns: 
1. Unit number 
2. Time (cycles) 
3. Operational settings (3 columns) 
4. Sensor measurements (26 columns).
"""

# Display dataset description in the sidebar
st.sidebar.header("Dataset Description")
st.sidebar.markdown(dataset_description)  # Display the dataset description

# Input fields for operational settings and sensors
operational_setting_1 = st.slider("Operational Setting 1 (units: %)", 0, 100, 50)
operational_setting_2 = st.slider("Operational Setting 2 (units: %)", 0, 100, 50)
operational_setting_3 = st.slider("Operational Setting 3 (units: %)", 0, 100, 50)
sensor_1 = st.slider("Sensor 1 (units: °C)", 0.0, 100.0, 50.0)
sensor_2 = st.slider("Sensor 2 (units: bar)", 0.0, 100.0, 50.0)
sensor_3 = st.slider("Sensor 3 (units: RPM)", 0.0, 1000.0, 500.0)

# Create the input data array
input_data = [
    operational_setting_1, operational_setting_2, operational_setting_3,
    sensor_1, sensor_2, sensor_3
]

# Example of padding or expanding the input data (if needed)
# If you need 25 sensors with 13 time steps, you'll need to replicate data or input it manually
# This is just for example purposes and should be replaced with actual time-series data

# Assuming you only have 6 sensor values, replicate them to match the expected shape
# If this is not how your data is structured, modify the input preparation accordingly

input_data_expanded = np.pad(input_data, (0, 19))  # Pad with zeros to match 25 values
input_data_expanded = np.tile(input_data_expanded, (13, 1))  # Replicate for 13 time steps
input_data_expanded = input_data_expanded.reshape(1, 25, 13, 1)  # Reshape to expected input shape

# Make the prediction
if st.button("Predict RUL"):
    prediction = predict(input_data_expanded)
    predicted_rul = prediction[0][0]
    
    # Display the predicted RUL
    st.write(f"Predicted RUL: {predicted_rul:.2f} cycles")
    
    # Display a message about what will happen after the predicted RUL
    if predicted_rul < 10:
        st.write("The engine is predicted to fail very soon.")
    else:
        st.write(f"After {predicted_rul:.2f} cycles, the engine will likely fail. Maintenance is recommended.")