import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from keras.models import load_model
from sklearn.model_selection import train_test_split

# Load the pre-trained model
model = load_model('CNN_Traffic.h5')

# Initialize the StandardScaler (same as in your training code)
scaler = StandardScaler()

# Streamlit interface for user input
st.title('Traffic Prediction using CNN')

# Input data (example with random values, adjust to your actual feature inputs)
st.write("Enter the features for prediction:")
input_features = []

# Collecting user input for features (adjust this for your specific features)
# For simplicity, assume the input has 10 features (adjust as needed)
for i in range(10):
    input_value = st.number_input(f'Feature {i+1}', min_value=0, max_value=1000, step=1)
    input_features.append(input_value)

# Convert input features to numpy array
input_features = np.array(input_features).reshape(1, -1)

# Scale the input features (using the same scaler as in training)
scaled_input = scaler.fit_transform(input_features)  # Fit and transform for demonstration purposes

# Expand dims to match the model's expected input shape (1 sample, 10 features, 1 channel)
scaled_input = np.expand_dims(scaled_input, axis=2)

# Make predictions when button is clicked
if st.button('Predict'):
    prediction = model.predict(scaled_input)
    st.write(f'Prediction: {prediction[0][0]:.4f}')  # Output prediction


