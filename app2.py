import streamlit as st
import numpy as np
import tensorflow as tf
# from tensorflow.keras.preprocessing import image
from tensorflow import keras

# Define the class indices dictionary
class_indices = {0: 'Half Moon Betta', 1: 'Plakat Betta', 2: 'Crowntail Betta'}

# Define the Streamlit app
st.title("Web Application for Physical Classification of Betta Fish based on Species Standard in Thailand")

# Create a file uploader
uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png", "webp"])

# Load the model
model = keras.models.load_model("model")

# Process the uploaded image and make a prediction
if uploaded_file is not None:
    # Load the image
    img = keras.preprocessing.image.load_img(uploaded_file, target_size=(224,224))
    # img = image.load_img(uploaded_file, target_size=(224,224))
    
    # Preprocess the image
    # img_array = image.img_to_array(img)
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.

    # Predict the label of the image
    pred = model.predict(img_array)
    pred = np.argmax(pred,axis=1)

    # Map the label
    pred_label = class_indices[pred[0]]

    # Display the image
    st.image(img, caption=f"Predicted as: {pred_label}")

    # Display the prediction label
    st.write(f"Prediction: {pred_label}")
