import streamlit as st
import tensorflow as tf
from tensorflow import keras
from PIL import Image, ImageOps
import numpy as np

# Load the trained model
model = keras.models.load_model('mask_detector_model.h5')

st.title("Face Mask Detection")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

if uploaded_file is not None:
    # Convert the file to an opencv image.
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    
    # Preprocess the image
    image = ImageOps.fit(image, (128, 128), Image.LANCZOS)
    image_array = np.asarray(image)
    image_scaled = image_array / 255.0
    image_reshaped = np.reshape(image_scaled, (1, 128, 128, 3))
    
    # Predict the image
    prediction = model.predict(image_reshaped)
    pred_label = np.argmax(prediction)
    
    if pred_label == 1:
        st.write("The person in the image is wearing a mask.")
    else:
        st.write("The person in the image is not wearing a mask.")
