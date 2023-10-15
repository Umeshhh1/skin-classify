import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import tempfile
import os

# Define the path to the saved model (HDF5 file)
model_path = 'skin_disease_model.h5'  # Update with the correct path

# Load the saved model
model = tf.keras.models.load_model(model_path)
class_labels = {
    0: 'Acne',
    1: 'Bullous Disease',
    2: 'Cellulitis',
    3: 'Eczema',
    4: 'Melanoma',
    5: 'Nail Fungus'
}

# Set Streamlit app title and description
st.title("Skin Disease Image Classifier")
st.write("Upload an image to classify the skin disease.")

# Upload an image for classification
uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_image is not None:
    # Create a temporary file to save the uploaded image
    temp_file = tempfile.NamedTemporaryFile(delete=False)
    temp_file.write(uploaded_image.read())
    temp_file.close()
    
    # Load and preprocess the uploaded image for prediction
    img = image.load_img(temp_file.name, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize the image

    # Make predictions
    predictions = model.predict(img_array)

    # Get the class index with the highest probability
    predicted_class_index = np.argmax(predictions)

    # Get the label name using the class index
    predicted_label = class_labels.get(predicted_class_index, "Unknown")

    # Display the uploaded image
    st.image(img, caption=f"Uploaded Image: {predicted_label}", use_column_width=True)

    # Display the predicted class label
    st.write(f"Predicted Class: {predicted_label}")
    
    # Delete the temporary file
    os.remove(temp_file.name)
