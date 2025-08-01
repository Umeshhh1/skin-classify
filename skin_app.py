import streamlit as st
import tensorflow as tf
from keras.preprocessing import image
import numpy as np
import tempfile
import os

# Load model
model_path = 'skin_disease_model.h5'
model = tf.keras.models.load_model(model_path)

class_labels = {
    0: 'Acne',
    1: 'Bullous Disease',
    2: 'Cellulitis',
    3: 'Eczema',
    4: 'Melanoma',
    5: 'Nail Fungus'
}

st.title("Skin Disease Image Classifier")
st.write("Upload an image to classify the skin disease.")

uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    try:
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(uploaded_image.read())
        temp_file.close()

        img = image.load_img(temp_file.name, target_size=(150, 150))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0

        with st.spinner("Classifying..."):
            predictions = model.predict(img_array)

        predicted_class_index = np.argmax(predictions)
        predicted_label = class_labels.get(predicted_class_index, "Unknown")
        confidence = np.max(predictions) * 100

        st.image(img, caption=f"Predicted: {predicted_label}", use_column_width=True)
        st.write(f"Predicted Class: {predicted_label}")
        st.write(f"Confidence: {confidence:.2f}%")

    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
    finally:
        os.remove(temp_file.name)
