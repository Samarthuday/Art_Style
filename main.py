import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array
import matplotlib.pyplot as plt

# Load the pre-trained model
model = tf.keras.models.load_model('art_style_transfer_cnn_model.h5')

# Function to preprocess the uploaded images
def preprocess_image(image, target_size=(128, 128)):
    img = load_img(image, target_size=target_size)
    img_array = img_to_array(img) / 255.0  # Normalize image
    return np.expand_dims(img_array, axis=0)  # Add batch dimension

# Function to generate a stylized image (simplified for demonstration)
def generate_stylized_image(content_image, style_image):
    # For demo purposes, applying a simple blend
    return content_image * 0.5 + style_image * 0.5

# Streamlit App
st.title("Art Style Transfer")

# Upload Content Image
st.subheader("Upload Content Image")
content_image_file = st.file_uploader("Choose a content image...", type=["jpg", "jpeg", "png", "webp"])

# Upload Style Image
st.subheader("Upload Style Image")
style_image_file = st.file_uploader("Choose a style image...", type=["jpg", "jpeg", "png", "webp"])

if content_image_file is not None and style_image_file is not None:
    # Preprocess uploaded images
    content_image = preprocess_image(content_image_file)
    style_image = preprocess_image(style_image_file)

    # Generate stylized image
    stylized_image = generate_stylized_image(content_image[0], style_image[0])  # Remove batch dimension

    # Display images
    st.subheader("Content Image")
    st.image(content_image_file, use_column_width=True)

    st.subheader("Style Image")
    st.image(style_image_file, use_column_width=True)

    st.subheader("Stylized Output")
    st.image(stylized_image, use_column_width=True)

# Run the Streamlit app
if __name__ == "__main__":
    st.write("Developed by [Samarth]")  # Replace with your name or information
