import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.title("OpenCV Image Processing App")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Convert uploaded file to OpenCV format
    image = Image.open(uploaded_file)
    image = np.array(image)

    st.subheader("Original Image")
    st.image(image, use_column_width=True)

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    st.subheader("Grayscale Image")
    st.image(gray, use_column_width=True)

    # Blur
    blur = cv2.GaussianBlur(image, (5, 5), 0)
    st.subheader("Blur Image")
    st.image(blur, use_column_width=True)

    # Edge Detection
    edges = cv2.Canny(image, 100, 200)
    st.subheader("Edge Detection")
    st.image(edges, use_column_width=True)
