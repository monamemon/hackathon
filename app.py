import streamlit as st
import cv2
import numpy as np
import pytesseract
from PIL import Image

# Configure Tesseract path if needed (Windows users may need to set this)
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def preprocess_image(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Noise reduction
    denoised = cv2.fastNlMeansDenoising(gray, h=30)
    
    # Contrast enhancement using histogram equalization
    enhanced = cv2.equalizeHist(denoised)
    
    # Thresholding (binary image)
    _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    return thresh

def extract_text(image):
    # OCR extraction
    text = pytesseract.image_to_string(image)
    return text

def main():
    st.title("🧾 Receipt OCR App")
    st.write("Upload a receipt image to extract text information.")

    uploaded_file = st.file_uploader("Upload Receipt Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Load image
        image = Image.open(uploaded_file)
        image_np = np.array(image)

        st.image(image, caption="Uploaded Receipt", use_column_width=True)

        # Preprocess
        processed_image = preprocess_image(image_np)

        st.image(processed_image, caption="Preprocessed Image", use_column_width=True, channels="GRAY")

        # Extract text
        extracted_text = extract_text(processed_image)

        st.subheader("Extracted Text")
        st.text(extracted_text)

        # Optional: parse items, quantities, prices, totals
        st.subheader("Parsed Information (Basic Example)")
        lines = extracted_text.split("\n")
        items = [line for line in lines if line.strip()]
        for item in items:
            st.write(item)

if __name__ == "__main__":
    main()
