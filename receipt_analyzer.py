# ==========================================
# AI POWERED RECEIPT ANALYZER - STREAMLIT
# ==========================================

import streamlit as st
import cv2
import easyocr
import numpy as np
import re
import os
import requests
from collections import defaultdict
import tempfile

# ------------------------------------------
# IMAGE PREPROCESSING
# ------------------------------------------
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(
        blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )[1]
    return thresh

# ------------------------------------------
# OCR
# ------------------------------------------
def extract_text(image):
    reader = easyocr.Reader(['en'])
    results = reader.readtext(image)
    return "\n".join([res[1] for res in results])

# ------------------------------------------
# PARSE ITEMS
# ------------------------------------------
def parse_items(text):
    items = []
    for line in text.split("\n"):
        match = re.search(r"([A-Za-z ]+)\s+(\d+\.\d{2})", line)
        if match:
            items.append({
                "item": match.group(1).strip(),
                "price": float(match.group(2))
            })
    return items

# ------------------------------------------
# CATEGORIZATION
# ------------------------------------------
CATEGORY_MAP = {
    "milk": "dairy",
    "cheese": "dairy",
    "bread": "bakery",
    "cake": "bakery",
    "chicken": "meat",
    "beef": "meat",
    "chips": "snacks",
    "biscuit": "snacks",
    "chocolate": "snacks"
}

def categorize_items(items):
    for item in items:
        item["category"] = "others"
        for key in CATEGORY_MAP:
            if key in item["item"].lower():
                item["category"] = CATEGORY_MAP[key]
                break
    return items

# ------------------------------------------
# ANALYSIS
# ------------------------------------------
def analyze_spending(items):
    totals = defaultdict(float)
    total_spent = 0

    for item in items:
        totals[item["category"]] += item["price"]
        total_spent += item["price"]

    percentages = {
        k: round((v / total_spent) * 100, 2)
        for k, v in totals.items()
    }
    return total_spent, totals, percentages

# ------------------------------------------
# GROQ LLaMA-3 ADVICE
# ------------------------------------------
def llm_advice(total, percentages):
    api_key = os.getenv("GROQ_API_KEY") or "PASTE_API_KEY_HERE"

    prompt = f"""
You are a financial advisor.
Total spending: {total}
Category percentages: {percentages}

Give short, clear budgeting advice.
"""

    response = requests.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        },
        json={
            "model": "llama3-8b-8192",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.4
        }
    )

    return response.json()["choices"][0]["message"]["content"]

# ------------------------------------------
# STREAMLIT UI
# ------------------------------------------
st.set_page_config(page_title="AI Receipt Analyzer", layout="centered")

st.title("🧾 AI-Powered Receipt Analyzer")
st.write("Upload a receipt image to analyze spending and get AI advice.")

uploaded_file = st.file_uploader("Upload Receipt Image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Receipt", use_column_width=True)

    with tempfile.NamedTemporaryFile(delete=False) as tmp:
        tmp.write(uploaded_file.read())
        image_path = tmp.name

    processed_image = preprocess_image(image_path)
    text = extract_text(processed_image)

    st.subheader("📄 OCR Extracted Text")
    st.text(text)

    items = parse_items(text)
    items = categorize_items(items)

    if items:
        total, totals, percentages = analyze_spending(items)

        st.subheader("📦 Extracted Items")
        st.table(items)

        st.subheader("💰 Spending Summary")
        st.write(f"**Total Spent:** {total}")
        st.write("**Category Totals:**", dict(totals))
        st.write("**Percentages:**", percentages)

        with st.spinner("Generating AI Advice..."):
            advice = llm_advice(total, percentages)

        st.subheader("🤖 AI Budgeting Advice")
        st.success(advice)
    else:
        st.warning("No items detected. Try a clearer receipt.")
