import streamlit as st
from PIL import Image
import numpy as np
import cv2

st.set_page_config(page_title="Armrest Height Classification", layout="centered")

st.title("Ergonomic Armrest Height Classifier")
st.markdown("""
Upload a side-profile image of a person working at their desk
""")
uploaded_file = st.file_uploader("Upload a side-profile image", type=['png', 'jpg', 'jpeg', 'webp'])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Analyzing image..."):
        result = classify_armrest_height(image)

    classification = result["classification"]
    annotated_image = result["annotated_image"]
    intermediate = result["intermediate_values"]
    debug_json = result["debug_json"]

    st.header("Classification Result")
    st.markdown(f"### Armrest Height: **{classification}**")

    st.header("Annotated Image with Heights")
    st.image(annotated_image, use_column_width=True)

    st.header("Intermediate Height Values")
    for key, val in intermediate.items():
        st.write(f"- **{key.replace('_', ' ').title()}**: {val} px")

    with st.expander("Show Debug Info"):
        st.json(debug_json)
else:
    st.info("Please upload an image to start analysis.")

