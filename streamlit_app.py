import streamlit as st
from PIL import Image
import io
from flow import process_image_flow
from classify import classify_armrest_height

st.set_page_config(page_title="Armrest Height Classification", layout="centered")
st.title("Ergonomic Armrest Height Classifier")
st.markdown("Upload a side-profile image of a person working at their desk")

uploaded_file = st.file_uploader("Upload a side-profile image", type=['png', 'jpg', 'jpeg', 'webp'])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Analyzing image..."):
        annotated_img, result_json = process_image_flow(image, uploaded_file.name)

    st.header("Armrest Assessment - "+classify_armrest_height(result_json))
    st.image(annotated_img, use_column_width=True)

    st.header("Detection JSON")
    st.json(result_json)
    

else:
    st.info("Please upload an image to start analysis.")