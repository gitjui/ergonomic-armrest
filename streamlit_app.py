import streamlit as st
import re
import matplotlib.pyplot as plt
from PIL import Image
import os
from flow import process_image_flow
from classify import classify_armrest_height

st.set_page_config(page_title="Armrest Height Classification", layout="centered")
st.title("Ergonomic Armrest Height Classifier")
st.markdown("Upload a side-profile image of a person working at their desk.")
st.markdown("Accepted image formats: .png, .jpg, .jpeg, .webp")

uploaded_file = st.file_uploader(
    "Upload a side-profile image", 
    type=['png', 'jpg', 'jpeg', 'webp']
)

if uploaded_file:
    try:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        with st.spinner("Analyzing image..."):
            annotated_img, result_json = process_image_flow(image, uploaded_file.name)

        st.header("Armrest Assessment - " + classify_armrest_height(result_json))
        st.image(annotated_img, use_container_width=True)
        st.header("Detection JSON")
        st.json(result_json)





       
        # New part: show intermediate images grouped by suffix

        intermediate_dir = "intermediate_images"
        all_files = os.listdir(intermediate_dir)

        # Extract suffix pattern and group by it
        # Example filename: s4_cropped_353_442.png
        pattern = re.compile(r"^(?:[^_]+)_([a-z_]+)_(\d+_\d+)\.png$")

        # We want to group by suffix (the x_y part) and then sort by the prefixes in order
        # But the naming has a prefix part before x_y, like 'cropped', 'candidate_canny', etc.
        # Actually the pattern above does not separate the prefix correctly, we want to separate the
        # prefix and suffix properly:
        # Format looks like s4_<prefix>_<x>_<y>.png
        # So let's parse as: s4_<prefix>_<x>_<y>.png

        # Let's do better regex:
        pattern = re.compile(r"^(?:[^_]+)_([a-z_]+)_(\d+)_(\d+)\.png$")

        images_by_suffix = {}

        for fname in all_files:
            match = pattern.match(fname)
            if match:
                prefix = match.group(1)          # e.g. cropped, candidate_canny
                x = match.group(2)
                y = match.group(3)
                suffix = f"{x}_{y}"              # group by this
                if suffix not in images_by_suffix:
                    images_by_suffix[suffix] = {}
                images_by_suffix[suffix][prefix] = os.path.join(intermediate_dir, fname)

        # Order to display prefixes
        display_order = ["cropped", "candidate_canny", "candidate_mask", "candidates"]
        if images_by_suffix and len(images_by_suffix) > 0:
            st.header("Intermediate Results")
            st.markdown("Region of interest processing for above and below elbow")

        # Now display each suffix group in one line with subplots
        for suffix, images_dict in images_by_suffix.items():
            fig, axes = plt.subplots(1, len(display_order), figsize=(15, 4))
            if len(display_order) == 1:
                axes = [axes]  # make iterable if only 1 subplot
            for ax, prefix in zip(axes, display_order):
                ax.axis('off')
                if prefix in images_dict:
                    img_path = images_dict[prefix]
                    img = Image.open(img_path)
                    ax.imshow(img)
                    ax.set_title(prefix)
                else:
                    ax.set_title(prefix + "\n(Not found)")
            st.pyplot(fig)

    except Exception as e:
        st.error(f"Error processing image: {e}")



 



    except Exception as e:
        st.error(f"Error processing image: {e}")

else:
    st.info("Please upload an image to start analysis.")
