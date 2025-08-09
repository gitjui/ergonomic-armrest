import streamlit as st
import cv2
import numpy as np
import base64
import json
from ultralytics import YOLO
from PIL import Image

# ==============================
# CONSTANTS
# ==============================
MODEL_PATH = "yolov8n.pt"
IMG_SIZE = 320
CONF_THRESHOLD = 0.4

DRAW_CLASSES = {"person", "chair", "desk"}
DESK_ALTERNATES = {"laptop", "mouse", "bottle", "desktop"}
BOX_COLOR = (170, 200, 0)
HEADER_COLOR_PASS = (255, 40, 0)
HEADER_COLOR_FAIL = (0, 0, 255)

# ==============================
# MODEL LOAD
# ==============================
model = YOLO(MODEL_PATH)
CLASS_NAMES = model.names


# ==============================
# DETECTION FUNCTIONS
# ==============================
def run_precheck(frame):
    """Run YOLO model on frame and return detection status, missing items, boxes, and labels."""
    results = model.predict(frame, imgsz=IMG_SIZE, conf=CONF_THRESHOLD, verbose=False)[0]
    detected_labels, filtered_boxes = set(), []

    for box in results.boxes:
        cls_id = int(box.cls[0])
        label = CLASS_NAMES[cls_id]

        if label in DESK_ALTERNATES:
            detected_labels.add("desk")

        if label in DRAW_CLASSES:
            detected_labels.add(label)
            filtered_boxes.append((box.xyxy[0], label, box.conf[0]))

    pass_conditions = [
        {"person", "desk"},
        {"person", "chair"},
        {"person", "desk", "chair"}
    ]
    status = any(pc.issubset(detected_labels) for pc in pass_conditions)
    missing = None if status else DRAW_CLASSES - detected_labels

    return status, missing, frame, filtered_boxes, detected_labels


def get_posture(detected_labels):
    if "person" not in detected_labels:
        return "Unknown"
    return "Sitting" if "chair" in detected_labels else "Standing"


# ==============================
# ANNOTATION FUNCTIONS
# ==============================
def draw_filtered_boxes(frame, boxes):
    for (coords, label, conf) in boxes:
        x1, y1, x2, y2 = map(int, coords)
        cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, 2)
        cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, BOX_COLOR, 2)
    return frame


def add_header_info(frame, detected_labels, status, missing):
    posture = get_posture(detected_labels)
    if status:
        detected_str = ", ".join(sorted(detected_labels))
        header_text = f"Setup detected {detected_str} | Posture: {posture}"
        color = HEADER_COLOR_PASS
    else:
        header_text = f"Missing: {', '.join(missing)}"
        color = HEADER_COLOR_FAIL

    cv2.putText(frame, header_text, (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    return frame, posture


# ==============================
# OUTPUT FUNCTIONS
# ==============================
def build_json(detected_labels, posture):
    return {
        "isChair": "chair" in detected_labels,
        "isDesk": "desk" in detected_labels,
        "isPerson": "person" in detected_labels,
        "isSitting": posture.lower() == "sitting",
        "isStanding": posture.lower() == "standing",
    }


def setupAnalysis(pil_image):
    frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
    status, missing, frame, filtered_boxes, detected_labels = run_precheck(frame)
    frame = draw_filtered_boxes(frame, filtered_boxes)
    frame, posture = add_header_info(frame, detected_labels, status, missing)
    result_json = build_json(detected_labels, posture)
    return frame, result_json


# ==============================
# STREAMLIT APP
# ==============================
st.set_page_config(page_title="Armrest Height Classification", layout="centered")
st.title("Ergonomic Armrest Height Classifier")
st.markdown("Upload a side-profile image of a person working at their desk")

uploaded_file = st.file_uploader("Upload a side-profile image", type=['png', 'jpg', 'jpeg', 'webp'])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Analyzing image..."):
        annotated_frame, debug_json = setupAnalysis(image)

    # Convert OpenCV BGR to RGB for display
    annotated_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
    st.header("Annotated Image")
    st.image(annotated_rgb, use_column_width=True)

    st.header("Detection JSON")
    st.json(debug_json)
else:
    st.info("Please upload an image to start analysis.")