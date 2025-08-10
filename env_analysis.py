import cv2
import numpy as np
from ultralytics import YOLO

# Constants
MODEL_PATH = "yolov8n.pt"
IMG_SIZE = 320
CONF_THRESHOLD = 0.4
DRAW_CLASSES = {"person", "chair", "desk"}
DESK_ALTERNATES = {"laptop", "mouse", "bottle", "desktop", "keyboard", "monitor", "wallet"}
BOX_COLOR = (170, 200, 0)
HEADER_COLOR_PASS = (255, 40, 0)
HEADER_COLOR_FAIL = (0, 0, 255)

# Load model once
model = YOLO(MODEL_PATH)
CLASS_NAMES = model.names

def run_precheck(frame):
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

    return status, missing, filtered_boxes, detected_labels

def get_posture(detected_labels):
    if "person" not in detected_labels:
        return "Unknown"
    return "Sitting" if "chair" in detected_labels else "Standing"

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

def build_json(detected_labels, posture):
    return {
        "isChair": "chair" in detected_labels,
        "isDesk": "desk" in detected_labels,
        "isPerson": "person" in detected_labels,
        "isSitting": posture.lower() == "sitting",
        "isStanding": posture.lower() == "standing",
    }

def analyze_environment(frame, base_name):
    frame_copy = frame.copy()
    status, missing, filtered_boxes, detected_labels = run_precheck(frame_copy)
    frame_copy = draw_filtered_boxes(frame_copy, filtered_boxes)
    frame_copy, posture = add_header_info(frame_copy, detected_labels, status, missing)
    result_json = build_json(detected_labels, posture)
    return frame_copy, result_json
