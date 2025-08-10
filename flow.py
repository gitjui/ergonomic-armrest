import cv2
import numpy as np
from PIL import Image
import os
from env_analysis import analyze_environment
from arm_detection import detect_arm_landmarks, detect_armrest_and_annotate
from image_handler import save_intermediate_image
from image_handler import  clean_intermediate_dir



def process_image_flow(pil_image, original_filename):
    clean_intermediate_dir()
    frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

    # Step 1: Environment detection (person, chair, desk)
    env_annotated, env_json = analyze_environment(frame, original_filename)

    # Save environment annotated image
    save_intermediate_image(env_annotated, original_filename, "env_annotated")

    # Step 2: Detect arm landmarks using mediapipe
    landmarks = detect_arm_landmarks(frame, side='right')
    if not landmarks:
        return env_annotated, {**env_json, "arm_landmarks_detected": False}

    # Step 3: Detect armrest and annotate over environment annotated image
    armrest_annotated, armrest_box, desk_y = detect_armrest_and_annotate(env_annotated, landmarks, original_filename,
                                                                          isDesk=env_json.get("isDesk", False), 
                                                                          isChair=env_json.get("isChair", True))

    # Save armrest annotated image
    save_intermediate_image(armrest_annotated, original_filename, "armrest_annotated")

    result_json = {
        **env_json,
        "arm_landmarks_detected": True,
        "landmarks": landmarks,
        "armrest_box": armrest_box,
        "desk_y" : desk_y
    }

    annotated_pil = Image.fromarray(cv2.cvtColor(armrest_annotated, cv2.COLOR_BGR2RGB))

    return annotated_pil, result_json