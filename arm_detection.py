import cv2
import mediapipe as mp
import numpy as np
from image_handler import save_intermediate_image
mp_pose = mp.solutions.pose

def detect_arm_landmarks(image, side='right'):
    with mp_pose.Pose(static_image_mode=True) as pose:
        results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if not results.pose_landmarks:
            return None

        landmarks = results.pose_landmarks.landmark
        h, w, _ = image.shape

        if side == 'right':
            shoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
            elbow = landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW]
            wrist = landmarks[mp_pose.PoseLandmark.RIGHT_WRIST]
        else:
            shoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
            elbow = landmarks[mp_pose.PoseLandmark.LEFT_ELBOW]
            wrist = landmarks[mp_pose.PoseLandmark.LEFT_WRIST]

        return {
            "shoulder": {"x": int(shoulder.x * w), "y": int(shoulder.y * h)},
            "elbow": {"x": int(elbow.x * w), "y": int(elbow.y * h)},
            "wrist": {"x": int(wrist.x * w), "y": int(wrist.y * h)}
        }

def crop_below_point(image, start_x, start_y, crop_width, crop_height, base_name=None, suffix=None):
    h, w = image.shape[:2]
    x1 = max(start_x - crop_width // 2, 0)
    y1 = start_y
    x2 = min(x1 + crop_width, w)
    y2 = min(y1 + crop_height, h)
    cropped = image[y1:y2, x1:x2]
    if base_name is not None and suffix is not None:
        save_intermediate_image(cropped, base_name, suffix)
    return image[y1:y2, x1:x2], x1, y1

def detect_armrest_and_annotate(image, landmarks, base_name, isDesk=False):
    elbow = landmarks["elbow"]
    wrist = landmarks["wrist"]
    shoulder = landmarks["shoulder"]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def detect_armrest_and_annotate(image, landmarks, base_name, isDesk=False, isChair=True):
    elbow = landmarks["elbow"]
    wrist = landmarks["wrist"]
    shoulder = landmarks["shoulder"]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def detect_in_roi(center_x, center_y, crop_w, crop_h):
        cropped_edges, crop_x1, crop_y1 = crop_below_point(image, center_x, center_y, crop_w, crop_h,
                                                            base_name, f"cropped_{center_x}_{center_y}")
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        blurred = cv2.GaussianBlur(cropped_edges, (7, 7), 0)
        edges_canny = cv2.Canny(blurred, 50, 150)
        contours, _ = cv2.findContours(edges_canny.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        save_intermediate_image(edges_canny, base_name, f"candidate_canny_{center_x}_{center_y}")
        filtered_contours = [cnt for cnt in contours if len(cnt) > 100]
        mask = np.zeros_like(edges_canny)
        cv2.drawContours(mask, filtered_contours, -1, 255, thickness=2)
        save_intermediate_image(mask, base_name, f"candidate_mask_{center_x}_{center_y}")
        lines = cv2.HoughLinesP(mask, rho=1, theta=np.pi/180, threshold=30, minLineLength=100, maxLineGap=10)
        candidates = []
        line_img = 255 * np.ones_like(mask)  
        if lines is not None:
            for x1, y1, x2, y2 in lines[:, 0]:
                cv2.line(line_img, (x1, y1), (x2, y2), (200, 200, 200), 1)  
                slope = (y2 - y1) / (x2 - x1) if (x2 - x1) != 0 else 9999
                if -0.5 < slope < 0.5:  
                    length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                    avg_y = (y1 + y2) / 2
                    candidates.append({
                        "x": crop_x1 + x1,
                        "y": crop_y1 + int(avg_y) - 5,
                        "w": x2 - x1,
                        "h": 20,
                        "score": length  
                    })
                    cv2.line(line_img, (x1, y1), (x2, y2), 0, 2)
        # Construct suffix with center coordinates
        suffix = f"candidates_{center_x}_{center_y}"
        save_intermediate_image(line_img, base_name, suffix)
        return candidates
    annotated = image.copy()
    # Draw arm landmarks
    for joint, coord in landmarks.items():
        cv2.circle(annotated, (coord["x"], coord["y"]), 8, (0, 255, 0), -1)
        cv2.putText(annotated, joint, (coord["x"] + 5, coord["y"] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    best_candidate = {}

    if isChair:    
        # Search below elbow, in case the armrest is below the elbow
        below_candidates = detect_in_roi(elbow["x"], elbow["y"], 300, 150)

        # Search above elbow (Assumption - from elbow up towards shoulder as the armrest will not be above the shoulder)
        above_roi_height = max(50, int(abs(shoulder["y"] - elbow["y"])/2))
        above_candidates = detect_in_roi(elbow["x"], elbow["y"] - above_roi_height, 300, above_roi_height)

        # Combine & vote (above_candidates and below candidates are two dict with score and the maximum is selected)
        all_candidates = below_candidates + above_candidates
        best_candidate = max(all_candidates, key=lambda c: c["score"], default=None)

        
    
        print("details", landmarks)
        print("best candidate", best_candidate)
        # Draw best armrest box
        if best_candidate:
            cv2.rectangle(annotated,
                      (best_candidate["x"], best_candidate["y"]),
                      (best_candidate["x"] + best_candidate["w"], best_candidate["y"] + best_candidate["h"]),
                      (200, 180, 0), 4)
            cv2.putText(annotated, "Armrest", (best_candidate["x"], best_candidate["y"] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 100, 0), 3)

    desk_y = -1
    # Desk annotation (unchanged)
    if isDesk and wrist:
        dx = wrist["x"] - elbow["x"]
        dy = wrist["y"] - elbow["y"]
        arm_length = int(np.sqrt(dx**2 + dy**2))
        desk_offset = int(arm_length / 5)
        desk_x = wrist["x"] - 50
        desk_y = wrist["y"] + desk_offset
        h, w, _ = image.shape
        desk_x = max(0, desk_x)
        desk_w = w - desk_x
        desk_h = 10
        desk_y = min(desk_y, h - desk_h)
        cv2.rectangle(annotated, (desk_x, desk_y), (desk_x + desk_w, desk_y + desk_h), (100, 100, 255), 4)
        cv2.putText(annotated, "Desk Height Annotation", (desk_x + 5, desk_y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 255), 2)

    return annotated, best_candidate, desk_y
