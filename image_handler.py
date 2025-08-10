import os
import cv2
import shutil

INTERMEDIATE_DIR = "intermediate_images"
os.makedirs(INTERMEDIATE_DIR, exist_ok=True)

def clean_intermediate_dir():
    if not os.path.exists(INTERMEDIATE_DIR):
        os.makedirs(INTERMEDIATE_DIR)
        return

    for item in os.listdir(INTERMEDIATE_DIR):
        item_path = os.path.join(INTERMEDIATE_DIR, item)
        if os.path.isfile(item_path) or os.path.islink(item_path):
            os.remove(item_path)
        else:
            shutil.rmtree(item_path)

def save_intermediate_image(image, base_name, suffix):
    filename = f"{os.path.splitext(base_name)[0]}_{suffix}.png"
    path = os.path.join(INTERMEDIATE_DIR, filename)
    cv2.imwrite(path, image)
    return path