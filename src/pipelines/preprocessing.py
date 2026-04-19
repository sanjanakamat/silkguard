import os
import cv2
import shutil

def preprocess_data():
    raw_path = "data/raw/train/images"
    proc_path = "data/processed/train/images"
    os.makedirs(proc_path, exist_ok=True)

    print("🛠️ Preprocessing: Standardizing images to 640x640...")
    for img_name in os.listdir(raw_path):
        if img_name.endswith(('.jpg', '.png')):
            img = cv2.imread(os.path.join(raw_path, img_name))
            # Heavy lifting happens here
            resized = cv2.resize(img, (640, 640))
            cv2.imwrite(os.path.join(proc_path, img_name), resized)
    print("✅ Preprocessing Complete: Clean data is in data/processed")

if __name__ == "__main__":
    preprocess_data()