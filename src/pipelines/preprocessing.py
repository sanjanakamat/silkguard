import os
import cv2
import shutil

def preprocess_all_splits():
    # Define our folders
    splits = ['train', 'valid', 'test']
    raw_base = "data/raw"
    proc_base = "data/processed"

    for split in splits:
        raw_path = os.path.join(raw_base, split, "images")
        proc_path = os.path.join(proc_base, split, "images")
        
        # Check if the raw folder exists before trying to process it
        if not os.path.exists(raw_path):
            print(f"⚠️ Skipping {split}: Folder not found.")
            continue

        os.makedirs(proc_path, exist_ok=True)
        print(f"🛠️ Preprocessing {split} images...")

        for img_name in os.listdir(raw_path):
            if img_name.lower().endswith(('.jpg', '.png', '.jpeg')):
                img = cv2.imread(os.path.join(raw_path, img_name))
                
                if img is not None:
                    # The exact same transformation for every single image
                    resized = cv2.resize(img, (640, 640))
                    cv2.imwrite(os.path.join(proc_path, img_name), resized)
                
        # Pro-Tip: We also need to copy the labels (text files) to the processed folder
        # YOLOv8 needs the .txt labels to be in a 'labels' folder next to 'images'
        raw_labels = os.path.join(raw_base, split, "labels")
        proc_labels = os.path.join(proc_base, split, "labels")
        if os.path.exists(raw_labels):
            if os.path.exists(proc_labels):
                shutil.rmtree(proc_labels)
            shutil.copytree(raw_labels, proc_labels)

    print("✅ All splits (Train/Val/Test) are now standardized and ready!")

if __name__ == "__main__":
    preprocess_all_splits()