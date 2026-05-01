import os
import mlflow
import random
import numpy as np
import torch
from ultralytics import YOLO
from dotenv import load_dotenv

load_dotenv()


mlflow.set_tracking_uri("file:///content/mlruns")
mlflow.set_experiment("SilkGuard")

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def train_model():
    set_seed()
    
    # 🔥 DYNAMIC PATH: Works on Windows (D:/) and Colab (/content/)
    # It will save inside your project folder under 'mlruns'
    project_root = os.getcwd()
    mlflow_path = f"file:///{project_root}/mlruns"
    mlflow.set_tracking_uri(mlflow_path)

    mlflow.set_experiment("SilkGuard_Disease_Detection")

    with mlflow.start_run():
        model = YOLO("yolov8s.pt")

        params = {
            "epochs": 50,
            "imgsz": 640,
            "batch": 16,
            "patience": 10,
            "lr0": 0.01,
            "optimizer": "AdamW",
            "model": "yolov8s"
        }
        mlflow.log_params(params)

        print("🚀 Training starting...")
        results = model.train(
            data="config/data.yaml",
            epochs=params["epochs"],
            imgsz=params["imgsz"],
            batch=params["batch"],
            patience=params["patience"],
            lr0=params["lr0"],
            optimizer=params["optimizer"],
            # 🔥 Augmentation
            hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
            degrees=10, translate=0.1, scale=0.5, fliplr=0.5
        )

        print("📊 Logging metrics...")
        try:
            # YOLOv8 stores results in a results_dict
            metrics = results.results_dict
            mlflow.log_metrics({
                "precision": metrics.get("metrics/precision(B)", 0),
                "recall": metrics.get("metrics/recall(B)", 0),
                "mAP50": metrics.get("metrics/mAP50(B)", 0),
                "mAP50-95": metrics.get("metrics/mAP50-95(B)", 0)
            })
        except Exception as e:
            print("⚠️ Could not log metrics:", e)

        # 6. Log Model + Artifacts
        # Note: YOLOv8 might create 'runs/detect/train2', 'train3', etc.
        # This points to the latest 'train' folder
        base_path = "runs/detect/train"

        if os.path.exists(f"{base_path}/weights/best.pt"):
            mlflow.log_artifact(f"{base_path}/weights/best.pt")

        for file in ["results.png", "confusion_matrix.png", "labels.jpg"]:
            file_path = f"{base_path}/{file}"
            if os.path.exists(file_path):
                mlflow.log_artifact(file_path)

        print(f"✅ Training complete. Everything logged to: {mlflow_path}")

if __name__ == "__main__":
    train_model()