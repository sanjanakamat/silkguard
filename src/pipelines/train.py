import os
import mlflow
import random
import numpy as np
import torch
from ultralytics import YOLO
from dotenv import load_dotenv


load_dotenv()

# 🔁 Reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def train_model():
    set_seed()
    mlflow.set_tracking_uri("file:///D:/silkguard/runs/mlflow")

    # 1. MLflow Experiment
    mlflow.set_experiment("SilkGuard_Disease_Detection")

    with mlflow.start_run():

        # 2. Load Model (Better than nano)
        model = YOLO("yolov8s.pt")

        # 3. Parameters
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

        # 4. Training
        print("🚀 Training starting...")

        results = model.train(
            data="config/data.yaml",   # <-- dataset defined here
            epochs=params["epochs"],
            imgsz=params["imgsz"],
            batch=params["batch"],
            patience=params["patience"],
            lr0=params["lr0"],
            optimizer=params["optimizer"],

            # 🔥 Augmentation
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=10,
            translate=0.1,
            scale=0.5,
            fliplr=0.5
        )

        print("📊 Logging metrics...")

        # 5. Extract Metrics
        try:
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
        base_path = "runs/detect/train"

        mlflow.log_artifact(f"{base_path}/weights/best.pt")

        # Optional but useful
        for file in ["results.png", "confusion_matrix.png", "labels.jpg"]:
            file_path = f"{base_path}/{file}"
            if os.path.exists(file_path):
                mlflow.log_artifact(file_path)

        print("✅ Training complete. Everything logged to MLflow.")

if __name__ == "__main__":
    train_model()