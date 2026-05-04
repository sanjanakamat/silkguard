import os
import mlflow
import random
import numpy as np
import torch
from ultralytics import YOLO, settings
from dotenv import load_dotenv

# 🔹 Disable Ultralytics MLflow conflict
settings.update({"mlflow": False})

load_dotenv()

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def train_model():
    set_seed()

    # 🔹 MLflow path (local)
    project_root = os.getcwd()
    mlflow_path = f"file:///{project_root}/mlruns"
    mlflow.set_tracking_uri(mlflow_path)

    mlflow.set_experiment("SilkGuard_Disease_Detection")

    with mlflow.start_run():

        # 🔹 Load model
        model = YOLO("yolov8s.pt")

        # 🔹 Parameters
        params = {
            "epochs": 2,
            "imgsz": 640,
            "batch": 16,
            "patience": 10,
            "lr0": 0.01,
            "optimizer": "AdamW",
            "model": "yolov8s"
        }

        mlflow.log_params(params)

        print("🚀 Training starting...")

        # 🔹 Train model
        results = model.train(
            data="config/data.yaml",
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

        # 🔹 Log metrics safely
        try:
            metrics = results.results_dict

            mlflow.log_metrics({
                "precision": metrics.get("metrics/precision(B)", 0),
                "recall": metrics.get("metrics/recall(B)", 0),
                "mAP50": metrics.get("metrics/mAP50(B)", 0),
                "mAP50-95": metrics.get("metrics/mAP50-95(B)", 0)
            })
        except Exception as e:
            print("⚠️ Metric logging failed:", e)

        # 🔹 Save directory
        save_dir = results.save_dir
        print(f"📂 Saving artifacts from: {save_dir}")

        # 🔹 Log best model
        best_model_path = os.path.join(save_dir, "weights", "best.pt")

        if os.path.exists(best_model_path):
            mlflow.log_artifact(best_model_path)

        # 🔹 Log plots
        for file in ["results.png", "confusion_matrix.png", "labels.jpg"]:
            file_path = os.path.join(save_dir, file)
            if os.path.exists(file_path):
                mlflow.log_artifact(file_path)

        print(f"✅ Training complete. Logged to: {mlflow_path}")


if __name__ == "__main__":
    train_model()