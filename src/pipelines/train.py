import os
import mlflow
import random
import numpy as np
import torch
from ultralytics import YOLO, settings
from dotenv import load_dotenv


settings.update({"mlflow": False})

load_dotenv()

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def train_model():
    set_seed()

    # 🔹 Dynamic MLflow path (works in Colab + local)
    project_root = os.getcwd()
    mlflow_path = f"file:///{project_root}/mlruns"
    mlflow.set_tracking_uri(mlflow_path)

    mlflow.set_experiment("SilkGuard_Disease_Detection")

    with mlflow.start_run():

        model = YOLO("yolov8s.pt")

        # 🔹 Parameters
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
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=10,
            translate=0.1,
            scale=0.5,
            fliplr=0.5
        )

        print("📊 Logging metrics...")

        # 🔹 Safe metric logging
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

        # 🔹 Correct save directory (VERY IMPORTANT FIX)
        save_dir = results.save_dir

        print(f"📂 Saving artifacts from: {save_dir}")

        # 🔹 Log best model
        best_model_path = os.path.join(save_dir, "weights", "bestimport os
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

def train_model(resume=False):
    set_seed()

    # 🔥 STEP 1: Mount Google Drive (IMPORTANT)
    from google.colab import drive
    drive.mount('/content/drive')

    # 🔥 STEP 2: Persistent project directory
    BASE_DIR = "/content/drive/MyDrive/SilkGuard"
    os.makedirs(BASE_DIR, exist_ok=True)

    # 🔹 MLflow persistent storage
    mlflow_path = f"file://{BASE_DIR}/mlruns"
    mlflow.set_tracking_uri(mlflow_path)
    mlflow.set_experiment("SilkGuard_Disease_Detection")

    with mlflow.start_run():

        # 🔥 STEP 3: Load model
        if resume:
            print("🔁 Resuming from checkpoint...")
            model_path = os.path.join(BASE_DIR, "last.pt")
            model = YOLO(model_path)
        else:
            model = YOLO("yolov8s.pt")

        # 🔹 Parameters
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

            project=BASE_DIR,     # 🔥 VERY IMPORTANT
            name="yolo_run",      # saved in Drive
            exist_ok=True,

            # 🔥 Save frequently (checkpoint safety)
            save=True,
            save_period=5,        # save every 5 epochs

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

        # 🔹 Safe metric logging
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

        # 🔥 STEP 4: Correct save directory
        save_dir = results.save_dir
        print(f"📂 Saving artifacts from: {save_dir}")

        # 🔹 Save best + last model for resume
        best_model_path = os.path.join(save_dir, "weights", "best.pt")
        last_model_path = os.path.join(save_dir, "weights", "last.pt")

        if os.path.exists(best_model_path):
            mlflow.log_artifact(best_model_path)
            os.system(f"cp {best_model_path} {BASE_DIR}/best.pt")

        if os.path.exists(last_model_path):
            os.system(f"cp {last_model_path} {BASE_DIR}/last.pt")

        # 🔹 Log useful plots
        for file in ["results.png", "confusion_matrix.png", "labels.jpg"]:
            file_path = os.path.join(save_dir, file)
            if os.path.exists(file_path):
                mlflow.log_artifact(file_path)

        print(f"✅ Training complete. Logged to: {mlflow_path}")


if __name__ == "__main__":
    train_model(resume=False)  # 🔁 change to True if restarting.pt")
        if os.path.exists(best_model_path):
            mlflow.log_artifact(best_model_path)

        # 🔹 Log useful plots
        for file in ["results.png", "confusion_matrix.png", "labels.jpg"]:
            file_path = os.path.join(save_dir, file)
            if os.path.exists(file_path):
                mlflow.log_artifact(file_path)

        print(f"✅ Training complete. Logged to: {mlflow_path}")

if __name__ == "__main__":
    train_model()