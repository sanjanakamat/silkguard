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

def train_model(resume=False):
    set_seed()

    try:
        import google.colab
        in_colab = True
    except ImportError:
        in_colab = False

    if in_colab:
        print("detected google colab. mounting google drive...")
        from google.colab import drive
        drive.mount('/content/drive')
        BASE_DIR = "/content/drive/MyDrive/SilkGuard"
    else:
        BASE_DIR = os.getcwd()

    os.makedirs(BASE_DIR, exist_ok=True)

    mlflow_path = f"file:///{BASE_DIR}/mlruns".replace("\\", "/")
    mlflow.set_tracking_uri(mlflow_path)
    mlflow.set_experiment("SilkGuard_Disease_Detection")

    with mlflow.start_run():
        if resume:
            print("resuming from checkpoint...")
            model_path = os.path.join(BASE_DIR, "yolo_run", "weights", "last.pt")
            model = YOLO(model_path)
        else:
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
        print("training starting...")

        results = model.train(
            data="config/data.yaml",
            epochs=params["epochs"],
            imgsz=params["imgsz"],
            batch=params["batch"],
            patience=params["patience"],
            lr0=params["lr0"],
            optimizer=params["optimizer"],
            project=BASE_DIR,     
            name="yolo_run",      
            exist_ok=True,
            save=True,
            save_period=5,        
            hsv_h=0.015,
            hsv_s=0.7,
            hsv_v=0.4,
            degrees=10,
            translate=0.1,
            scale=0.5,
            fliplr=0.5
        )

        print("logging metrics...")
        try:
            metrics = results.results_dict
            mlflow.log_metrics({
                "precision": metrics.get("metrics/precision(B)", 0),
                "recall": metrics.get("metrics/recall(B)", 0),
                "mAP50": metrics.get("metrics/mAP50(B)", 0),
                "mAP50-95": metrics.get("metrics/mAP50-95(B)", 0)
            })
        except Exception as e:
            print("metric logging failed:", e)

        save_dir = results.save_dir
        print(f"saving artifacts from: {save_dir}")

        best_model_path = os.path.join(save_dir, "weights", "best.pt")
        last_model_path = os.path.join(save_dir, "weights", "last.pt")

        if os.path.exists(best_model_path):
            mlflow.log_artifact(best_model_path)
            if in_colab:
                os.system(f"cp {best_model_path} {BASE_DIR}/best.pt")

        if os.path.exists(last_model_path) and in_colab:
            os.system(f"cp {last_model_path} {BASE_DIR}/last.pt")

        for file in ["results.png", "confusion_matrix.png", "labels.jpg"]:
            file_path = os.path.join(save_dir, file)
            if os.path.exists(file_path):
                mlflow.log_artifact(file_path)

        print(f"training complete. logged to: {mlflow_path}")

if __name__ == "__main__":
    train_model(resume=False)