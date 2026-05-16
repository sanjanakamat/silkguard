from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from ultralytics import YOLO
import shutil
import os

app = FastAPI(title="SilkGuard API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")

# load weights
try:
    model_path = "runs/detect/train-2/weights/best.pt"
    if not os.path.exists(model_path):
        print(f"model weights not found at {model_path}")
        model = None
    else:
        model = YOLO(model_path)
except Exception as e:
    print(f"failed to load model: {e}")
    model = None

@app.get("/")
def home():
    return RedirectResponse(url="/frontend/index.html")

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    if model is None:
        return {"error": "model not loaded"}

    temp_path = f"temp_{file.filename}"
    
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        # run inference
        results = model.predict(source=temp_path, conf=0.25)

        predictions = []
        for r in results:
            for box in r.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                
                predictions.append({
                    "class": model.names[cls],
                    "confidence": round(conf, 3)
                })
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    return {"predictions": predictions}