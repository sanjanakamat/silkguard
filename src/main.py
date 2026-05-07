from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from ultralytics import YOLO
import shutil
import os

app = FastAPI()

# 🔹 Add CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🔹 Serve Frontend Files
app.mount("/frontend", StaticFiles(directory="frontend"), name="frontend")

# 🔹 Load your trained model
model = YOLO("runs/detect/train-2/weights/best.pt")

@app.get("/")
def home():
    return RedirectResponse(url="/frontend/index.html")

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    
    # 🔹 Save uploaded file temporarily
    temp_path = f"temp_{file.filename}"
    
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 🔹 Run prediction
    results = model.predict(source=temp_path, conf=0.25)

    # 🔹 Extract results
    predictions = []
    
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            
            predictions.append({
                "class": model.names[cls],
                "confidence": round(conf, 3)
            })

    # 🔹 Remove temp file
    os.remove(temp_path)

    return {"predictions": predictions}