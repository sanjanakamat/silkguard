import os
from dotenv import load_dotenv
from roboflow import Roboflow

load_dotenv()

def ingest_data():
    print("starting data ingestion from roboflow...")
    api_key = os.getenv("ROBOFLOW_API_KEY")
    if not api_key:
        raise ValueError("ROBOFLOW_API_KEY is missing")

    rf = Roboflow(api_key=api_key)
    
    project = rf.workspace("sanjanas-workspace-wscv9").project("silkworm-data")
    version = project.version(1)

    dataset = version.download("yolov8", location="data/raw", overwrite=True)
    print(f"data downloaded to: {dataset.location}")

if __name__ == "__main__":
    ingest_data()