import os
from dotenv import load_dotenv
from roboflow import Roboflow

load_dotenv()

def ingest_data():
    api_key = os.getenv("ROBOFLOW_API_KEY")
    print("API KEY:", api_key)   # ✅ debug

    rf = Roboflow(api_key=api_key)

    project = rf.workspace("sanjanas-workspace-wscv9").project("silkworm-data")
    print("Project loaded")     # ✅ debug

    version = project.version(1)
    print("Version loaded")     # ✅ debug

    dataset = version.download("yolov8", location="data/raw",overwrite=True)
    print("Downloaded to:", dataset.location)   # ✅ VERY IMPORTANT

    print("✅ Done")

if __name__ == "__main__":
    ingest_data()