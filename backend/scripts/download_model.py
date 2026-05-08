# Location: backend/scripts/download_model.py
import os
from huggingface_hub import snapshot_download

# Use your actual Repo ID here
REPO_ID = "mhass909/medgemma-medical-finetuned" 

# We point to the storage folder relative to this script
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCAL_DIR = os.path.join(BASE_DIR, "storage", "medgemma_offline")

def run_download():
    print(f"Targeting directory: {LOCAL_DIR}")
    snapshot_download(
        repo_id=REPO_ID,
        local_dir=LOCAL_DIR,
        local_dir_use_symlinks=False
    )
    print("✅ Model downloaded successfully to storage/medgemma_offline")

if __name__ == "__main__":
    run_download()