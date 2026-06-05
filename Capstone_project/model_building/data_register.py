
from huggingface_hub.utils import RepositoryNotFoundError
from huggingface_hub import HfApi, create_repo
import os

repo_id = "chaitram/Engine-Failure-Prediction"
repo_type = "dataset"

# FIXED PATH (Colab-friendly)
folder_path = "Capstone_project/data"

# Debug checks
print("Folder exists:", os.path.exists(folder_path))
if os.path.exists(folder_path):
    print("Files in folder:", os.listdir(folder_path))
else:
    print("Folder not found. Check path!")

print("Token present:", os.getenv("HF_TOKEN") is not None)

# Initialize API client
api = HfApi(token=os.getenv("HF_TOKEN"))

# Step 1: Check if dataset exists
try:
    api.repo_info(repo_id=repo_id, repo_type=repo_type)
    print(f"Dataset '{repo_id}' already exists. Using it.")
except RepositoryNotFoundError:
    print(f"Dataset '{repo_id}' not found. Creating new dataset...")
    create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
    print(f"Dataset '{repo_id}' created.")

# Step 2: Upload folder
try:
    api.upload_folder(
        folder_path=folder_path,
        repo_id=repo_id,
        repo_type=repo_type,
    )
    print("✅ Upload successful!")
except Exception as e:
    print("Upload failed:", e)
