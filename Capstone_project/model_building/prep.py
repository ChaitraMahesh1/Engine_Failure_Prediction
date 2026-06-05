# for data manipulation
import pandas as pd
import sklearn
# for creating a folder
import os
# for data preprocessing and pipeline creation
from sklearn.model_selection import train_test_split
# for converting text data in to numerical representation
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi

# Load Dataset from Hugging Face
api = HfApi(token=os.getenv("HF_TOKEN"))
DATASET_PATH = "hf://datasets/chaitram/Engine-Failure-Prediction/engine_data.csv"
df = pd.read_csv(DATASET_PATH)
print("✅ Dataset loaded successfully.")
print("Shape:", df.shape)

# Drop Unnecessary Columns
# No unique ID or categorical columns to drop
# Only drop unnamed index column if present
df.drop(columns=['Unnamed: 0'], inplace=True, errors='ignore')
print("✅ Unnecessary columns dropped.")
print("Shape after dropping:", df.shape)

# Handle Outliers using IQR Method
num_cols = [col for col in df.columns if col != 'Engine Condition']

for col in num_cols:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    # Cap outliers instead of dropping
    df[col] = df[col].clip(lower=lower, upper=upper)

print("✅ Outliers handled using IQR capping.")
print("Shape after outlier treatment:", df.shape)

# Split into X (features) and y (target)
target_col = 'Engine Condition'

X = df.drop(columns=[target_col])
y = df[target_col]

print(f"\nClass distribution before SMOTE:\n{y.value_counts()}")

#Handle Class Imbalance using SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

print(f"\n✅ Class distribution after SMOTE:\n{pd.Series(y_resampled).value_counts()}")

# Perform train-test split
Xtrain, Xtest, ytrain, ytest = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42
)

print(f"\nXtrain shape: {Xtrain.shape}")
print(f"Xtest shape : {Xtest.shape}")

#Save Files Locally
os.makedirs("Capstone_project/data", exist_ok=True)

Xtrain.to_csv("Capstone_project/data/Xtrain.csv", index=False)
Xtest.to_csv("Capstone_project/data/Xtest.csv", index=False)
ytrain.to_csv("Capstone_project/data/ytrain.csv", index=False)
ytest.to_csv("Capstone_project/data/ytest.csv", index=False)

print("✅ Files saved locally.")

#Upload Files to Hugging Face

files = [
    "Capstone_project/data/Xtrain.csv",
    "Capstone_project/data/Xtest.csv",
    "Capstone_project/data/ytrain.csv",
    "Capstone_project/data/ytest.csv"
]

for file_path in files:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=f"data/{file_path.split('/')[-1]}",
        repo_id="chaitram/Engine-Failure-Prediction",
        repo_type="dataset",
        commit_message=f"Uploading {file_path.split('/')[-1]}",
        create_pr=False
    )
    print(f"✅ Uploaded: {file_path.split('/')[-1]}")

print("\n✅ All files uploaded successfully to HuggingFace.")

import time
time.sleep(60)
