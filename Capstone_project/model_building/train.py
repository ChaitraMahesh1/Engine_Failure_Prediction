# for data manipulation
import pandas as pd

# for class imbalance and confusion matrix
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, balanced_accuracy_score
import matplotlib.pyplot as plt

# preprocessing
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline

# model training
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

# model serialization
import joblib

# hugging face
from huggingface_hub import login, HfApi, create_repo
from huggingface_hub.utils import RepositoryNotFoundError

# mlflow
import mlflow

import os

# ---------------- MLflow Setup ----------------
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("capstone-training-experiment")

# ---------------- Hugging Face Auth ----------------
HF_TOKEN = os.environ.get("HF_TOKEN")
if HF_TOKEN:
    login(token=HF_TOKEN)

api = HfApi()

# ---------------- Data Paths ----------------
Xtrain_path = "hf://datasets/chaitram/Engine-Failure-Prediction/data/Xtrain.csv"
Xtest_path  = "hf://datasets/chaitram/Engine-Failure-Prediction/data/Xtest.csv"
ytrain_path = "hf://datasets/chaitram/Engine-Failure-Prediction/data/ytrain.csv"
ytest_path  = "hf://datasets/chaitram/Engine-Failure-Prediction/data/ytest.csv"

# ---------------- Load Data ----------------
Xtrain = pd.read_csv(Xtrain_path)
Xtest = pd.read_csv(Xtest_path)
ytrain = pd.read_csv(ytrain_path).squeeze()
ytest = pd.read_csv(ytest_path).squeeze()

# ---------------- Feature Separation ----------------
categorical_features = Xtrain.select_dtypes(include=['object']).columns.tolist()
numeric_features = Xtrain.select_dtypes(exclude=['object']).columns.tolist()

# ---------------- Class Imbalance Handling ----------------
sm = SMOTE(random_state=42)
Xtrain, ytrain = sm.fit_resample(Xtrain, ytrain)
print("After SMOTE:", ytrain.value_counts())

# ---------------- Preprocessing ----------------
preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown='ignore'), categorical_features)
)

# ---------------- Model ----------------
xgb_model = xgb.XGBClassifier(random_state=42)

# ---------------- Pipeline ----------------
model_pipeline = make_pipeline(preprocessor, xgb_model)

# ---------------- Hyperparameter Grid ----------------
param_grid = {
    'xgbclassifier__n_estimators': [50, 75, 100],
    'xgbclassifier__max_depth': [2, 3, 4],
    'xgbclassifier__colsample_bytree': [0.4, 0.5, 0.6],
    'xgbclassifier__colsample_bylevel': [0.4, 0.5, 0.6],
    'xgbclassifier__learning_rate': [0.01, 0.05, 0.1],
    'xgbclassifier__reg_lambda': [0.4, 0.5, 0.6],
}

# ---------------- Training with MLflow ----------------
with mlflow.start_run():

    grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, n_jobs=-1, scoring='f1_macro')
    grid_search.fit(Xtrain, ytrain)

    # Log all parameter combinations
    results = grid_search.cv_results_
    for i in range(len(results['params'])):
        with mlflow.start_run(nested=True):
            mlflow.log_params(results['params'][i])
            mlflow.log_metric("mean_test_score", results['mean_test_score'][i])
            mlflow.log_metric("std_test_score", results['std_test_score'][i])

    # Log best params
    mlflow.log_params(grid_search.best_params_)

    best_model = grid_search.best_estimator_

    # Threshold tuning
    threshold = 0.45

    y_pred_train = (best_model.predict_proba(Xtrain)[:, 1] >= threshold).astype(int)
    y_pred_test = (best_model.predict_proba(Xtest)[:, 1] >= threshold).astype(int)

    train_report = classification_report(ytrain, y_pred_train, output_dict=True)
    test_report = classification_report(ytest, y_pred_test, output_dict=True)

    # Consusion Matrix

    # Train CM
    train_cm = confusion_matrix(ytrain, y_pred_train)

    plt.figure(figsize=(5,4))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=train_cm,
        display_labels=["Healthy (0)", "Faulty (1)"]
    )

    disp.plot(cmap='Blues')
    plt.title("Train Confusion Matrix")
    plt.savefig("train_confusion_matrix.png")
    plt.close()

    # Test CM
    test_cm = confusion_matrix(ytest, y_pred_test)

    plt.figure(figsize=(5,4))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=test_cm,
        display_labels=["Healthy (0)", "Faulty (1)"]
    )
    disp.plot(cmap='Blues')
    plt.title("Test Confusion Matrix")
    plt.savefig("test_confusion_matrix.png")
    plt.close()

    # Log metrics
    mlflow.log_metrics({
        "train_accuracy": train_report['accuracy'],
        "train_precision": train_report['1']['precision'],
        "train_recall": train_report['1']['recall'],
        "train_f1": train_report['1']['f1-score'],
        "test_accuracy": test_report['accuracy'],
        "test_precision": test_report['1']['precision'],
        "test_recall": test_report['1']['recall'],
        "test_f1": test_report['1']['f1-score'],
        "train_healthy_recall" : train_report['0']['recall'],
        "train_healthy_f1"     : train_report['0']['f1-score'],
        "test_healthy_recall"  : test_report['0']['recall'],
        "test_healthy_f1"      : test_report['0']['f1-score'],
        "test_macro_f1"        : test_report['macro avg']['f1-score'],
        "train_macro_f1"       : train_report['macro avg']['f1-score'],
    })

    # Log confusion matrices to MLflow (ADDED)
    mlflow.log_artifact("train_confusion_matrix.png")
    mlflow.log_artifact("test_confusion_matrix.png")

    # ---------------- Save Model ----------------
    model_path = "best_engine_failure_model_v1.joblib"
    joblib.dump(best_model, model_path)

    mlflow.log_artifact(model_path, artifact_path="model")

    print(f"Model saved locally at: {model_path}")

    # ---------------- Best Model Results ----------------

    print("\n Best Model Parameters:")
    print(grid_search.best_params_)

    print("\n Train Recall (Faulty):",
        round(train_report['1']['recall'], 4))
    print(" Test Recall (Faulty):",
        round(test_report['1']['recall'], 4))

    print("\n Train Accuracy:",
        round(train_report['accuracy'], 4))
    print(" Test Accuracy:",
        round(test_report['accuracy'], 4))

    print(" Balanced Accuracy (Train):",
        round(balanced_accuracy_score(ytrain, y_pred_train), 4))
    print(" Macro F1 (Train):",
        round(train_report['macro avg']['f1-score'], 4))
    print("\n Balanced Accuracy (Test):",
        round(balanced_accuracy_score(ytest, y_pred_test), 4))
    print(" Macro F1 (Test):",
        round(test_report['macro avg']['f1-score'], 4))
    print(" Healthy Recall (Test):",
        round(test_report['0']['recall'], 4))


    # ---------------- Hugging Face Upload ----------------
    repo_id = "chaitram/Engine-Failure-Prediction"
    repo_type = "model"

    try:
        api.repo_info(repo_id=repo_id, repo_type=repo_type)
        print(f"Repo '{repo_id}' already exists.")
    except RepositoryNotFoundError:
        print(f"Creating repo '{repo_id}'...")
        create_repo(repo_id=repo_id, repo_type=repo_type, private=False)

    api.upload_file(
        path_or_fileobj=model_path,
        path_in_repo=model_path,
        repo_id=repo_id,
        repo_type=repo_type
    )

    print("Model uploaded successfully to Hugging Face 🚀")
