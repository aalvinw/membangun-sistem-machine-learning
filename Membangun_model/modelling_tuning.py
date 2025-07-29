import pandas as pd
import os
import joblib
import shutil
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    log_loss,
    classification_report,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay,
)
import matplotlib.pyplot as plt

# 1. Koneksi ke DagsHub
os.environ["MLFLOW_TRACKING_USERNAME"] = "aalvinw"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "fd0f43f8991441bdbdfa6807f7e71cf33c2b9082"
mlflow.set_tracking_uri("https://dagshub.com/aalvinw/submission-msml_Agustinusalvinwicaksono.mlflow")

# 2. Load Data
data = pd.read_csv("Eksperimen_SML_AgustinusAlvinWicaksono/preprocessing/E-Commerce_Shipping_preprocessing/preprocessed_data.csv")
X = data.drop(columns=["Reached.on.Time_Y.N"])
y = data["Reached.on.Time_Y.N"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Setup MLflow
mlflow.set_experiment("Advanced_Model_RF")
mlflow.sklearn.autolog()

# 4. Tuning dan Training
params = {"n_estimators": [100, 150], "max_depth": [5, 10]}

with mlflow.start_run(run_name="GridSearch_RF_Advanced"):
    clf = GridSearchCV(RandomForestClassifier(random_state=42), param_grid=params, cv=3)
    clf.fit(X_train, y_train)

    best_model = clf.best_estimator_
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)

    # 5. Manual Logging Tambahan
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    loss = log_loss(y_test, y_proba)
    report = classification_report(y_test, y_pred, output_dict=True)

    mlflow.log_metric("log_loss", loss)
    mlflow.log_metric("precision_macro", report["macro avg"]["precision"])
    mlflow.log_metric("recall_macro", report["macro avg"]["recall"])

    # 6. Simpan Artefak Manual
    with open("classification_report.txt", "w") as f:
        f.write(classification_report(y_test, y_pred))
    mlflow.log_artifact("classification_report.txt")

    # 7. Visualisasi Metrik
    ConfusionMatrixDisplay.from_estimator(best_model, X_test, y_test)
    plt.title("Confusion Matrix")
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")
    plt.close()

    RocCurveDisplay.from_estimator(best_model, X_test, y_test)
    plt.title("ROC Curve")
    plt.savefig("roc_curve.png")
    mlflow.log_artifact("roc_curve.png")
    plt.close()

    PrecisionRecallDisplay.from_estimator(best_model, X_test, y_test)
    plt.title("Precision-Recall Curve")
    plt.savefig("precision_recall_curve.png")
    mlflow.log_artifact("precision_recall_curve.png")
    plt.close()

    # 8. Simpan Model Manual ke folder model/
    os.makedirs("model", exist_ok=True)
    joblib.dump(best_model, "model/model.pkl")

    with open("model/MLmodel", "w") as f:
        f.write("artifact_path: model\nflavors: {sklearn: {}}")

    with open("model/conda.yaml", "w") as f:
        f.write("dependencies:\n  - scikit-learn")

    with open("model/requirements.txt", "w") as f:
        f.write("scikit-learn")

    mlflow.log_artifacts("model", artifact_path="model")

    print("✅ Autolog aktif & folder model/ dilog lengkap ke artifacts")
