import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import mlflow
import mlflow.sklearn
import joblib
import os

# 1. Load preprocessed data
df = pd.read_csv("Eksperimen_SML_AgustinusAlvinWicaksono\\preprocessing\\E-Commerce_Shipping_preprocessing\\preprocessed_data.csv")

X = df.drop(columns=["Reached.on.Time_Y.N"])
y = df["Reached.on.Time_Y.N"]

# 2. Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Set experiment
mlflow.set_experiment("Basic_Model_Logistics")

with mlflow.start_run(run_name="LogReg_autolog_with_model_folder"):
    mlflow.autolog()

    model = LogisticRegression(max_iter=500)
    model.fit(X_train, y_train)

    # 4. Manual tambahan logging
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    # Log report
    with open("classification_report.txt", "w") as f:
        f.write(report)
    mlflow.log_artifact("classification_report.txt")

    # 5. Tambahan log_model ke folder 'model/'
    os.makedirs("model", exist_ok=True)
    joblib.dump(model, "model/model.pkl")
    mlflow.sklearn.log_model(model, artifact_path="model")

    print("✅ Autolog aktif & model dilog ke folder model/")
