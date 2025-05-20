import argparse
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import mlflow
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="olshopdatapreprocesed/preprocessed.csv")
    args = parser.parse_args()
mlflow.set_tracking_uri("https://dagshub.com/AkasSakti/Eksperimen_SML_Akas-Bagus-Setiawan.mlflow")
mlflow.set_experiment("online-shop-intention")
    mlflow.start_run()

    df = pd.read_csv(args.data_path)
    X = df.drop(columns=["Revenue"])
    y = df["Revenue"]

    model = RandomForestClassifier()
    model.fit(X, y)
    y_pred = model.predict(X)

    report = classification_report(y, y_pred, output_dict=True)
    mlflow.log_metric("accuracy", report["accuracy"])
    mlflow.sklearn.log_model(model, "model")
    print("restart retrain")
    mlflow.end_run()
