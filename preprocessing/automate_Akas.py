import pandas as pd
from sklearn.preprocessing import StandardScaler

def load_and_preprocess(path):
    df = pd.read_csv(path)
    # lakukan scaling/encoding sesuai eksperimen
    scaler = StandardScaler()
    features = df.drop(columns=["Revenue"])
    target = df["Revenue"]
    X_scaled = scaler.fit_transform(features)
    df_scaled = pd.DataFrame(X_scaled, columns=features.columns)
    df_scaled["Revenue"] = target.values
    return df_scaled

if __name__ == "__main__":
    df_ready = load_and_preprocess("Eksperimen_SML_Akas-Bagus-Setiawan/preprocessing/online_shoppers_intention_preprocessed.csv")
    df_ready.to_csv("Eksperimen_SML_Akas-Bagus-Setiawan/preprocessing/olshopdatapreprocesed/preprocessed.csv", index=False)
