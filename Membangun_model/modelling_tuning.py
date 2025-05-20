import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV

# Load dataset
df = pd.read_csv("Membangun_model/olshopdatapreprocesed/preprocessed.csv")
X = df.drop(columns=["Revenue"])
y = df["Revenue"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Hyperparameter tuning
param_grid = {
    "n_estimators": [50, 100],
    "max_depth": [5, 10, None],
    "min_samples_split": [2, 5]
}

grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=3, scoring='accuracy')
grid_search.fit(X_train, y_train)

print("Best Params:", grid_search.best_params_)

# Save best model
import joblib
joblib.dump(grid_search.best_estimator_, "Membangun_model/rf_model_tuned.pkl")
