import numpy as np
import pandas as pd
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load pre-split datasets
train_data = pd.read_csv("data/bbbp_train_featurised.csv")
valid_data = pd.read_csv("data/bbbp_valid_featurised.csv")
test_data = pd.read_csv("data/bbbp_test_featurised.csv")

# Separate features (X) and target variable (y)
X_train, y_train = train_data.drop(columns=["Y"]), train_data["Y"]
X_valid, y_valid = valid_data.drop(columns=["Y"]), valid_data["Y"]
X_test, y_test = test_data.drop(columns=["Y"]), test_data["Y"]

# Drop non-numeric columns
X_train = X_train.select_dtypes(include=["number"])
X_valid = X_valid.select_dtypes(include=["number"])
X_test = X_test.select_dtypes(include=["number"])

# Replace infinity values with NaN
X_train.replace([np.inf, -np.inf], np.nan, inplace=True)
X_valid.replace([np.inf, -np.inf], np.nan, inplace=True)
X_test.replace([np.inf, -np.inf], np.nan, inplace=True)

# Cap large values to a reasonable range (1e6)
X_train = np.clip(X_train, -1e6, 1e6)
X_valid = np.clip(X_valid, -1e6, 1e6)
X_test = np.clip(X_test, -1e6, 1e6)

# Check for NaNs (which may arise after replacing `inf`)
if X_train.isnull().sum().sum() > 0:
    X_train.fillna(X_train.median(), inplace=True)  # Fill NaNs with median values

if X_valid.isnull().sum().sum() > 0:
    X_valid.fillna(X_valid.median(), inplace=True)

if X_test.isnull().sum().sum() > 0:
    X_test.fillna(X_test.median(), inplace=True)

# Create XGBoost classifier
xgb_model = xgb.XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    use_label_encoder=False,
    learning_rate=0.1,
    n_estimators=100,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

# Train the model
xgb_model.fit(X_train, y_train)

# Make predictions
y_pred = xgb_model.predict(X_test)

# Compute accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model Accuracy: {accuracy:.4f}")

# Display classification report
print("Classification Report:\n", classification_report(y_test, y_pred))
