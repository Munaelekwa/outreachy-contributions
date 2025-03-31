import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier

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

# Check dataset
print("Feature Data Types:\n", X_train.dtypes)
print("Sample Data:\n", X_train.head())

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

# Apply SMOTE to the training set only
smote = SMOTE(sampling_strategy="auto", random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train_resampled, y_train_resampled)

# Predict on validation set
y_pred = model.predict(X_valid)

# Calculate accuracy
accuracy = accuracy_score(y_valid, y_pred)
print(f"✅ Model trained successfully! Accuracy: {accuracy:.4f}")

# Save the new trained model
joblib.dump(model, "models/bbbp_model_smote.pkl")

