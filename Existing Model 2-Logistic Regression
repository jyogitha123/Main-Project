import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss
from sklearn.linear_model import LogisticRegression
from imblearn.combine import SMOTEENN
from sklearn.preprocessing import StandardScaler

# Load Datasets
train_path = "/content/drive/MyDrive/Colab Notebooks/Binary Dataset/train.csv"
test_path = "/content/drive/MyDrive/Colab Notebooks/Binary Dataset/test.csv"
submission_path = "/content/drive/MyDrive/Colab Notebooks/Binary Dataset/sample_submission.csv"

df_train = pd.read_csv(train_path)
df_test = pd.read_csv(test_path)
df_submission = pd.read_csv(submission_path)

# Define Target and Features
target_column = "defects"
if target_column not in df_train.columns:
    raise ValueError(f" Error: Target column '{target_column}' not found in dataset!")

# Remove rows where target column is NaN
df_train = df_train.dropna(subset=[target_column])

# Identify Numerical & Categorical Features
categorical_features = df_train.select_dtypes(include=['object']).columns.tolist()
numerical_features = df_train.select_dtypes(exclude=['object']).columns.tolist()
numerical_features.remove(target_column)

# Fill Missing Values
df_train.fillna(df_train.median(), inplace=True)
df_test.fillna(df_test.median(), inplace=True)

# Standardize Numerical Features
scaler = StandardScaler()
df_train[numerical_features] = scaler.fit_transform(df_train[numerical_features])
df_test[numerical_features] = scaler.transform(df_test[numerical_features])

# Separate Features and Target
X = df_train.drop(columns=[target_column])
y = df_train[target_column]

# Apply SMOTEENN for Better Class Balancing
smoteenn = SMOTEENN(random_state=42)
X_resampled, y_resampled = smoteenn.fit_resample(X, y)

# Train-Test Split
X_train, X_val, y_train, y_val = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=None
)

# Train Logistic Regression Classifier
model = LogisticRegression(max_iter=500, solver='lbfgs', random_state=42)
model.fit(X_train, y_train)

# Predict on Validation Set
y_pred = model.predict(X_val)
y_pred_proba = model.predict_proba(X_val)[:, 1]

# Evaluate Model
accuracy = accuracy_score(y_val, y_pred)
precision = precision_score(y_val, y_pred, zero_division=0)
recall = recall_score(y_val, y_pred)
f1 = f1_score(y_val, y_pred)
loss = log_loss(y_val, y_pred_proba)

print(f"Final Model Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Log Loss: {loss:.4f}")

# Predict on Test Data
test_predictions = model.predict(df_test)

# Save Predictions in Submission Format
df_submission["defects"] = test_predictions
df_submission.to_csv("/content/drive/MyDrive/Colab Notebooks/Binary Dataset/sample_submission.csv", index=False)
