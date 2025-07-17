import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, log_loss
from sklearn.ensemble import AdaBoostClassifier
from catboost import CatBoostClassifier
from sklearn.naive_bayes import GaussianNB
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

# Initialize Models
models = {
    "AdaBoost": AdaBoostClassifier(n_estimators=100, random_state=42),
    "CatBoost (Proposed)": CatBoostClassifier(iterations=300, depth=6, learning_rate=0.1, verbose=0, random_seed=42),
    "Naive Bayes": GaussianNB(),
    "Logistic Regression": LogisticRegression(max_iter=500, solver='lbfgs', random_state=42)
}

# Store Metrics
metrics = {model: {} for model in models}

# Train and Evaluate Models
for model_name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    y_pred_proba = model.predict_proba(X_val)[:, 1] if hasattr(model, "predict_proba") else np.zeros(len(y_pred))

    metrics[model_name]["accuracy"] = accuracy_score(y_val, y_pred)
    metrics[model_name]["precision"] = precision_score(y_val, y_pred, zero_division=0)
    metrics[model_name]["recall"] = recall_score(y_val, y_pred)
    metrics[model_name]["f1_score"] = f1_score(y_val, y_pred)
    metrics[model_name]["loss"] = log_loss(y_val, y_pred_proba)

# Create Bar Graphs
fig, axes = plt.subplots(1, 5, figsize=(20, 5))
metrics_names = ["accuracy", "precision", "recall", "f1_score", "loss"]
model_names = list(models.keys())

for idx, metric in enumerate(metrics_names):
    values = [metrics[model][metric] for model in models]

    axes[idx].bar(model_names, values, color=['blue', 'green', 'red', 'purple'])
    axes[idx].set_title(metric.capitalize())
    axes[idx].set_ylabel("Score")

    # Ensure correct alignment of model names
    axes[idx].set_xticklabels(model_names, rotation=80)
    axes[idx].grid(axis='y')

plt.tight_layout()
plt.show()


#  Predict on Test Data using the Best Model (CatBoost (Proposed))
best_model = models["CatBoost (Proposed)"]
test_predictions = best_model.predict(df_test)

# Save Predictions in Submission Format
df_submission["defects"] = test_predictions
df_submission.to_csv("/content/drive/MyDrive/Colab Notebooks/Binary Dataset/sample_submission.csv", index=False)
