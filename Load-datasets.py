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
