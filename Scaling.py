# Standardize Numerical Features
scaler = StandardScaler()
df_train[numerical_features] = scaler.fit_transform(df_train[numerical_features])
df_test[numerical_features] = scaler.transform(df_test[numerical_features])
# Separate Features and Target
X = df_train.drop(columns=[target_column])
y = df_train[target_column]
