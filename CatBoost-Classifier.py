# Train CatBoost Classifier
model = CatBoostClassifier(
    iterations=3000,
    depth=12,
    learning_rate=0.015,
    l2_leaf_reg=5,
    loss_function='Logloss',
    eval_metric="Accuracy",
    early_stopping_rounds=200,
    verbose=200,
    random_seed=42
)

model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=200)
