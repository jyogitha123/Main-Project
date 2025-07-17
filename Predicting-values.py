# Predict on Validation Set
y_pred = model.predict(X_val)
y_pred_proba = model.predict_proba(X_val)[:, 1]
