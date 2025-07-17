# Predict on Test Data
test_predictions = model.predict(df_test)
# Save Predictions in Submission Format
df_submission["defects"] = test_predictions
df_submission.to_csv("/content/drive/MyDrive/Colab Notebooks/Binary Dataset/sample_submission.csv", index=False)
