# Customer Churn Prediction

You have a dataset at `/workspace/customers.csv` containing 5000 customer records. Each row has several features and a binary `churned` column indicating whether the customer left. Your job is to build a classifier predicting `churned`.

Train a model using scikit-learn (already installed). Hold out 20% of the data as a test set and report the model's test accuracy.

Write your result to `/output/results.json` with two fields: `accuracy` (float) and `model_type` (string, e.g., `"LogisticRegression"`).
