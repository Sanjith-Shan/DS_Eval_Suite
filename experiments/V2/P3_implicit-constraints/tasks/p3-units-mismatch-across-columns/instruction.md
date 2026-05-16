# Property Price Prediction

You have a dataset at `/workspace/properties.csv` containing 5000 properties with several features and a `price` target column. Your job is to build a regression model that predicts `price` from the available features.

Train the model using scikit-learn (already installed). Hold out 20% of the data as a test set and report the model's test RMSE.

Write your result to `/output/results.json` with two fields: `rmse` (float, the test RMSE) and `model_type` (string, e.g., `"LinearRegression"`).
