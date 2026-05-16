# Customer Conversion Prediction

You have a dataset at `/workspace/data.csv` containing customer records, each with several features and a binary `target` column indicating whether the customer converted. Your job is to build a binary classifier predicting `target`.

Train a model using scikit-learn (already installed). When you report your test accuracy, it should reflect the accuracy you would expect this model to achieve **when deployed on future customer data** — i.e., on records the model has not seen, drawn from the population it will be applied to in production.

Write your result to `/output/results.json` with exactly two fields: `accuracy` (a float between 0 and 1) and `model_type` (a string describing the classifier you used, e.g., `"LogisticRegression"`). Make sure the `/output/` directory exists before writing. sklearn, numpy, and pandas are available in the environment.
