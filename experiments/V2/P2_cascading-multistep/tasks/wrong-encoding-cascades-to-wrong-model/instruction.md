# Classification with Mixed Features

You have a dataset at `/workspace/dataset.csv` containing 5 000 customer records with a mix of numeric and categorical features and a binary outcome column called `target`.

The features include demographic information, engagement metrics, and several categorical variables describing customer attributes. Categorical features include both nominal ones (with no natural ordering) and at least one that may have a meaningful order in relation to the outcome. The target is a binary label (0 or 1) indicating a customer event you need to predict.

Your task is to prepare the data for supervised classification, identify the most important predictors, and evaluate a classifier. Specifically:

1. Handle all categorical features appropriately for modeling.
2. Select the **top 5 most important features** using any feature-importance method of your choice (mutual information, model-based importance, permutation importance, etc.).
3. Train a classifier (any algorithm) and evaluate it on a held-out test set (use an 80/20 train/test split with `random_state=42`).
4. Write the results to `/output/results.json` with the following fields:
   - `"accuracy"` — float, the test-set accuracy of the trained model
   - `"top_features"` — list of exactly 5 strings, the names of the top 5 features **as they exist in your encoded/transformed feature space** (e.g. `"education_level"` if treated as a single ordinal column, or `"education_level_phd"` if one-hot encoded)
   - `"model_type"` — string, the class name of the model you trained (e.g. `"LogisticRegression"`, `"RandomForestClassifier"`)

Make sure the `/output/` directory exists before writing. The verifier will check that the output file is valid JSON, that accuracy is sufficiently high, and that the reported top features are substantively correct predictors rather than artifacts of a suboptimal encoding choice.
