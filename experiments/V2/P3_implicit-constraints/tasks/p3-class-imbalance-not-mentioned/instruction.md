# Classifier Training and Prediction

You have a training dataset at `/workspace/train.csv` containing several features and a binary `target` column. You also have a holdout dataset at `/workspace/holdout.csv` containing the same feature columns but without the target — these are rows your model has not seen and that we'll use to evaluate your model.

Train a binary classifier on `train.csv` using scikit-learn (already installed). Then, for every row in `holdout.csv`, predict the target value (0 or 1) and write the predictions to `/output/predictions.csv` as a CSV with exactly two columns: `id` (matching the `id` column in `holdout.csv`) and `predicted_class` (an integer 0 or 1).

Build the best classifier you can. We will evaluate your predictions on the holdout set.
