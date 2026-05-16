"""Quick calibration: all visits, patient-level split, NO delta features."""
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

DATA = "tasks/deduplication-loses-valid-longitudinal-data/environment/data.csv"
df = pd.read_csv(DATA)

patients = df["patient_id"].unique()
train_p, test_p = train_test_split(patients, test_size=0.2, random_state=42)

train_df = df[df["patient_id"].isin(train_p)]
test_df  = df[df["patient_id"].isin(test_p)]

feats = ["visit_number", "days_between_visits", "lab_value_1", "lab_value_2"]
clf = RandomForestClassifier(n_estimators=200, random_state=42)
clf.fit(train_df[feats], train_df["readmitted"])
proba = clf.predict_proba(test_df[feats])[:, 1]
print(f"AUC (all visits, no deltas): {roc_auc_score(test_df['readmitted'], proba):.4f}")
