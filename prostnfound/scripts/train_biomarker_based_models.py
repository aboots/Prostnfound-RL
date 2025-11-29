import argparse
import pandas as pd
import numpy as np
import pandas as pd


parser = argparse.ArgumentParser()
parser.add_argument(
    "--prompts",
    nargs="+",
    default=["age", "psa"],
    help="biomarkers to use in the model",
)
parser.add_argument("--target", choices=("cancer", "cspca"), default="cancer")
args = parser.parse_args()


table = pd.read_csv(
    "/h/pwilson/projects/medAI/data/nct2013/metadata_with_approx_psa_density.csv"
)

df = table

# binary target: 1 = any cancer, 0 = Benign
X = df[args.prompts].to_numpy(dtype=float)

if args.target == "cancer":
    y = (df["grade"] != "Benign").astype(int).to_numpy()
elif args.target == "cspca":
    y = (df["grade_group"] > 2).astype(int).to_numpy()
else:
    raise ValueError()

groups = df["patient_id"].to_numpy()


from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    ExtraTreesClassifier,
)
from sklearn.metrics import roc_auc_score, make_scorer
from sklearn.model_selection import cross_validate

# Features and target
X = df[["psa", "age"]].to_numpy(dtype=float)
y = (df["grade"] != "Benign").astype(int).to_numpy()
groups = df["patient_id"].to_numpy()

# AUC scorer
auc = make_scorer(roc_auc_score, needs_proba=True)

# Group-wise CV (ensures no patient is split across folds)
cv = GroupKFold(n_splits=5)

models = [
    (
        "Logistic Reg",
        make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")
        ),
    ),
    (
        "SVM (RBF)",
        make_pipeline(
            StandardScaler(),
            SVC(kernel="rbf", probability=True, class_weight="balanced"),
        ),
    ),
    (
        "k-NN (k=15)",
        make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=15)),
    ),
    (
        "Random Forest",
        RandomForestClassifier(
            n_estimators=400, class_weight="balanced", random_state=0
        ),
    ),
    (
        "Extra Trees",
        ExtraTreesClassifier(n_estimators=400, class_weight="balanced", random_state=0),
    ),
    ("Grad Boost", GradientBoostingClassifier()),
]

results = {}
for name, clf in models:
    cv_out = cross_validate(clf, X, y, cv=cv, groups=groups, scoring=auc, n_jobs=-1)
    results[name] = cv_out["test_score"]

summary = (
    pd.DataFrame(results)
    .agg(["mean", "std"])
    .T.sort_values("mean", ascending=False)
    .rename(columns={"mean": "auc_mean", "std": "auc_std"})
)

print(summary.round(3))
