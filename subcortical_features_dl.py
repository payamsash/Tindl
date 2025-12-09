import pandas as pd
from pathlib import Path
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

#### read subjects and group information (will be fixed when dataset is complete)
fyle = open("/Users/payamsadeghishabestari/tinception/material/with_qc/behavioural/tinception_matched_optimal.txt")
lines = fyle.readlines()
subjects = [line.strip() for line in lines]
fname_group = "/Users/payamsadeghishabestari/tinception/material/with_qc/behavioural/tinception_master_ready_for_matching.csv"
df_group = pd.read_csv(fname_group)
df_group = df_group.query('`subject ID` == @subjects')

#### creating features df
rois = ["amygdalar-nuclei_lh", "amygdalar-nuclei_lh", "amygdalar-nuclei_rh", "amygdalar-nuclei_rh",
        "amygdalar-nuclei_rh", "amygdalar-nuclei_rh", "amygdalar-nuclei_rh",
        "thalamic-nuclei_lh", "thalamic-nuclei_lh", "thalamic-nuclei_lh",
        "thalamic-nuclei_rh", "thalamic-nuclei_rh", "thalamic-nuclei_rh", "thalamic-nuclei_rh"
        ]

rois_sel = ["Accessory-Basal-nucleus", "Paralaminar-nucleus", "Accessory-Basal-nucleus", "Paralaminar-nucleus",
            "Central-nucleus", "Cortical-nucleus", "Lateral-nucleus",
            "PuL", "LP", "PuM",
            "VA", "LP", "CeM", "MV(Re)"
            ]

rois_dir = Path("/Users/payamsadeghishabestari/tinception/material/ROI/tables")
df_dict = {}

for roi, roi_sel in zip(rois, rois_sel):
        fname = rois_dir / f"{roi}.csv"
        df = pd.read_csv(fname, index_col=0)

        # filter subjects
        df = df.query("subjects == @subjects")

        # extract *that* roi_sel column and rename it uniquely
        df_dict[roi_sel] = df[roi_sel].reset_index(drop=True)

df_final = pd.concat(df_dict, axis=1)
df_final["subjects"] = df_group["subject ID"]
df_final["group"] = df_group["group"]
df_final["group"] = df_final["group"].astype(str)

#### try RF for now
feature_cols = df_final.columns[:-2]   
X = df_final[feature_cols]
y = df_final["group"]                  
clf = Pipeline([
    ("scaler", StandardScaler()),
    ("model", RandomForestClassifier(
        n_estimators=300,
        random_state=42
    ))
])
scores = cross_val_score(clf, X, y, cv=5)

print("Cross-validation accuracy scores:", scores)
print("Mean accuracy:", scores.mean())
print("Std deviation:", scores.std())