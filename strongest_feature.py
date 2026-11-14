import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.base import clone
import os

CUT = 0.75

# ---------- Data and target (same as the classifier script) ----------
df = pd.read_json("cache/17097_bg.json", dtype={"ejam_uniq_id": str})
df = df[df["pop"] > 0].copy()
env_cols = [c for c in ["dpm", "traffic.score", "proximity.rmp",
                        "proximity.tsdf", "proximity.npl"] if c in df.columns]
df = df.dropna(subset=env_cols + ["pctnohealthinsurance"])
if "healthinsurance_universe" in df.columns:
    df = df[df["healthinsurance_universe"] > 200]
df = df.reset_index(drop=True)

pr = lambda s: s.rank(pct=True)
df["env_rank"] = df[env_cols].apply(pr).mean(axis=1).pipe(pr)
df["unins_rank"] = pr(df["pctnohealthinsurance"])
df["y"] = ((df["env_rank"] >= CUT) & (df["unins_rank"] >= CUT)).astype(int)
df["tract"] = df["ejam_uniq_id"].str[:11]

features = [c for c in ["pctlowinc", "pctlingiso", "pctunemployed", "pctlths",
                        "pctownedunits", "pctnobroadband", "pctdisability",
                        "pctunder5", "pctover64"] if c in df.columns]
X, y, groups = df[features], df["y"], df["tract"]

rf = make_pipeline(SimpleImputer(strategy="median"),
                   RandomForestClassifier(n_estimators=300, min_samples_leaf=3,
                                          class_weight="balanced_subsample",
                                          random_state=0, n_jobs=-1))
SEEDS, FOLDS = 3, 5

def folds():
    for seed in range(SEEDS):
        cv = StratifiedGroupKFold(n_splits=FOLDS, shuffle=True, random_state=seed)
        for tr, te in cv.split(X, y, groups):
            if y.iloc[te].nunique() == 2:
                yield tr, te

def cv_auc(cols):
    s = []
    for tr, te in folds():
        m = clone(rf).fit(X.iloc[tr][cols], y.iloc[tr])
        s.append(roc_auc_score(y.iloc[te], m.predict_proba(X.iloc[te][cols])[:, 1]))
    return np.mean(s)

full = cv_auc(features)
print(f"Full model AUC (held-out tracts): {full:.3f}\n")

# ---------- Method 1: drop-column importance ----------
# How much does AUC fall if the model never gets to see this feature?
drop = pd.Series({f: full - cv_auc([c for c in features if c != f]) for f in features})

# ---------- Method 2: solo importance ----------
# How well does this feature predict by itself?
solo = pd.Series({f: cv_auc([f]) for f in features})

# ---------- Method 3: grouped permutation importance ----------
# Correlated features are shuffled together, so one can't cover for another.
feature_groups = {
    "Economic (income, jobs, education, ownership)":
        [c for c in ["pctlowinc", "pctunemployed", "pctlths", "pctownedunits"] if c in features],
    "Language and connectivity":
        [c for c in ["pctlingiso", "pctnobroadband"] if c in features],
    "Age and disability":
        [c for c in ["pctunder5", "pctover64", "pctdisability"] if c in features],
}
rng = np.random.default_rng(0)
gperm = {k: [] for k in feature_groups}
for tr, te in folds():
    m = clone(rf).fit(X.iloc[tr], y.iloc[tr])
    base = roc_auc_score(y.iloc[te], m.predict_proba(X.iloc[te])[:, 1])
    for k, cols in feature_groups.items():
        for _ in range(5):
            Xp = X.iloc[te].copy()
            Xp[cols] = Xp[cols].values[rng.permutation(len(Xp))]   # same shuffle for the whole group
            gperm[k].append(base - roc_auc_score(y.iloc[te], m.predict_proba(Xp)[:, 1]))
gperm = pd.Series({k: np.mean(v) for k, v in gperm.items()})

# ---------- Method 4: SHAP (direction + size), on a model fit to all the data ----------
import shap
Xi = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(X), columns=features)
rf_full = RandomForestClassifier(n_estimators=500, min_samples_leaf=3,
                                 class_weight="balanced_subsample", random_state=0, n_jobs=-1).fit(Xi, y)
sv = shap.TreeExplainer(rf_full).shap_values(Xi)
sv = sv[1] if isinstance(sv, list) else (sv[:, :, 1] if sv.ndim == 3 else sv)
shap_imp = pd.Series(np.abs(sv).mean(axis=0), index=features)

# logistic standardized coefficients: simple, signed, linear view
lr = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                   LogisticRegression(class_weight="balanced", max_iter=2000)).fit(X, y)
coef = pd.Series(lr[-1].coef_[0], index=features)

# ---------- Compare ----------
table = pd.DataFrame({"drop_column": drop, "solo_AUC": solo,
                      "mean_|SHAP|": shap_imp, "logit_coef(std)": coef})
for c in ["drop_column", "mean_|SHAP|"]:
    table[c + "_rank"] = table[c].rank(ascending=False).astype(int)
table["avg_rank"] = table[["drop_column_rank", "mean_|SHAP|_rank"]].join(
    table["solo_AUC"].rank(ascending=False).rename("solo_rank")).mean(axis=1)
print(table.sort_values("avg_rank").round(3).to_string())
print("\nGrouped permutation importance (drop in AUC):")
print(gperm.sort_values(ascending=False).round(3).to_string())

# ---------- Figures ----------#
plt.close("all")
plt.rcParams.update({
    "font.size": 13, "axes.titlesize": 16, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
})

nice = {
    "pctlths": "No high school diploma", "pctownedunits": "Owner-occupied homes",
    "pctlowinc": "Low-income residents", "pctlingiso": "Linguistically isolated",
    "pctnobroadband": "No broadband", "pctunemployed": "Unemployed",
    "pctdisability": "With a disability", "pctover64": "Age 65+", "pctunder5": "Under age 5",
}
short_groups = {
    "Economic (income, jobs, education, ownership)": "Economic",
    "Language and connectivity": "Language & broadband",
    "Age and disability": "Age & disability",
}

os.makedirs("figures", exist_ok=True)


def barh(series, title, xlabel, color, fname, rename=None, fmt="{:.3f}"):
    s = series.sort_values()
    labels = [(rename or nice).get(i, i) for i in s.index]
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.barh(labels, s.values, color=color)
    ax.axvline(0, color="black", lw=0.8)
    ax.bar_label(bars, labels=[fmt.format(v) for v in s.values], padding=4, fontsize=11)
    ax.set_xlabel(xlabel)
    ax.set_title(title, loc="left")
    ax.margins(x=0.18)
    fig.tight_layout()
    fig.savefig(fname, dpi=200, bbox_inches="tight")
    plt.close(fig)

barh(gperm, "Which kind of information matters most?",
     "Drop in AUC when the whole group is shuffled", "darkorange",
     "figures/imp_1_groups.png", rename=short_groups)
barh(solo - 0.5, "How much does each factor know on its own?",
     "AUC above chance (0 = no better than guessing)", "seagreen", "figures/imp_2_solo.png")
barh(drop, "What is lost if the factor is removed?",
     "Drop in AUC (near zero = other factors cover for it)", "steelblue", "figures/imp_3_drop.png")

# SHAP: fresh figure, with readable names
plt.figure(figsize=(11, 6.5))
shap.summary_plot(sv, Xi.rename(columns=nice), show=False, plot_size=None)
plt.title("Which way does each factor push the prediction?", loc="left",
          fontsize=16, fontweight="bold")
plt.xlabel("← less likely double-burden    |    more likely double-burden →")
plt.tight_layout()
plt.savefig("figures/imp_4_shap.png", dpi=200, bbox_inches="tight")
plt.close()
