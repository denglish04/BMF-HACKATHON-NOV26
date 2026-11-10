import pathlib
import numpy as np, pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
from sklearn.inspection import permutation_importance
import os

CUT = 0.75            # "top quarter"; also try 0.67 if there are too few positives
USE_RACE = False      # True adds pctmin as a final check, not as a main feature

# ---------- 1) Data and the double-burden flag ----------
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
df["double_burden"] = ((df["env_rank"] >= CUT) & (df["unins_rank"] >= CUT)).astype(int)
df["tract"] = df["ejam_uniq_id"].str[:11]

print(f"{len(df)} blockgroups, {df['double_burden'].sum()} double-burden "
      f"({df['double_burden'].mean():.1%}), "
      f"{100 * df.loc[df['double_burden'] == 1, 'pop'].sum() / df['pop'].sum():.1f}% of residents")

# ---------- 2) Features: social and housing ONLY (no exposure, no insurance) ----------
features = [c for c in ["pctlowinc", "pctlingiso", "pctunemployed", "pctlths",
                        "pctownedunits", "pctnobroadband", "pctdisability",
                        "pctunder5", "pctover64"] if c in df.columns]
if USE_RACE and "pctmin" in df.columns:
    features.append("pctmin")
X, y, groups = df[features], df["double_burden"], df["tract"]

models = {
    "Logistic (baseline)": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                         LogisticRegression(class_weight="balanced", max_iter=2000)),
    "Random forest": make_pipeline(SimpleImputer(strategy="median"),
                                   RandomForestClassifier(n_estimators=500, min_samples_leaf=3,
                                                          class_weight="balanced_subsample",
                                                          random_state=0, n_jobs=-1)),
}

# ---------- 3) Grouped cross-validation (whole tracts held out), repeated ----------
SEEDS, FOLDS = 5, 5
scores = {n: {"auc": [], "ap": []} for n in models}
oof = {n: np.zeros(len(df)) for n in models}
imp = pd.Series(0.0, index=features)
n_imp = 0

for seed in range(SEEDS):
    cv = StratifiedGroupKFold(n_splits=FOLDS, shuffle=True, random_state=seed)
    for tr, te in cv.split(X, y, groups):
        if y.iloc[te].nunique() < 2:
            continue
        for name, mdl in models.items():
            mdl.fit(X.iloc[tr], y.iloc[tr])
            p = mdl.predict_proba(X.iloc[te])[:, 1]
            oof[name][te] += p / SEEDS
            scores[name]["auc"].append(roc_auc_score(y.iloc[te], p))
            scores[name]["ap"].append(average_precision_score(y.iloc[te], p))
        r = permutation_importance(models["Random forest"], X.iloc[te], y.iloc[te],
                                   scoring="roc_auc", n_repeats=10, random_state=0)
        imp += pd.Series(r.importances_mean, index=features)
        n_imp += 1

base = y.mean()
print(f"\nBaseline (random guessing) precision = {base:.2f}")
for name, s in scores.items():
    print(f"{name:22s} AUC {np.mean(s['auc']):.2f} ± {np.std(s['auc']):.2f} | "
          f"avg precision {np.mean(s['ap']):.2f} ± {np.std(s['ap']):.2f}")

# Precision when targeting the top 10% / 20% of predicted blockgroups
for frac in [0.10, 0.20]:
    k = int(len(df) * frac)
    top = np.argsort(-oof["Random forest"])[:k]
    print(f"Targeting top {frac:.0%} by predicted risk: {y.iloc[top].mean():.0%} are truly double-burden "
          f"(vs {base:.0%} by chance)")

imp = (imp / n_imp).sort_values()
print("\nPermutation importance (drop in AUC when shuffled):")
print(imp.sort_values(ascending=False).round(3).to_string())

# ---------- 4) Figure ----------
os.makedirs("figures", exist_ok=True)
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
for name in models:
    prec, rec, _ = precision_recall_curve(y, oof[name])
    ax[0].plot(rec, prec, lw=2.5, label=f"{name} (AP {average_precision_score(y, oof[name]):.2f})")
ax[0].axhline(base, ls="--", color="gray", label=f"Chance ({base:.2f})")
ax[0].set_xlabel("Recall (share of double-burden blockgroups found)")
ax[0].set_ylabel("Precision")
ax[0].set_title("Can social indicators locate double-burden areas?\n(held-out tracts)")
ax[0].legend(); ax[0].grid(alpha=0.3)

imp.plot.barh(ax=ax[1], color="steelblue")
ax[1].set_xlabel("Drop in AUC when feature is shuffled")
ax[1].set_title("What drives the prediction")
plt.tight_layout(); plt.savefig("figures/double_burden_ml.png", dpi=200)

# ---------- 5) Optional map: predicted vs. actual ----------
zpath = pathlib.Path("cache/cb_il_bg.zip")
if zpath.exists():
    import geopandas as gpd
    shp = gpd.read_file(zpath)
    shp = shp[shp["COUNTYFP"] == "097"][["GEOID", "geometry"]]
    df["pred_prob"] = oof["Random forest"]
    g = shp.merge(df[["ejam_uniq_id", "pred_prob", "double_burden"]],
                  left_on="GEOID", right_on="ejam_uniq_id", how="inner")
    fig, ax = plt.subplots(1, 2, figsize=(13, 6.5))
    g.plot(ax=ax[0], column="pred_prob", cmap="YlOrRd", legend=True, edgecolor="white", linewidth=0.2)
    ax[0].set_title("Predicted probability of double burden\n(from social indicators only, out-of-fold)")
    g.plot(ax=ax[1], color="#eeeeee", edgecolor="white", linewidth=0.2)
    g[g["double_burden"] == 1].plot(ax=ax[1], color="#3b4994", edgecolor="white", linewidth=0.2)
    ax[1].set_title("Actual double-burden blockgroups")
    for a in ax: a.axis("off")
    plt.tight_layout(); plt.savefig("figures/double_burden_map.png", dpi=200)

df.to_csv("lake_county_double_burden.csv", index=False)
