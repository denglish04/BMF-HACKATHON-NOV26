"""Export data for the interactive "Double Burden" project site.

Re-runs the analysis from double_burden.py and strongest_feature.py at several
cutoffs and writes:
  <out>/blockgroups.geojson  - Lake County block groups + per-cutoff predictions
  <out>/results.json         - metrics, PR curves, feature importance, SHAP

Usage:
  .venv/bin/python export_site_data.py [--out ../double-burden-lake-county/data]
"""
import argparse, json, pathlib
import numpy as np, pandas as pd
import geopandas as gpd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve
from sklearn.inspection import permutation_importance

CUTS = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85]
DEFAULT_CUT = 0.75
SEEDS, FOLDS = 5, 5
MIN_POSITIVES = 15

ENV_COLS = ["dpm", "traffic.score", "proximity.rmp", "proximity.tsdf", "proximity.npl"]
FEATURES = ["pctlowinc", "pctlingiso", "pctunemployed", "pctlths", "pctownedunits",
            "pctnobroadband", "pctdisability", "pctunder5", "pctover64"]
FEATURE_GROUPS = {
    "Economic": ["pctlowinc", "pctunemployed", "pctlths", "pctownedunits"],
    "Language & broadband": ["pctlingiso", "pctnobroadband"],
    "Age & disability": ["pctunder5", "pctover64", "pctdisability"],
}
NICE = {
    "pctlths": "No high school diploma", "pctownedunits": "Owner-occupied homes",
    "pctlowinc": "Low-income residents", "pctlingiso": "Linguistically isolated",
    "pctnobroadband": "No broadband", "pctunemployed": "Unemployed",
    "pctdisability": "With a disability", "pctover64": "Age 65+", "pctunder5": "Under age 5",
}


def key(cut):
    return f"{cut:.2f}"


def r(x, n=4):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), n)


# ---------- Data (same prep as double_burden.py) ----------
def load():
    df = pd.read_json("cache/17097_bg.json", dtype={"ejam_uniq_id": str})
    df = df[df["pop"] > 0].copy()
    df = df.dropna(subset=ENV_COLS + ["pctnohealthinsurance"])
    df = df[df["healthinsurance_universe"] > 200].reset_index(drop=True)
    pr = lambda s: s.rank(pct=True)
    df["env_rank"] = df[ENV_COLS].apply(pr).mean(axis=1).pipe(pr)
    df["unins_rank"] = pr(df["pctnohealthinsurance"])
    df["tract"] = df["ejam_uniq_id"].str[:11]
    return df


def flag(df, cut):
    return ((df["env_rank"] >= cut) & (df["unins_rank"] >= cut)).astype(int)


def make_models():
    return {
        "logistic": make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                                  LogisticRegression(class_weight="balanced", max_iter=2000)),
        "forest": make_pipeline(SimpleImputer(strategy="median"),
                                RandomForestClassifier(n_estimators=500, min_samples_leaf=3,
                                                       class_weight="balanced_subsample",
                                                       random_state=0, n_jobs=-1)),
    }


def folds(X, y, groups, seeds):
    for seed in range(seeds):
        cv = StratifiedGroupKFold(n_splits=FOLDS, shuffle=True, random_state=seed)
        for tr, te in cv.split(X, y, groups):
            if y.iloc[te].nunique() == 2:
                yield tr, te


def pr_points(y, p, n=80):
    prec, rec, _ = precision_recall_curve(y, p)
    idx = np.unique(np.linspace(0, len(rec) - 1, n).astype(int))
    return [[r(rec[i], 3), r(prec[i], 3)] for i in idx]


# ---------- Classifier at one cutoff (double_burden.py) ----------
def run_cut(df, cut):
    X, y, groups = df[FEATURES], flag(df, cut), df["tract"]
    models = make_models()
    scores = {n: {"auc": [], "ap": []} for n in models}
    oof = {n: np.zeros(len(df)) for n in models}
    imp = pd.Series(0.0, index=FEATURES)
    n_imp = 0
    for tr, te in folds(X, y, groups, SEEDS):
        for name, mdl in models.items():
            mdl.fit(X.iloc[tr], y.iloc[tr])
            p = mdl.predict_proba(X.iloc[te])[:, 1]
            oof[name][te] += p / SEEDS
            scores[name]["auc"].append(roc_auc_score(y.iloc[te], p))
            scores[name]["ap"].append(average_precision_score(y.iloc[te], p))
        pi = permutation_importance(models["forest"], X.iloc[te], y.iloc[te],
                                    scoring="roc_auc", n_repeats=10, random_state=0)
        imp += pd.Series(pi.importances_mean, index=FEATURES)
        n_imp += 1
    out = {
        "positives": int(y.sum()),
        "share": r(y.mean()),
        "pop_share": r(df.loc[y == 1, "pop"].sum() / df["pop"].sum()),
        "models": {
            n: {"auc": r(np.mean(s["auc"])), "auc_sd": r(np.std(s["auc"])),
                "ap": r(np.mean(s["ap"])), "ap_sd": r(np.std(s["ap"])),
                "pr": pr_points(y, oof[n])}
            for n, s in scores.items()
        },
        "permutation": {f: r(v) for f, v in (imp / n_imp).items()},
    }
    return out, oof["forest"]


# ---------- Extra importance views at the default cutoff (strongest_feature.py) ----------
def importance_views(df, cut):
    import shap
    X, y, groups = df[FEATURES], flag(df, cut), df["tract"]
    rf = make_pipeline(SimpleImputer(strategy="median"),
                       RandomForestClassifier(n_estimators=300, min_samples_leaf=3,
                                              class_weight="balanced_subsample",
                                              random_state=0, n_jobs=-1))
    fold_list = list(folds(X, y, groups, 3))

    def cv_auc(cols):
        s = []
        for tr, te in fold_list:
            m = clone(rf).fit(X.iloc[tr][cols], y.iloc[tr])
            s.append(roc_auc_score(y.iloc[te], m.predict_proba(X.iloc[te][cols])[:, 1]))
        return np.mean(s)

    full = cv_auc(FEATURES)
    drop = {f: r(full - cv_auc([c for c in FEATURES if c != f])) for f in FEATURES}
    solo = {f: r(cv_auc([f]) - 0.5) for f in FEATURES}

    rng = np.random.default_rng(0)
    gperm = {k: [] for k in FEATURE_GROUPS}
    for tr, te in fold_list:
        m = clone(rf).fit(X.iloc[tr], y.iloc[tr])
        base = roc_auc_score(y.iloc[te], m.predict_proba(X.iloc[te])[:, 1])
        for k, cols in FEATURE_GROUPS.items():
            for _ in range(5):
                Xp = X.iloc[te].copy()
                Xp[cols] = Xp[cols].values[rng.permutation(len(Xp))]
                gperm[k].append(base - roc_auc_score(y.iloc[te], m.predict_proba(Xp)[:, 1]))
    gperm = {k: r(np.mean(v)) for k, v in gperm.items()}

    Xi = pd.DataFrame(SimpleImputer(strategy="median").fit_transform(X), columns=FEATURES)
    rf_full = RandomForestClassifier(n_estimators=500, min_samples_leaf=3,
                                     class_weight="balanced_subsample",
                                     random_state=0, n_jobs=-1).fit(Xi, y)
    sv = shap.TreeExplainer(rf_full).shap_values(Xi)
    sv = sv[1] if isinstance(sv, list) else (sv[:, :, 1] if sv.ndim == 3 else sv)
    pct = Xi.rank(pct=True)  # feature value as a percentile, for coloring
    shap_out = {f: {"shap": [r(v) for v in sv[:, j]],
                    "value_pct": [r(v, 3) for v in pct[f]],
                    "value": [r(v, 4) for v in Xi[f]]}
                for j, f in enumerate(FEATURES)}
    shap_mean = {f: r(np.abs(sv[:, j]).mean()) for j, f in enumerate(FEATURES)}

    print(f"  full-model AUC {full:.3f}")
    return {"cut": cut, "full_auc": r(full), "drop_column": drop, "solo_auc_above_chance": solo,
            "grouped_permutation": gperm, "shap_mean_abs": shap_mean, "shap": shap_out}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../double-burden-lake-county/data")
    out = pathlib.Path(ap.parse_args().out)
    out.mkdir(parents=True, exist_ok=True)

    df = load()
    results = {"n_blockgroups": len(df), "total_pop": int(df["pop"].sum()),
               "default_cut": DEFAULT_CUT, "features": FEATURES, "nice": NICE,
               "feature_groups": FEATURE_GROUPS, "env_cols": ENV_COLS, "cuts": {}}
    preds = {}
    for cut in CUTS:
        npos = int(flag(df, cut).sum())
        if npos < MIN_POSITIVES:
            print(f"cut {cut:.2f}: only {npos} positives, skipping")
            continue
        print(f"cut {cut:.2f}: {npos} positives ...")
        res, oof_rf = run_cut(df, cut)
        for n, m in res["models"].items():
            print(f"  {n:9s} AUC {m['auc']:.2f} ± {m['auc_sd']:.2f} | AP {m['ap']:.2f} ± {m['ap_sd']:.2f}")
        results["cuts"][key(cut)] = res
        preds[key(cut)] = oof_rf

    print(f"importance views at {DEFAULT_CUT:.2f} ...")
    results["importance"] = importance_views(df, DEFAULT_CUT)

    # ---------- Geometry ----------
    shp = gpd.read_file("cache/cb_il_bg.zip")
    shp = shp[shp["COUNTYFP"] == "097"][["GEOID", "geometry"]].to_crs(4326)
    shp["geometry"] = shp.geometry.simplify(0.0003, preserve_topology=True)
    keep = ["ejam_uniq_id", "pop", "env_rank", "unins_rank", "pctnohealthinsurance"] + ENV_COLS + FEATURES
    props = df[keep].copy()
    for k, v in preds.items():
        props[f"oof_{k}"] = v
    g = shp.merge(props, left_on="GEOID", right_on="ejam_uniq_id", how="left")
    g["included"] = g["ejam_uniq_id"].notna()
    g = g.drop(columns=["ejam_uniq_id"])
    for c in g.columns:
        if c not in ("GEOID", "geometry", "included") and g[c].dtype.kind == "f":
            g[c] = g[c].round(4)
    (out / "blockgroups.geojson").write_text(g.to_json(drop_id=True, na="null", to_wgs84=True))
    # Drop excess coordinate precision (~1 m is plenty).
    gj = json.loads((out / "blockgroups.geojson").read_text())

    def rnd(c):
        return [rnd(x) for x in c] if isinstance(c[0], list) else [round(c[0], 5), round(c[1], 5)]
    for f in gj["features"]:
        f["geometry"]["coordinates"] = rnd(f["geometry"]["coordinates"])
    (out / "blockgroups.geojson").write_text(json.dumps(gj, separators=(",", ":")))
    (out / "results.json").write_text(json.dumps(results, separators=(",", ":")))
    print(f"wrote {out/'blockgroups.geojson'} ({len(g)} shapes, {g['included'].sum()} with data)")
    print(f"wrote {out/'results.json'}")


if __name__ == "__main__":
    main()
