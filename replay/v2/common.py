"""Shared loading and coding for the FReD v2 analyses. The holdout half can only be loaded once the
v2 pre-registration has been hashed (file PREREG_V2_HASH.txt must exist)."""
import os, re
import numpy as np, pandas as pd, pyreadr
from scipy.stats import norm

HERE = os.path.dirname(os.path.abspath(__file__))
nz = lambda s: re.sub(r"\W+", " ", str(s).lower()).strip()


def _num(x):
    return pd.to_numeric(x, errors="coerce")


def to_r(val, typ):
    """Port of FReD::convert_effect_sizes (chi-squared statistics not converted). Returns |r| or nan."""
    if pd.isna(val) or pd.isna(typ): return np.nan
    t = str(typ).lower().replace("’", "'").strip(); v = str(val).strip()
    M = {"or": "or", "odds ratio": "or", "d": "d", "cohen's d": "d", "hedges' g": "d", "hedges'g": "d", "hedge's g": "d",
         "hedges g": "d", "hedgesg": "d", "smd": "d", "etasq": "eta2", "etaq": "eta2", "eta^2": "eta2", "η²": "eta2",
         "f": "f", "cohen's f": "f", "f2": "f2", "f^2": "f2", "f²": "f2", "cohen's f^2": "f2", "r": "r", "phi": "r",
         "φ": "r", "r2": "r2", "r^2": "r2", "r²": "r2", "r-square": "r2", "test statistic": "ts", "test statistics": "ts", "test": "ts"}
    k = M.get(t)
    if k is None: return np.nan
    if k == "ts":
        s = v.lower().replace(" ", "")
        m = re.fullmatch(r"t\((\d+(?:\.\d+)?)\)=(-?\d+\.?\d*)", s)
        if m: tv, df = float(m.group(2)), float(m.group(1)); return abs(tv / np.sqrt(tv ** 2 + df))
        m = re.fullmatch(r"f\((\d+),(\d+(?:\.\d+)?)\)=(\d+\.?\d*)", s)
        if m:
            if m.group(1) != "1": return np.nan
            tv = np.sqrt(float(m.group(3))); return abs(tv / np.sqrt(tv ** 2 + float(m.group(2))))
        m = re.fullmatch(r"z=(-?\d+\.?\d*),n=(\d+)", s)
        if m: zv, n = float(m.group(1)), float(m.group(2)); return abs(zv / np.sqrt(zv ** 2 + n))
        return np.nan
    x = _num(v)
    if pd.isna(x): return np.nan
    if k == "r": return abs(x)
    if k == "r2": return np.sqrt(abs(x))
    if k == "d": return abs(x / np.sqrt(x ** 2 + 4))
    if k == "or":
        if x <= 0: return np.nan
        d = np.log(x) * np.sqrt(3) / np.pi; return abs(d / np.sqrt(d ** 2 + 4))
    if k == "eta2":
        if not 0 <= x < 1: return np.nan
        d = 2 * np.sqrt(x / (1 - x)); return d / np.sqrt(d ** 2 + 4)
    if k == "f": d = 2 * x; return abs(d / np.sqrt(d ** 2 + 4))
    if k == "f2": return np.sqrt(x / (1 + x)) if x >= 0 else np.nan
    return np.nan


def load(split="development"):
    if split != "development" and not os.path.exists(os.path.join(HERE, "PREREG_V2_HASH.txt")):
        raise SystemExit("holdout locked: hash the v2 pre-registration first")
    df = list(pyreadr.read_r(os.path.join(HERE, "..", "fred_snapshot_data.RDS")).values())[0].copy()
    df["paper"] = df["doi_o"].str.lower().str.strip()
    sp = pd.read_csv(os.path.join(HERE, "split.csv"))
    df = df.merge(sp, on="paper", how="left")
    if split != "all": df = df[df["split"] == split]
    df = df.reset_index(drop=True)
    df["y"] = df["reported_success"].map({"successful": 1.0, "failed": 0.0})
    df["y_mixed0"] = df["reported_success"].map({"successful": 1.0, "failed": 0.0, "mixed": 0.0})
    df["report"] = df["ref_r"].map(lambda s: nz(s)[:60])
    df["n_o"] = _num(df["n_o"]).where(lambda s: s > 3); df["n_r"] = _num(df["n_r"]).where(lambda s: s > 3)
    df["year_r"] = _num(df["year_r"]); df["overlap"] = _num(df["author_overlap"])
    p = _num(df["pval_value_o"]).clip(1e-12, 0.999)
    df["z_p"] = norm.ppf(1 - p / 2)
    df["r_o"] = [to_r(v, t) for v, t in zip(df["es_value_o"], df["es_type_o"])]
    df["r_r"] = [to_r(v, t) for v, t in zip(df["es_value_r"], df["es_type_r"])]
    df.loc[df.r_o >= 0.999, "r_o"] = np.nan
    df["z_es"] = np.arctanh(df.r_o) * np.sqrt(df.n_o - 3)
    return df


def auc(s, y):
    s = np.asarray(s, float); y = np.asarray(y, float); ok = np.isfinite(s) & np.isfinite(y); s, y = s[ok], y[ok]
    P, N = (y == 1).sum(), (y == 0).sum()
    if P == 0 or N == 0: return np.nan
    r = pd.Series(s).rank().values
    return (r[y == 1].sum() - P * (P + 1) / 2) / (P * N)


def cluster_boot(d, clusters, stat, B=10000, seed=1):
    rng = np.random.default_rng(seed)
    groups = [g.index.values for _, g in d.groupby(clusters)]
    out = np.empty(B)
    for b in range(B):
        idx = np.concatenate([groups[j] for j in rng.integers(0, len(groups), len(groups))])
        out[b] = stat(d.loc[idx])
    return out
