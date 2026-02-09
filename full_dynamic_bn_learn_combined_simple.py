import pandas as pd
import numpy as np
import random
import ast

from pgmpy.estimators import HillClimbSearch, BicScore
from pgmpy.models import BayesianModel
from pgmpy.estimators import BayesianEstimator, MaximumLikelihoodEstimator
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Dynamic BN
from pgmpy.models import DynamicBayesianNetwork as DBN

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------
METHOD = "normal"
INPUT_CSV = "prom_all_metrics_wide.csv"

DISCRETIZE = True
MAX_INDEGREE = 4

MARKOV_TARGETS = ["throughput", "avg_p_latency", "cores", "data_quality"]
MRMR_TARGETS  = ["throughput", "avg_p_latency", "cores", "data_quality"]
MRMR_K        = 15

PCA_VAR_THRESHOLD = 0.90
MUST_HAVE = ["throughput", "avg_p_latency", "cores", "data_quality"]

DBN_EDGES_PATH = {
    "normal": "full_DBN_edges_normal_simple.csv",
    "markov": "full_DBN_edges_markov_simple.csv",
    "mrmr":   "full_DBN_edges_mrmr_simple.csv",
    "pca":    "full_DBN_edges_pca_simple.csv",
}

# -------------------------------------------------------------------
# RANDOM SEEDS
# -------------------------------------------------------------------
np.random.seed(42)
random.seed(42)

# -------------------------------------------------------------------
# LOAD + BASIC CLEANING
# -------------------------------------------------------------------
def load_and_clean(csv_path):
    df = pd.read_csv(csv_path)

    if "s_config" in df.columns:
        def parse(val):
            try:
                d = ast.literal_eval(val)
                return d if isinstance(d, dict) else {}
            except Exception:
                return {}
        configs = df["s_config"].apply(parse).apply(pd.Series)
        configs = configs.rename(columns=lambda c: f"s_config_{c}")
        df = pd.concat([df.drop(columns=["s_config"]), configs], axis=1)

    for col in list(df.columns):
        if "time" in col.lower():
            df = df.drop(columns=[col])

    df = df.loc[:, df.nunique(dropna=True) > 1].dropna()
    df = df.apply(pd.to_numeric, errors="ignore")

    return df

# -------------------------------------------------------------------
# DISCRETIZATION
# -------------------------------------------------------------------
def discretize_fd(df):
    if not DISCRETIZE:
        return df

    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        if df[c].nunique() <= 5:
            continue

        vmin, vmax = df[c].min(), df[c].max()
        if vmin == vmax:
            continue

        q75, q25 = np.percentile(df[c], [75, 25])
        iqr = q75 - q25

        if iqr == 0:
            k = min(5, df[c].nunique())
        else:
            bin_width = 2 * iqr / (len(df) ** (1 / 3))
            k = int(np.ceil((vmax - vmin) / bin_width))
            k = max(2, min(k, 5))

        bins = np.linspace(vmin, vmax, k + 1)
        df[c] = pd.cut(df[c], bins=bins, labels=range(k), include_lowest=True).astype(int)

    return df

# -------------------------------------------------------------------
# BLACKLISTS
# -------------------------------------------------------------------
def build_blacklist(df):
    all_vars = list(df.columns)

    layer0 = [v for v in all_vars if v in ["cores", "data_quality"]]
    layer1 = [v for v in all_vars if v.startswith("container_")]
    layer2 = [v for v in all_vars if v in ["throughput", "avg_p_latency"]]

    layers = [layer0, layer1, layer2]
    layer_index = {v: i for i, L in enumerate(layers) for v in L}

    black = []

    for child in layer0:
        for parent in all_vars:
            if parent != child:
                black.append((parent, child))

    for p in all_vars:
        for c in all_vars:
            if p != c and p in layer_index and c in layer_index:
                if layer_index[p] > layer_index[c]:
                    black.append((p, c))

    return black

def build_2slice_blacklist(df, nodes):
    base_black = build_blacklist(df)
    black_2s = []

    for p, c in base_black:
        black_2s.append((f"{p}_t",  f"{c}_t"))
        black_2s.append((f"{p}_t1", f"{c}_t1"))

    for v1 in nodes:
        for v2 in nodes:
            black_2s.append((f"{v1}_t1", f"{v2}_t"))

    return black_2s

# -------------------------------------------------------------------
# BUILD DBN (CORRECT & STABLE)
# -------------------------------------------------------------------
def build_dbn(df, method):

    nodes = list(df.columns)

    # ---------- 2-slice dataset ----------
    df_t  = df.iloc[:-1].reset_index(drop=True).add_suffix("_t")
    df_t1 = df.iloc[1:].reset_index(drop=True).add_suffix("_t1")
    df_2s = pd.concat([df_t, df_t1], axis=1)

    blacklist_2s = build_2slice_blacklist(df, nodes)

    est = HillClimbSearch(df_2s)
    best_2s = est.estimate(
        scoring_method=BicScore(df_2s),
        max_indegree=MAX_INDEGREE,
        black_list=blacklist_2s
    )

    edges = list(best_2s.edges())

    # ---------- static 2-slice BN (BayesianEstimator OK here) ----------
    model_2s = BayesianModel(edges)
    model_2s.fit(
        df_2s,
        estimator=BayesianEstimator,
        prior_type="BDeu",
        equivalent_sample_size=10
    )

    # ---------- build DBN structure ----------
    dbn = DBN()

    for u, v in edges:
        if u.endswith("_t") and v.endswith("_t"):
            dbn.add_edge((u[:-2], 0), (v[:-2], 0))
        elif u.endswith("_t") and v.endswith("_t1"):
            dbn.add_edge((u[:-2], 0), (v[:-3], 1))

    # ---------- DBN DATA (tuple columns) ----------
    df_dbn = pd.concat(
        [df.iloc[:-1].reset_index(drop=True),
         df.iloc[1:].reset_index(drop=True)],
        axis=1,
        keys=[0, 1]
    )
    df_dbn.columns = [(var, t) for t, var in df_dbn.columns]

    # ---------- FIT DBN (ONLY MLE IS SUPPORTED) ----------
    dbn.fit(df_dbn)

    dbn.check_model()

    pd.DataFrame(edges, columns=["from", "to"]).to_csv(DBN_EDGES_PATH[method], index=False)

    return model_2s, dbn

def add_pseudocounts(df):       # (this thing with BDeu)
    """
    Adds one occurrence of each state per variable
    to avoid zero-probability CPDs in DBN MLE fitting.
    """
    rows = []

    for col in df.columns:
        states = df[col].unique()
        for s in states:
            row = df.iloc[0].copy()
            row[col] = s
            rows.append(row)

    return pd.concat([df, pd.DataFrame(rows)], ignore_index=True)


# -------------------------------------------------------------------
# PREP
# -------------------------------------------------------------------
def prepare_normal(df):
    return discretize_fd(df)

# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------
def main():
    df_raw = load_and_clean(INPUT_CSV)
    df_ready = prepare_normal(df_raw)
    df_ready = add_pseudocounts(df_ready)

    model_2s, dbn = build_dbn(df_ready, METHOD)
    print("DBN built successfully")

if __name__ == "__main__":
    main()
