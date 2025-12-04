import pandas as pd
import numpy as np
import random
import ast
import itertools

from pgmpy.estimators import HillClimbSearch, BicScore
from pgmpy.models import BayesianModel
from pgmpy.estimators import BayesianEstimator
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------
METHOD = "normal"     # "normal", "markov", "mrmr", "pca"
INPUT_CSV = "prom_all_metrics_wide.csv"

DISCRETIZE = True
MAX_INDEGREE = 4

MARKOV_TARGETS = ["throughput", "avg_p_latency", "cores", "data_quality"]
MRMR_TARGETS  = ["throughput", "avg_p_latency", "cores", "data_quality"]
MRMR_K        = 15

PCA_VAR_THRESHOLD = 0.90
MUST_HAVE = ["throughput", "avg_p_latency", "cores", "data_quality"]

DBN_EDGES_PATH = {
    "normal": "DBN_edges_normal.csv",
    "markov": "DBN_edges_markov.csv",
    "mrmr":   "DBN_edges_mrmr.csv",
    "pca":    "DBN_edges_pca.csv",
}

DBN_CPDS_PATH = {
    "normal": "DBN_cpds_normal.csv",
    "markov": "DBN_cpds_markov.csv",
    "mrmr":   "DBN_cpds_mrmr.csv",
    "pca":    "DBN_cpds_pca.csv",
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

    # Expand s_config dict → columns
    if "s_config" in df.columns:
        def parse(val):
            try:
                d = ast.literal_eval(val)
                return d if isinstance(d, dict) else {}
            except:
                return {}
        configs = df["s_config"].apply(parse).apply(pd.Series)
        configs = configs.rename(columns=lambda c: f"s_config_{c}")
        df = pd.concat([df.drop(columns=["s_config"]), configs], axis=1)

    # Drop timestamp columns
    for col in df.columns:
        if "time" in col.lower():
            df = df.drop(columns=[col])

    # Drop constants and NaN rows
    df = df.loc[:, df.nunique(dropna=True) > 1].dropna()

    # Convert to numeric
    df = df.apply(pd.to_numeric, errors="ignore")

    return df

# -------------------------------------------------------------------
# DISCRETIZATION: FD METHOD
# -------------------------------------------------------------------
def discretize_fd(df):
    if not DISCRETIZE:
        return df

    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        col = df[c]
        if col.nunique() <= 5:
            continue

        vmin, vmax = col.min(), col.max()
        if vmin == vmax:
            continue

        n = len(col)
        q75, q25 = np.percentile(col, [75, 25])
        iqr = q75 - q25

        if iqr == 0:
            k = min(5, col.nunique())
        else:
            bin_width = 2 * iqr / (n ** (1 / 3))
            if bin_width <= 0:
                k = min(5, col.nunique())
            else:
                k = int(np.ceil((vmax - vmin) / bin_width))
                k = max(2, min(k, 5))

        bins = np.linspace(vmin, vmax, k + 1)
        df[c] = pd.cut(df[c], bins=bins, labels=range(k), include_lowest=True).astype(int)

    return df

# -------------------------------------------------------------------
# BLACKLIST (Causal Constraints) - SINGLE SLICE
# -------------------------------------------------------------------
def build_blacklist(df):
    all_vars = list(df.columns)

    def has_prefix(c, prefixes):
        return any(c.lower().startswith(p) for p in prefixes)

    def has_substring(c, subs):
        return any(s in c.lower() for s in subs)

    layer0 = [v for v in all_vars if v in ["cores", "data_quality"]]
    layer1 = [v for v in all_vars if has_prefix(v,
              ["container_cpu_", "container_memory_", "container_network_", "container_fs_"])]
    layer2 = [v for v in all_vars if v in ["throughput", "avg_p_latency", "buffer_size"]]
    layer3 = [v for v in all_vars if has_substring(v, ["fail", "oom", "scrape_error"])]

    layers = [layer0, layer1, layer2, layer3]
    layer_index = {v: i for i, L in enumerate(layers) for v in L}

    black = []

    # the things I change have no parents
    for child in layer0:
        for parent in all_vars:
            if parent != child:
                black.append((parent, child))

    # forbid reversed direction (later → earlier)
    for p in all_vars:
        for c in all_vars:
            if p == c:
                continue
            if p in layer_index and c in layer_index:
                if layer_index[p] > layer_index[c]:
                    black.append((p, c))

    return black

# -------------------------------------------------------------------
# 2-SLICE BLACKLIST (FULL DBN)
# -------------------------------------------------------------------
def build_2slice_blacklist(df, nodes):
    """
    Build blacklist for the 2-slice DBN:
      - import single-slice causal constraints into both slices
      - forbid any future → past edges: *_t1 -> *_t
    """
    base_black = build_blacklist(df)
    black_2s = []

    # apply single-slice constraints to both t and t+1
    for p, c in base_black:
        black_2s.append((f"{p}_t",  f"{c}_t"))
        black_2s.append((f"{p}_t1", f"{c}_t1"))

    # forbid future -> past for all nodes
    for v1 in nodes:
        for v2 in nodes:
            black_2s.append((f"{v1}_t1", f"{v2}_t"))

    return black_2s

# -------------------------------------------------------------------
# SAVE CPDs
# -------------------------------------------------------------------
def save_cpds(model, output_path):
    all_rows = []

    for cpd in model.get_cpds():
        var = cpd.variable
        values = np.array(cpd.values)

        # Correct parent extraction
        parents = cpd.get_evidence()
        card_list = cpd.cardinality
        parent_cards = card_list[1:]  # skip child card

        # Build state names
        var_states = [f"{var}_{i}" for i in range(values.shape[0])]
        parent_states = []
        for p, card in zip(parents, parent_cards):
            parent_states.append([f"{p}_{i}" for i in range(card)])

        # No parents: simple probability table
        if not parents:
            for i, s in enumerate(var_states):
                all_rows.append({
                    "variable": var,
                    "variable_state": s,
                    "prob": float(values[i])
                })
            continue

        # Parents exist: flatten and enumerate parent combinations
        flat = values.reshape(len(var_states), -1)

        for col, comb in enumerate(itertools.product(*parent_states)):
            for i, s in enumerate(var_states):
                row = {
                    "variable": var,
                    "variable_state": s,
                    "prob": float(flat[i, col])
                }
                for p, pname in zip(parents, comb):
                    row[p] = pname
                all_rows.append(row)

    pd.DataFrame(all_rows).to_csv(output_path, index=False)

# -------------------------------------------------------------------
# (OPTIONAL) STRUCTURE LEARNING FOR SINGLE-SLICE MODEL
# (Now mainly used by MARKOV / MRMR prep; DBN uses full 2-slice learning)
# -------------------------------------------------------------------
def learn_structure(df):
    blacklist = build_blacklist(df)
    est = HillClimbSearch(df)
    best = est.estimate(scoring_method=BicScore(df),
                        max_indegree=MAX_INDEGREE,
                        black_list=blacklist)
    return list(best.edges()), list(best.nodes())

# -------------------------------------------------------------------
# FULL 2-SLICE DBN CONSTRUCTION + FIT
# -------------------------------------------------------------------
def build_dbn(df, method):
    """
    Build a full first-order DBN over variables at times t and t+1
    by learning structure directly on the 2-slice data (df_2s),
    with constraints that forbid future->past and import intra-slice
    causal layering to both slices.
    """
    edges_path = DBN_EDGES_PATH[method]
    cpds_path  = DBN_CPDS_PATH[method]

    # base nodes (unsuffixed)
    nodes = list(df.columns)

    # build 2-slice dataset (t, t+1)
    df_t  = df.iloc[:-1].reset_index(drop=True).add_suffix("_t")
    df_t1 = df.iloc[1:].reset_index(drop=True).add_suffix("_t1")
    df_2s = pd.concat([df_t, df_t1], axis=1)

    # 2-slice blacklist: single-slice constraints + no future->past
    blacklist_2s = build_2slice_blacklist(df, nodes)

    # structure learning on 2-slice model
    est_2s = HillClimbSearch(df_2s)
    best_2s = est_2s.estimate(
        scoring_method=BicScore(df_2s),
        max_indegree=MAX_INDEGREE,
        black_list=blacklist_2s
    )

    edges = list(best_2s.edges())

    # build and fit DBN as a 2-slice Bayesian network
    model = BayesianModel(edges)
    model.fit(df_2s,
              estimator=BayesianEstimator,
              prior_type="BDeu",
              equivalent_sample_size=10)

    # save
    pd.DataFrame(edges, columns=["from", "to"]).to_csv(edges_path, index=False)
    save_cpds(model, cpds_path)

    print("Full 2-slice DBN saved:", edges_path, cpds_path)

# -------------------------------------------------------------------
# PREP METHODS
# -------------------------------------------------------------------
def prepare_normal(df):
    return discretize_fd(df)

def prepare_markov(df_raw):
    df = discretize_fd(df_raw.copy())
    bl_full = build_blacklist(df)
    est = HillClimbSearch(df)
    full = est.estimate(scoring_method=BicScore(df),
                        max_indegree=MAX_INDEGREE,
                        black_list=bl_full)

    mb_vars = set()
    for t in MARKOV_TARGETS:
        if t in df.columns and t in full.nodes():
            mb_vars.add(t)
            mb_vars.update(full.get_markov_blanket(t))
    return df[list(mb_vars)].copy()

def prepare_mrmr(df_raw):
    df = discretize_fd(df_raw.copy())
    from mrmr import mrmr_classif

    selected = set()
    for t in MRMR_TARGETS:
        if t not in df.columns:
            continue
        X = df.drop(columns=[t])
        y = df[t]
        k = min(MRMR_K, X.shape[1])
        feats = mrmr_classif(X=X, y=y, K=k)
        selected.update(feats)

    selected.update(MRMR_TARGETS)
    return df[list(selected)].copy()

def prepare_pca(df_raw):
    df = df_raw.copy()
    contin = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    X = df[contin].values
    Xs = StandardScaler().fit_transform(X)
    pca = PCA()
    pca.fit(Xs)
    cum = np.cumsum(pca.explained_variance_ratio_)
    n = np.searchsorted(cum, PCA_VAR_THRESHOLD) + 1
    n = min(n, X.shape[1])

    feats = list(dict.fromkeys([contin[np.argmax(abs(pca.components_[i]))] for i in range(n)]))
    for v in MUST_HAVE:
        if v in df.columns:
            feats.append(v)

    feats = list(dict.fromkeys(feats))
    df = df[feats].copy()
    return discretize_fd(df)

# -------------------------------------------------------------------
# MAIN
# -------------------------------------------------------------------
def main():
    df_raw = load_and_clean(INPUT_CSV)

    if METHOD == "normal":
        df_ready = prepare_normal(df_raw)
    elif METHOD == "markov":
        df_ready = prepare_markov(df_raw)
    elif METHOD == "mrmr":
        df_ready = prepare_mrmr(df_raw)
    elif METHOD == "pca":
        df_ready = prepare_pca(df_raw)
    else:
        raise ValueError("Invalid METHOD")

    build_dbn(df_ready, METHOD)

if __name__ == "__main__":
    main()
