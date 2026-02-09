import pandas as pd
import numpy as np
import random
import ast

from pgmpy.estimators import HillClimbSearch, BicScore
from pgmpy.models import BayesianModel
from pgmpy.estimators import BayesianEstimator

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import KBinsDiscretizer

from pgmpy.models import DynamicBayesianNetwork as DBN
from pgmpy.factors.discrete import TabularCPD

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

EDGES_OUT = {
    "normal": "dbn_edges_nddormal.csv",
    "markov": "dbn_edges_markov.csv",
    "mrmr":   "dbn_edges_mrmr.csv",
    "pca":    "dbn_edges_pca.csv",
}

# -------------------------------------------------------------------
# RANDOM SEEDS
# -------------------------------------------------------------------
np.random.seed(42)
random.seed(42)

# -------------------------------------------------------------------
# LOAD + BASIC CLEANING
# -------------------------------------------------------------------
def load_and_clean(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Expand s_config dict → columns
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

    # Drop timestamp-ish columns (aggressive; adjust if needed)
    for col in list(df.columns):
        if "time" in col.lower():
            df = df.drop(columns=[col])

    # Drop constants and NaN rows
    df = df.loc[:, df.nunique(dropna=True) > 1].dropna()

    # Convert to numeric where possible
    df = df.apply(pd.to_numeric, errors="ignore")

    return df

# -------------------------------------------------------------------
# DISCRETIZATION: Freedman–Diaconis (max 5 bins)
# -------------------------------------------------------------------
def discretize_fd(df: pd.DataFrame, n_bins: int = 4) -> pd.DataFrame:
    """
    Advanced discretization for system metrics:
    1. Preserves 'Zero' as a unique state for sparse metrics.
    2. Uses K-Means to find natural clusters in continuous data.
    3. Handles low-variance columns by reducing bin counts automatically.
    """
    df = df.copy()
    
    for col in df.columns:
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
            
        unique_vals = df[col].nunique()
        if unique_vals <= n_bins:
            # Already categorical or very low cardinality; just map to integers
            df[col] = pd.factorize(df[col])[0]
            continue

        # Handle Zero-Inflated Metrics (e.g., error counts, packet loss)
        # If more than 20% of data is 0, we treat 0 as its own bin.
        if (df[col] == 0).sum() / len(df) > 0.20:
            non_zero_mask = df[col] > 0
            # If there's enough non-zero data to bin, cluster it
            if df.loc[non_zero_mask, col].nunique() > 1:
                # Set aside zeros, bin the rest into (n_bins - 1)
                kbd = KBinsDiscretizer(n_bins=n_bins-1, encode='ordinal', strategy='kmeans')
                vals = df.loc[non_zero_mask, col].values.reshape(-1, 1)
                # +1 because 0 is reserved for the 'Zero' state
                df.loc[non_zero_mask, col] = kbd.fit_transform(vals).flatten() + 1
                df.loc[~non_zero_mask, col] = 0
            else:
                # Just binary: 0 and 1
                df[col] = non_zero_mask.astype(int)
            continue

        # Standard K-Means binning for continuous metrics (CPU, Mem, Latency)
        try:
            kbd = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='kmeans', random_state=42)
            df[col] = kbd.fit_transform(df[col].values.reshape(-1, 1)).flatten()
        except ValueError:
            # Fallback to simple uniform binning if K-Means fails (e.g., not enough unique clusters)
            df[col] = pd.cut(df[col], bins=n_bins, labels=False, duplicates='drop')

    return df.astype(int)

# -------------------------------------------------------------------
# BLACKLIST (Causal Constraints) - SINGLE SLICE (intra)
# -------------------------------------------------------------------
def build_blacklist(df: pd.DataFrame):
    all_vars = list(df.columns)

    def has_prefix(c, prefixes):
        return any(c.lower().startswith(p) for p in prefixes)

    def has_substring(c, subs):
        return any(s in c.lower() for s in subs)

    layer0 = [v for v in all_vars if v in ["cores", "data_quality"]]
    layer1 = [v for v in all_vars if has_prefix(
        v, ["container_cpu_", "container_memory_", "container_network_", "container_fs_"]
    )]
    layer2 = [v for v in all_vars if v in ["throughput", "avg_p_latency", "buffer_size"]]
    layer3 = [v for v in all_vars if has_substring(v, ["fail", "oom", "scrape_error"])]

    layers = [layer0, layer1, layer2, layer3]
    layer_index = {v: i for i, L in enumerate(layers) for v in L}

    black = []

    # cores and data_quality have no parents
    for child in layer0:
        for parent in all_vars:
            if parent != child:
                black.append((parent, child))

    # forbid later -> earlier layers (only among variables assigned a layer)
    for p in all_vars:
        for c in all_vars:
            if p == c:
                continue
            if p in layer_index and c in layer_index and layer_index[p] > layer_index[c]:
                black.append((p, c))

    return black

# -------------------------------------------------------------------
# FEATURE PREP METHODS
# -------------------------------------------------------------------
def prepare_normal(df: pd.DataFrame) -> pd.DataFrame:
    return discretize_fd(df)

def prepare_markov(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = discretize_fd(df_raw.copy())
    bl = build_blacklist(df)
    est = HillClimbSearch(df)
    full = est.estimate(scoring_method=BicScore(df), max_indegree=MAX_INDEGREE, black_list=bl)

    mb_vars = set()
    for t in MARKOV_TARGETS:
        if t in df.columns and t in full.nodes():
            mb_vars.add(t)
            mb_vars.update(full.get_markov_blanket(t))

    mb_vars = [v for v in mb_vars if v in df.columns]
    return df[mb_vars].copy()

def prepare_mrmr(df_raw: pd.DataFrame) -> pd.DataFrame:
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

    selected.update([t for t in MRMR_TARGETS if t in df.columns])
    selected = [v for v in selected if v in df.columns]
    return df[selected].copy()

def prepare_pca(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()
    contin = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if not contin:
        return discretize_fd(df)

    X = df[contin].values
    Xs = StandardScaler().fit_transform(X)

    pca = PCA()
    pca.fit(Xs)

    cum = np.cumsum(pca.explained_variance_ratio_)
    n = np.searchsorted(cum, PCA_VAR_THRESHOLD) + 1
    n = min(n, X.shape[1])

    feats = list(dict.fromkeys([contin[np.argmax(np.abs(pca.components_[i]))] for i in range(n)]))
    for v in MUST_HAVE:
        if v in df.columns:
            feats.append(v)

    feats = list(dict.fromkeys(feats))
    df = df[feats].copy()
    return discretize_fd(df)

# -------------------------------------------------------------------
# STRUCTURE LEARNING: TWO PHASES
#   1) Learn intra-slice DAG once on df
#   2) Learn inter-slice edges only on 2-slice data (forbid intra & future->past)
# -------------------------------------------------------------------
def learn_intra_edges(df: pd.DataFrame):
    bl = build_blacklist(df)
    est = HillClimbSearch(df)
    best = est.estimate(scoring_method=BicScore(df), max_indegree=MAX_INDEGREE, black_list=bl)
    return list(best.edges())

def build_inter_only_blacklist(nodes):
    """
    Blacklist for 2-slice learning that allows ONLY edges:
        X_t -> Y_t1
    Forbids:
        X_t -> Y_t      (intra at t)
        X_t1 -> Y_t1    (intra at t+1)
        X_t1 -> Y_t     (future->past)
    """
    black = []
    for a in nodes:
        for b in nodes:
            if a == b:
                continue
            black.append((f"{a}_t",  f"{b}_t"))    # no intra at t
            black.append((f"{a}_t1", f"{b}_t1"))   # no intra at t+1
            black.append((f"{a}_t1", f"{b}_t"))    # no future->past
    return black

def learn_inter_edges_only(df: pd.DataFrame):
    nodes = list(df.columns)
    df_t  = df.iloc[:-1].reset_index(drop=True).add_suffix("_t")
    df_t1 = df.iloc[1:].reset_index(drop=True).add_suffix("_t1")
    df_2s = pd.concat([df_t, df_t1], axis=1)

    black = build_inter_only_blacklist(nodes)

    est = HillClimbSearch(df_2s)
    best = est.estimate(scoring_method=BicScore(df_2s), max_indegree=MAX_INDEGREE, black_list=black)

    # keep only t -> t1 edges
    inter = []
    for u, v in best.edges():
        if u.endswith("_t") and v.endswith("_t1"):
            inter.append((u, v))
    return inter

# -------------------------------------------------------------------
# FIT A CONSISTENT 2-SLICE BN THAT MATCHES A pgmpy DBN TEMPLATE
# -------------------------------------------------------------------
def fit_consistent_2slice_bn(df: pd.DataFrame, intra_edges, inter_edges):
    nodes = list(df.columns)

    # Build edges for static 2-slice BN:
    # - intra template applied to both slices
    intra_t  = [(f"{u}_t",  f"{v}_t")  for (u, v) in intra_edges]
    intra_t1 = [(f"{u}_t1", f"{v}_t1") for (u, v) in intra_edges]
    edges_2s = intra_t + intra_t1 + inter_edges

    # 2-slice dataset
    df_t  = df.iloc[:-1].reset_index(drop=True).add_suffix("_t")
    df_t1 = df.iloc[1:].reset_index(drop=True).add_suffix("_t1")
    df_2s = pd.concat([df_t, df_t1], axis=1)

    model_2s = BayesianModel(edges_2s)
    model_2s.fit(
        df_2s,
        estimator=BayesianEstimator,
        prior_type="BDeu",
        equivalent_sample_size=10
    )

    return model_2s, edges_2s

# -------------------------------------------------------------------
# BUILD DBN FROM THIS CONSISTENT 2-SLICE MODEL
# -------------------------------------------------------------------
def build_dbn_from_consistent_2slice(model_2s: BayesianModel, intra_edges, inter_edges, base_nodes):
    dbn = DBN()

    # add nodes
    for v in base_nodes:
        dbn.add_node((v, 0))
        dbn.add_node((v, 1))

    # add intra template at slice 0
    for u, v in intra_edges:
        dbn.add_edge((u, 0), (v, 0))
        dbn.add_edge((u, 1), (v, 1)) ###
        # pgmpy will treat this as template for all slices

    # add inter edges
    for u, v in inter_edges:
        # u is like "X_t", v is like "Y_t1"
        dbn.add_edge((u[:-2], 0), (v[:-3], 1))

    cpds_dbn = []

    # Initial CPDs from *_t
    for v in base_nodes:
        cpd_t = model_2s.get_cpds(f"{v}_t")
        if cpd_t is None:
            continue

        parents = list(cpd_t.get_evidence() or [])
        parent_cards = list(cpd_t.cardinality[1:])

        evidence_nodes = []
        for p in parents:
            if not p.endswith("_t"):
                raise ValueError(f"Unexpected parent {p} in CPD of {v}_t")
            evidence_nodes.append((p[:-2], 0))

        vals = np.array(cpd_t.values, dtype=float).reshape(int(cpd_t.variable_card), -1)

        cpds_dbn.append(
            TabularCPD(
                variable=(v, 0),
                variable_card=int(cpd_t.variable_card),
                values=vals,
                evidence=evidence_nodes if evidence_nodes else None,
                evidence_card=parent_cards if evidence_nodes else None,
            )
        )

    # Transition CPDs from *_t1 (parents can be from *_t and *_t1)
    for v in base_nodes:
        cpd_t1 = model_2s.get_cpds(f"{v}_t1")
        if cpd_t1 is None:
            continue

        parents = list(cpd_t1.get_evidence() or [])
        parent_cards = list(cpd_t1.cardinality[1:])

        evidence_nodes = []
        for p in parents:
            if p.endswith("_t"):
                evidence_nodes.append((p[:-2], 0))
            elif p.endswith("_t1"):
                evidence_nodes.append((p[:-3], 1))
            else:
                raise ValueError(f"Unexpected parent {p} in CPD of {v}_t1")

        vals = np.array(cpd_t1.values, dtype=float).reshape(int(cpd_t1.variable_card), -1)

        cpds_dbn.append(
            TabularCPD(
                variable=(v, 1),
                variable_card=int(cpd_t1.variable_card),
                values=vals,
                evidence=evidence_nodes if evidence_nodes else None,
                evidence_card=parent_cards if evidence_nodes else None,
            )
        )
    dbn.add_cpds(*cpds_dbn)
    dbn.initialize_initial_state()
    dbn.check_model()
    return dbn

# -------------------------------------------------------------------
# MAIN DBN PIPELINE
# -------------------------------------------------------------------
def build_dbn(df_ready: pd.DataFrame, method: str):
    base_nodes = list(df_ready.columns)

    # 1) learn intra once (DAG)
    intra_edges = learn_intra_edges(df_ready)

    # 2) learn inter only
    inter_edges = learn_inter_edges_only(df_ready)

    # 3) fit a consistent 2-slice BN that uses the same intra template at t and t+1
    model_2s, edges_2s = fit_consistent_2slice_bn(df_ready, intra_edges, inter_edges)

    # save for inspection
    pd.DataFrame(edges_2s, columns=["from", "to"]).to_csv(EDGES_OUT[method], index=False)

    # 4) build DBN (this will not create loops, and CPDs will match parents)
    dbn = build_dbn_from_consistent_2slice(model_2s, intra_edges, inter_edges, base_nodes)

    print("Saved consistent 2-slice edges to:", EDGES_OUT[method])
    return model_2s, dbn

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

    model_2s, dbn = build_dbn(df_ready, METHOD)
    return model_2s, dbn

if __name__ == "__main__":
    main()
