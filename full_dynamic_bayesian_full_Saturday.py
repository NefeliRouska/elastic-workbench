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

# Dynamic BN + CPD
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

DBN_EDGES_PATH = {
    "normal": "full_DBN_edges_normal.csv",
    "markov": "full_DBN_edges_markov.csv",
    "mrmr":   "full_DBN_edges_mrmr.csv",
    "pca":    "full_DBN_edges_pca.csv",
}

DBN_CPDS_PATH = {
    "normal": "full_DBN_cpds_normal.csv",
    "markov": "full_DBN_cpds_markov.csv",
    "mrmr":   "full_DBN_cpds_mrmr.csv",
    "pca":    "full_DBN_cpds_pca.csv",
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
            except Exception:
                return {}
        configs = df["s_config"].apply(parse).apply(pd.Series)
        configs = configs.rename(columns=lambda c: f"s_config_{c}")
        df = pd.concat([df.drop(columns=["s_config"]), configs], axis=1)

    # Drop timestamp columns
    for col in list(df.columns):
        if "time" in col.lower():
            df = df.drop(columns=[col])  # time is the row index, not explicit timestamps

    # Drop constants and NaN rows
    df = df.loc[:, df.nunique(dropna=True) > 1].dropna()

    # Convert to numeric where possible
    df = df.apply(pd.to_numeric, errors="ignore")

    return df

# -------------------------------------------------------------------
# DISCRETIZATION: Freedman-Diaconis METHOD
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
    layer1 = [v for v in all_vars if has_prefix(
        v, ["container_cpu_", "container_memory_", "container_network_", "container_fs_"])]
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

    # forbid reversed direction (later → earlier layer)
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

    for p, c in base_black:
        black_2s.append((f"{p}_t",  f"{c}_t"))
        black_2s.append((f"{p}_t1", f"{c}_t1"))

    for v1 in nodes:
        for v2 in nodes:
            black_2s.append((f"{v1}_t1", f"{v2}_t"))

    return black_2s

# -------------------------------------------------------------------
# HELPERS TO BUILD DYNAMIC BAYESIAN NETWORK
# -------------------------------------------------------------------
def _split_edges_2slice(edges):
    """
    Split learned 2-slice edges into:
      - intra-slice edges at t (X_t -> Y_t)
      - inter-slice edges t -> t+1 (X_t -> Y_t1)
      - intra-slice edges at t+1 (X_t1 -> Y_t1)
    Return edges in base-name form: ('X', 'Y').
    """
    intra_t = []
    inter_t_to_t1 = []
    intra_t1 = []

    for u, v in edges:
        if u.endswith("_t") and v.endswith("_t"):
            intra_t.append((u[:-2], v[:-2]))
        elif u.endswith("_t") and v.endswith("_t1"):
            inter_t_to_t1.append((u[:-2], v[:-3]))
        elif u.endswith("_t1") and v.endswith("_t1"):
            intra_t1.append((u[:-3], v[:-3]))

    return intra_t, inter_t_to_t1, intra_t1

def _to_tabular_cpd_from_pgmpy_cpd(cpd_src, child_dyn, evidence_dyn, evidence_cards):
    var_card = int(cpd_src.variable_card)

    vals = np.asarray(cpd_src.values, dtype=float)

    # Force 2D: (var_card, num_parent_configs)
    if vals.ndim == 1:
        vals = vals.reshape(var_card, 1)
    else:
        vals = vals.reshape(var_card, -1)

    # No parents => must be (var_card, 1)
    if not evidence_dyn:
        vals = vals.reshape(var_card, 1)

    return TabularCPD(
        variable=child_dyn,
        variable_card=var_card,
        values=vals,
        evidence=evidence_dyn if evidence_dyn else None,
        evidence_card=evidence_cards if evidence_dyn else None,
    )

def _build_dynamic_dbn_from_2slice(model_2s, base_nodes):
    edges_2s = list(model_2s.edges())
    intra_t, inter_t_to_t1, intra_t1 = _split_edges_2slice(edges_2s)

    dbn = DBN()

    # Nodes for two slices (0 and 1)
    for v in base_nodes:
        dbn.add_node((v, 0))
        dbn.add_node((v, 1))

    # --- IMPORTANT FIX ---
    # Add intra-slice edges with loop protection (pgmpy requires each slice be acyclic).
    # If an edge creates a loop, skip it and continue.
    for u, v in sorted(intra_t):
        try:
            dbn.add_edge((u, 0), (v, 0))
        except ValueError as e:
            if "Loops are not allowed" in str(e):
                print(f"[WARN] Skipping intra edge (slice 0) that creates a loop: {u} -> {v}")
                continue
            raise

    for u, v in sorted(intra_t1):
        try:
            dbn.add_edge((u, 1), (v, 1))
        except ValueError as e:
            if "Loops are not allowed" in str(e):
                print(f"[WARN] Skipping intra edge (slice 1) that creates a loop: {u} -> {v}")
                continue
            raise

    # Inter-slice edges t -> t+1 (cannot create a directed cycle back to slice 0)
    for u, v in sorted(inter_t_to_t1):
        dbn.add_edge((u, 0), (v, 1))

    cpds_dbn = []

    # INITIAL SLICE CPDs from *_t
    for v in base_nodes:
        name_t = f"{v}_t"
        cpd_t = model_2s.get_cpds(name_t)
        if cpd_t is None:
            continue

        parents = list(cpd_t.get_evidence())
        parent_cards = list(cpd_t.cardinality[1:])

        evidence_nodes = []
        for p in parents:
            if p.endswith("_t"):
                evidence_nodes.append((p[:-2], 0))
            elif p.endswith("_t1"):
                evidence_nodes.append((p[:-3], 0))  # shouldn't happen due to blacklist, but safe
            else:
                evidence_nodes.append((p, 0))

        cpd_init = _to_tabular_cpd_from_pgmpy_cpd(
            cpd_src=cpd_t,
            child_dyn=(v, 0),
            evidence_dyn=evidence_nodes,
            evidence_cards=parent_cards,
        )
        cpds_dbn.append(cpd_init)

    # TRANSITION SLICE CPDs from *_t1
    for v in base_nodes:
        name_t1 = f"{v}_t1"
        cpd_t1 = model_2s.get_cpds(name_t1)
        if cpd_t1 is None:
            continue

        parents = list(cpd_t1.get_evidence())
        parent_cards = list(cpd_t1.cardinality[1:])

        evidence_nodes = []
        for p in parents:
            if p.endswith("_t"):
                evidence_nodes.append((p[:-2], 0))
            elif p.endswith("_t1"):
                continue  # drop intra-slice t+1 parents

            else:
                raise ValueError(f"Unexpected unsuffixed parent '{p}' found for transition CPD of {name_t1}")

        cpd_trans = _to_tabular_cpd_from_pgmpy_cpd(
            cpd_src=cpd_t1,
            child_dyn=(v, 1),
            evidence_dyn=evidence_nodes,
            evidence_cards=parent_cards,
        )
        cpds_dbn.append(cpd_trans)

    dbn.add_cpds(*cpds_dbn)

    # sanity check:
    dbn.check_model()
    return dbn

# -------------------------------------------------------------------
# FULL 2-SLICE DBN CONSTRUCTION + FIT
# -------------------------------------------------------------------
def build_dbn(df, method):
    edges_path = DBN_EDGES_PATH[method]
    cpds_path  = DBN_CPDS_PATH[method]

    nodes = list(df.columns)

    df_t  = df.iloc[:-1].reset_index(drop=True).add_suffix("_t")
    df_t1 = df.iloc[1:].reset_index(drop=True).add_suffix("_t1")
    df_2s = pd.concat([df_t, df_t1], axis=1)

    blacklist_2s = build_2slice_blacklist(df, nodes)

    est_2s = HillClimbSearch(df_2s)
    best_2s = est_2s.estimate(
        scoring_method=BicScore(df_2s),
        max_indegree=MAX_INDEGREE,
        black_list=blacklist_2s
    )

    edges = list(best_2s.edges())

    model_2s = BayesianModel(edges)
    model_2s.fit(
        df_2s,
        estimator=BayesianEstimator,
        prior_type="BDeu",
        equivalent_sample_size=10
    )

    pd.DataFrame(edges, columns=["from", "to"]).to_csv(edges_path, index=False)

    # NOTE: cpds_path is currently not written. Kept for compatibility with your config.
    _ = cpds_path

    dbn = _build_dynamic_dbn_from_2slice(model_2s, nodes)

    print("Full 2-slice BN + DBN template built.")
    print("Saved edges to:", edges_path)
    return model_2s, dbn

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

    feats = list(dict.fromkeys(
        [contin[np.argmax(abs(pca.components_[i]))] for i in range(n)]
    ))
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

    model_2s, dbn = build_dbn(df_ready, METHOD)

    print("DBN built successfully")
    print("DBN nodes:", len(list(dbn.nodes())))
    print("DBN edges:", len(list(dbn.edges())))

if __name__ == "__main__":
    main()
