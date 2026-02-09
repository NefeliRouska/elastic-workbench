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

#DBN_CPDS_PATH = {
#    "normal": "full_DBN_cpds_normal.csv",
#    "markov": "full_DBN_cpds_markov.csv",
#    "mrmr":   "full_DBN_cpds_mrmr.csv",
#    "pca":    "full_DBN_cpds_pca.csv",
#}

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
            df = df.drop(columns=[col]) #the notion of time in the model is purely the row index, not explicit timestamps.

    # Drop constants and NaN rows
    df = df.loc[:, df.nunique(dropna=True) > 1].dropna() #keeps only columns that vary

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
        if not pd.api.types.is_numeric_dtype(df[c]): #If it’s not numeric, skip (e.g. strings)
            continue
        col = df[c]
        if col.nunique() <= 5: #If it already has ≤ 5 unique values, you keep it as is
            continue

        vmin, vmax = col.min(), col.max() #If min = max → column is constant → already removed earlier, safety check.
        if vmin == vmax:
            continue

        n = len(col)
        q75, q25 = np.percentile(col, [75, 25]) #interquartile range = measure of spread of the central 50% of values.
        iqr = q75 - q25

        if iqr == 0:
            k = min(5, col.nunique())
        else:
            bin_width = 2 * iqr / (n ** (1 / 3))
            if bin_width <= 0:
                k = min(5, col.nunique())
            else:
                k = int(np.ceil((vmax - vmin) / bin_width)) #range/bin_width, how many bins I need to cover the range
                k = max(2, min(k, 5)) #at least 2 bins, at most 5 bins

        bins = np.linspace(vmin, vmax, k + 1) #bin edges
        df[c] = pd.cut(df[c], bins=bins, labels=range(k), include_lowest=True).astype(int) #assigns each value to bin 0…k-1.

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
        v, ["container_cpu_", "container_memory_", "container_network_", "container_fs_"] )]
    layer2 = [v for v in all_vars if v in ["throughput", "avg_p_latency", "buffer_size"]]
    layer3 = [v for v in all_vars if has_substring(v, ["fail", "oom", "scrape_error"])]

    layers = [layer0, layer1, layer2, layer3]
    layer_index = {v: i for i, L in enumerate(layers) for v in L}

    black = []

    # the things I change have no parents, forbid parent → cores and parent → data_quality
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

    # apply single-slice constraints to both t and t+1 // If p → c is forbidden within one slice, then: p_t → c_t is forbidden, p_t1 → c_t1 is also forbidden.
    for p, c in base_black:
        black_2s.append((f"{p}_t",  f"{c}_t"))
        black_2s.append((f"{p}_t1", f"{c}_t1"))

    # forbid future -> past for all nodes
    for v1 in nodes:
        for v2 in nodes:
            black_2s.append((f"{v1}_t1", f"{v2}_t"))

    return black_2s

"""
# -------------------------------------------------------------------
# SAVE CPDs (STATIC 2-SLICE MODEL)
# -------------------------------------------------------------------
def save_cpds(model, output_path):
    all_rows = []

    for cpd in model.get_cpds():
        var = cpd.variable #e.g. throughput_t1
        values = np.array(cpd.values) #numpy array of probabilities.

        parents = list(cpd.get_evidence()) #list of parent variable names
        card_list = list(cpd.cardinality) #[child_card, parent1_card, parent2_card, ...]
        parent_cards = card_list[1:]  # skip child card/ just parent cardinalities

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

        # Parents exist: flatten and enumerate parent combinations // which variable/state, which parent states, the probability.
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
# SINGLE-SLICE STRUCTURE LEARNING (prepare markov)
# -------------------------------------------------------------------
def learn_structure(df):
    blacklist = build_blacklist(df)
    est = HillClimbSearch(df)
    best = est.estimate(scoring_method=BicScore(df),
                        max_indegree=MAX_INDEGREE,
                        black_list=blacklist)
    return list(best.edges()), list(best.nodes())
# a set of edges among variables at one time, used later to extract Markov blankets around targets.
"""

# -------------------------------------------------------------------
# HELPERS TO BUILD DYNAMIC BAYESIAN NETWORK
# -------------------------------------------------------------------
def _split_edges_2slice(edges):
    """
    Split learned 2-slice edges into:
      - intra-slice edges at t (X_t -> Y_t)
      - inter-slice edges t -> t+1 (X_t -> Y_t1)
    
    """
    intra_t = []
    inter_t_to_t1 = []
    intra_t1 = []

    for u, v in edges:
        if u.endswith("_t") and v.endswith("_t"):
            intra_t.append((u[:-2], v[:-2]))  # within t
        elif u.endswith("_t") and v.endswith("_t1"):
            inter_t_to_t1.append((u[:-2], v[:-3]))  # from t to t1
        #elif u.endswith("_t1") and v.endswith("_t1"):
        #    intra_t1.append((u[:-3], v[:-3]))  # from t1 to t1
        

    #For "X_t" → remove 2 characters // For "Y_t1" → remove 3 characters // u[:-2] all elements of u except the last two

    return intra_t, inter_t_to_t1, intra_t1


def _build_dynamic_dbn_from_2slice(model_2s, base_nodes):  #heart
    """
    Construct a DynamicBayesianNetwork using:
      - structure learned on (t, t+1) slices
      - CPDs of *_t as initial slice
      - CPDs of *_t1 as transition slice
      - model_2s = a BayesianModel learned on variables: cores_t, … cores_t1.
      - base_nodes = the unsuffixed variable names in one slice: ["cores", "throughput"..]
    """
    edges_2s = list(model_2s.edges()) #edges_2s is a list of pairs like: ("cores_t", "throughput_t"), ("throughput_t", "throughput_t1")
    intra_t, inter_t_to_t1, intra_t1 = _split_edges_2slice(edges_2s) #put the within t in a group and from t to t1 in another group

    dbn = DBN()

    # Nodes for two slices (0 and 1) DBN skeleton
    for v in base_nodes:
        dbn.add_node((v, 0))
        dbn.add_node((v, 1))

    # Intra-slice edges at time 0
    for u, v in intra_t:
        dbn.add_edge((u, 0), (v, 0))
    # Intra-slice template (used for every slice): union of t and t1 intra edges
    #intra_template = list(set(intra_t) | set(intra_t1))
    #for u, v in intra_template:
    #    dbn.add_edge((u, 0), (v, 0))


    # Inter-slice edges t -> t+1
    for u, v in inter_t_to_t1:
        dbn.add_edge((u, 0), (v, 1))

    # Intra-slice edges t+1 -> t+1
    #for u, v in intra_t1:
    #    dbn.add_edge((u, 1), (v, 1))


    cpds_dbn = []

    # -------------------------------------------------
    # INITIAL SLICE CPDs from *_t
    # -------------------------------------------------
    for v in base_nodes: #for each variable
        name_t = f"{v}_t"
        try:
            cpd_t = model_2s.get_cpds(name_t) #picks the CPD for each variable at time t
        except Exception:
            continue

        parents = list(cpd_t.get_evidence())              # parent names (original order) Every CPD object in pgmpy stores its parents internally.
        parent_cards = list(cpd_t.cardinality[1:])        # in same order Every CPD object knows how many states each variable has.
        #Cardinality = number of possible discrete states the variable can take.
        #How many parent-state combinations exist→ needed to validate the probability table shape // How to interpret the columns of the probability matrix → each column corresponds to one parent configuration

        # Map parents to slice 0, preserve order
        evidence_nodes = []
        for p in parents:
            if p.endswith("_t"):
                evidence_nodes.append((p[:-2], 0))
            elif p.endswith("_t1"):
                evidence_nodes.append((p[:-3], 0))  # shouldn't happen due to blacklist, but safe
            else:
                evidence_nodes.append((p, 0))

        # Ensure values is 2D, shape (variable_card, -1)
        vals = np.array(cpd_t.values, dtype=float)
        vals = vals.reshape(int(cpd_t.variable_card), -1)

        cpd_init = TabularCPD( #brain of dbn
            variable=(v, 0),
            variable_card=int(cpd_t.variable_card),
            values=vals,
            evidence=evidence_nodes if evidence_nodes else None,
            evidence_card=parent_cards if evidence_nodes else None,
        )
        cpds_dbn.append(cpd_init) #list of TabularCPD objects

    # -------------------------------------------------
    # TRANSITION SLICE CPDs from *_t1
    # -------------------------------------------------
    for v in base_nodes:
        name_t1 = f"{v}_t1"
        try:
            cpd_t1 = model_2s.get_cpds(name_t1)
        except Exception:
            continue

        parents = list(cpd_t1.get_evidence())
        parent_cards = list(cpd_t1.cardinality[1:])

        evidence_nodes = []
        for p in parents:
            if p.endswith("_t"):
                evidence_nodes.append((p[:-2], 0))   # previous slice
            elif p.endswith("_t1"):
                evidence_nodes.append((p[:-3], 1))   # same slice (t+1)
            else:
                #evidence_nodes.append((p, 1))
                raise ValueError(f"Unexpected unsuffixed parent '{p}' found for transition CPD of {name_t1}")

        vals = np.array(cpd_t1.values, dtype=float)
        vals = vals.reshape(int(cpd_t1.variable_card), -1)

        cpd_trans = TabularCPD(
            variable=(v, 1),
            variable_card=int(cpd_t1.variable_card),
            values=vals,
            evidence=evidence_nodes if evidence_nodes else None,
            evidence_card=parent_cards if evidence_nodes else None,
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
    """
    Build a full first-order DBN over variables at times t and t+1
    by learning structure directly on the 2-slice data (df_2s),
    with constraints that forbid future->past and import intra-slice
    causal layering to both slices.

    Returns:
        model_2s: BayesianModel over {X_t, X_t1, ...}
        dbn:      DynamicBayesianNetwork with initial + transition slices
    """
    edges_path = DBN_EDGES_PATH[method]
    cpds_path  = DBN_CPDS_PATH[method]

    # base nodes (unsuffixed)
    nodes = list(df.columns)

    # build 2-slice dataset (t, t+1)
    df_t  = df.iloc[:-1].reset_index(drop=True).add_suffix("_t")
    df_t1 = df.iloc[1:].reset_index(drop=True).add_suffix("_t1")
    df_2s = pd.concat([df_t, df_t1], axis=1) #concatenate two or more pandas objects

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

    # build and fit 2-slice Bayesian network
    model_2s = BayesianModel(edges)
    model_2s.fit(
        df_2s,
        estimator=BayesianEstimator,
        prior_type="BDeu",
        equivalent_sample_size=10
    )

    # save static 2-slice representation
    pd.DataFrame(edges, columns=["from", "to"]).to_csv(edges_path, index=False)
    #save_cpds(model_2s, cpds_path)

    # build full Dynamic Bayesian Network template from 2-slice model
    dbn = _build_dynamic_dbn_from_2slice(model_2s, nodes)

    print("Full 2-slice BN + DBN template saved/built:", edges_path, cpds_path)
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
    # model_2s: static 2-slice BN
    # dbn:      full Dynamic Bayesian Network (initial + transition)
    #return model_2s, dbn

if __name__ == "__main__":
    main()

#if __name__ == "__main__":
#    model_2s, dbn = main()

#    print(dbn.nodes())
#    print(dbn.edges())

