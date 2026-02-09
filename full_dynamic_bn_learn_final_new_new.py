import pandas as pd
from pgmpy.estimators import HillClimbSearch, BicScore, BayesianEstimator
from pgmpy.models import BayesianNetwork

MAX_INDEGREE = 4
HC_MAX_ITER = 25000


# ============================================================
# BLACKLIST (your original logic)
# ============================================================
def build_blacklist(df):

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

    # layer0 no parents
    for child in layer0:
        for parent in all_vars:
            if parent != child:
                black.append((parent, child))

    # forbid later -> earlier
    for p in all_vars:
        for c in all_vars:
            if p == c:
                continue
            if p in layer_index and c in layer_index and layer_index[p] > layer_index[c]:
                black.append((p, c))

    return black


# ============================================================
# INTRA EDGES
# ============================================================
def learn_intra_edges(df):

    bl = build_blacklist(df)

    est = HillClimbSearch(df)
    best = est.estimate(
        scoring_method=BicScore(df),
        max_indegree=MAX_INDEGREE,
        black_list=bl,
        max_iter=HC_MAX_ITER,
        show_progress=False
    )

    return list(best.edges())


# ============================================================
# INTER EDGES
# ============================================================
def build_inter_only_blacklist(nodes):

    black = []

    for a in nodes:
        for b in nodes:
            if a == b:
                continue

            black.append((f"{a}_t", f"{b}_t"))
            black.append((f"{a}_t1", f"{b}_t1"))
            black.append((f"{a}_t1", f"{b}_t"))

    return black


def learn_inter_edges_only(df):

    nodes = list(df.columns)

    df_t = df.iloc[:-1].reset_index(drop=True).add_suffix("_t")
    df_t1 = df.iloc[1:].reset_index(drop=True).add_suffix("_t1")
    df_2s = pd.concat([df_t, df_t1], axis=1)

    black = build_inter_only_blacklist(nodes)

    est = HillClimbSearch(df_2s)
    best = est.estimate(
        scoring_method=BicScore(df_2s),
        max_indegree=MAX_INDEGREE,
        black_list=black,
        max_iter=HC_MAX_ITER,
        show_progress=False
    )

    return [(u, v) for (u, v) in best.edges()
            if u.endswith("_t") and v.endswith("_t1")]


# ============================================================
# FIT 2-SLICE BN
# ============================================================
def fit_consistent_2slice_bn(df, intra_edges, inter_edges):

    intra_t = [(f"{u}_t", f"{v}_t") for (u, v) in intra_edges]
    intra_t1 = [(f"{u}_t1", f"{v}_t1") for (u, v) in intra_edges]

    edges_2s = intra_t + intra_t1 + inter_edges

    df_t = df.iloc[:-1].reset_index(drop=True).add_suffix("_t")
    df_t1 = df.iloc[1:].reset_index(drop=True).add_suffix("_t1")
    df_2s = pd.concat([df_t, df_t1], axis=1)

    model_2s = BayesianNetwork(edges_2s)

    model_2s.fit(
        df_2s,
        estimator=BayesianEstimator,
        prior_type="BDeu",
        equivalent_sample_size=10
    )

    return model_2s, edges_2s


# ============================================================
# MAIN BUILDER
# ============================================================
def build_dbn_model_2s(df_ready):

    intra = learn_intra_edges(df_ready)
    inter = learn_inter_edges_only(df_ready)
    model_2s, edges = fit_consistent_2slice_bn(df_ready, intra, inter)

    return model_2s, edges, intra, inter
