import numpy as np
import pandas as pd

from pgmpy.estimators import HillClimbSearch, BicScore, BayesianEstimator, StructureScore
from pgmpy.models import BayesianNetwork


# ----------------------------
# CONFIG DEFAULTS
# ----------------------------
MAX_INDEGREE = 4 #no node can have more than 4 parents.
HC_MAX_ITER = 25000 #the hill-climb search can take up to 25,000 steps (moves)
HC_TABU_LENGTH = 100 #hill-climb uses a tabu list (memory of recent moves) to reduce cycling and local traps.
HC_EPSILON = 1e-4 #stopping threshold: if improvements are smaller than this, search can stop
HC_USE_CACHE = True #caches score computations to speed up repeated evaluations


# ----------------------------
# AIC score that pgmpy accepts
# (must subclass StructureScore in old pgmpy)
# ----------------------------
class AicScoreCustom(StructureScore):
    """
    Discrete AIC score: maximize LL - k
    Equivalent to minimizing AIC = -2LL + 2k.
    """

    def __init__(self, data):
        super().__init__(data)
        self.data = data
        self.state_names = {c: sorted(data[c].unique()) for c in data.columns}
        self.card = {c: len(self.state_names[c]) for c in data.columns}

    def local_score(self, variable, parents): #Structure scoring in pgmpy is “decomposable”: the total score of a graph can be computed as a sum of local scores per node
        df = self.data
        r_i = self.card[variable]

        if not parents:
            counts = df[variable].value_counts().reindex(self.state_names[variable], fill_value=0).to_numpy()
            total = counts.sum()

            with np.errstate(divide="ignore", invalid="ignore"):
                p = counts / total if total > 0 else np.zeros_like(counts, dtype=float)
                ll = np.nansum(counts * np.log(p, where=(p > 0)))

            k = (r_i - 1) * 1 #r_i is number of states of the variable
            return float(ll - k)

        group_cols = list(parents) + [variable]
        ct = df.groupby(group_cols).size().reset_index(name="n")
        pa_counts = ct.groupby(list(parents))["n"].sum().reset_index(name="n_pa") #for each parent configuration, count how many samples have that parent configuration
        merged = ct.merge(pa_counts, on=list(parents), how="left") #adds the parent totals n_pa back to each row where we have specific (parents, variable)

        n = merged["n"].to_numpy(dtype=float)
        n_pa = merged["n_pa"].to_numpy(dtype=float)

        with np.errstate(divide="ignore", invalid="ignore"):
            frac = n / n_pa
            ll = np.nansum(n * np.log(frac, where=(frac > 0)))

        q_i = 1 #number of possible parent configurations
        for p in parents:
            q_i *= self.card[p] #product of cardinalities of each parent.

        k = (r_i - 1) * q_i
        return float(ll - k)


def make_score(score_name: str, df: pd.DataFrame): #chooses which scoring object to use
    score_name = score_name.lower().strip()
    if score_name == "bic":
        return BicScore(df)
    if score_name == "aic":
        return AicScoreCustom(df)
    raise ValueError("score_name must be 'bic' or 'aic'")


# ============================================================
# BLACKLIST
# ============================================================
def build_blacklist(df):
    all_vars = list(df.columns)

    def has_prefix(c, prefixes):
        c = c.lower()
        return any(c.startswith(p) for p in prefixes)

    def has_substring(c, subs):
        c = c.lower()
        return any(s in c for s in subs)

    layer0 = [v for v in all_vars if has_prefix(
        v, ["cores_", "data_quality_"])]

    layer1 = [v for v in all_vars if has_prefix(
        v, ["container_cpu_", "container_memory_", "container_network_", "container_fs_", "container_blkio_"])]

    layer2 = [v for v in all_vars if has_prefix(
        v,["throughput_", "avg_p_latency_", "buffer_size_"])]

    layer3 = [v for v in all_vars if has_substring(
        v, ["fail", "oom", "scrape_error"])]

    layers = [layer0, layer1, layer2, layer3]
    layer_index = {v: i for i, L in enumerate(layers) for v in L}

    black = []

    # layer0 has no parents
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
def learn_intra_edges(df, score_name="bic",
                     max_indegree=MAX_INDEGREE, max_iter=HC_MAX_ITER,
                     tabu_length=HC_TABU_LENGTH, epsilon=HC_EPSILON,
                     use_cache=HC_USE_CACHE):

    bl = build_blacklist(df)

    est = HillClimbSearch(df, use_cache=use_cache)
    best = est.estimate(
        scoring_method=make_score(score_name, df),
        max_indegree=max_indegree,
        black_list=bl,
        max_iter=max_iter,
        tabu_length=tabu_length,
        epsilon=epsilon,
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


def learn_inter_edges_only(df, score_name="bic",
                          max_indegree=MAX_INDEGREE, max_iter=HC_MAX_ITER,
                          tabu_length=HC_TABU_LENGTH, epsilon=HC_EPSILON,
                          use_cache=HC_USE_CACHE):

    nodes = list(df.columns)

    df_t = df.iloc[:-1].reset_index(drop=True).add_suffix("_t")
    df_t1 = df.iloc[1:].reset_index(drop=True).add_suffix("_t1")
    df_2s = pd.concat([df_t, df_t1], axis=1)

    black = build_inter_only_blacklist(nodes)

    est = HillClimbSearch(df_2s, use_cache=use_cache)
    best = est.estimate(
        scoring_method=make_score(score_name, df_2s),
        max_indegree=max_indegree,
        black_list=black,
        max_iter=max_iter,
        tabu_length=tabu_length,
        epsilon=epsilon,
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
def build_dbn_model_2s(df_ready, score_name="bic",
                      max_indegree=MAX_INDEGREE, max_iter=HC_MAX_ITER,
                      tabu_length=HC_TABU_LENGTH, epsilon=HC_EPSILON,
                      use_cache=HC_USE_CACHE):

    intra = learn_intra_edges(df_ready, score_name, max_indegree, max_iter, tabu_length, epsilon, use_cache)
    inter = learn_inter_edges_only(df_ready, score_name, max_indegree, max_iter, tabu_length, epsilon, use_cache)
    model_2s, edges = fit_consistent_2slice_bn(df_ready, intra, inter)

    return model_2s, edges, intra, inter
