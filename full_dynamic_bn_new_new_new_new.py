import numpy as np
import pandas as pd

from pgmpy.estimators import HillClimbSearch, BicScore, BayesianEstimator, StructureScore
from pgmpy.models import BayesianNetwork


# ----------------------------
# CONFIG DEFAULTS
# ----------------------------
MAX_INDEGREE   = 4
HC_MAX_ITER    = 25000
HC_TABU_LENGTH = 100
HC_EPSILON     = 1e-4
HC_USE_CACHE   = True


# ============================================================
# AIC SCORE
# ============================================================
class AicScoreCustom(StructureScore):
    """
    Discrete AIC score: maximize LL - k
    Equivalent to minimising AIC = -2LL + 2k.
    """

    def __init__(self, data):
        super().__init__(data)
        self.data        = data
        self.state_names = {c: sorted(data[c].unique()) for c in data.columns}
        self.card        = {c: len(self.state_names[c]) for c in data.columns}

    def local_score(self, variable, parents):
        df  = self.data
        r_i = self.card[variable]

        if not parents:
            counts = (
                df[variable]
                .value_counts()
                .reindex(self.state_names[variable], fill_value=0)
                .to_numpy()
            )
            total = counts.sum()
            with np.errstate(divide="ignore", invalid="ignore"):
                p  = counts / total if total > 0 else np.zeros_like(counts, dtype=float)
                ll = np.nansum(counts * np.log(p, where=(p > 0)))
            return float(ll - (r_i - 1))

        group_cols = list(parents) + [variable]
        ct        = df.groupby(group_cols).size().reset_index(name="n")
        pa_counts = ct.groupby(list(parents))["n"].sum().reset_index(name="n_pa")
        merged    = ct.merge(pa_counts, on=list(parents), how="left")

        n    = merged["n"].to_numpy(dtype=float)
        n_pa = merged["n_pa"].to_numpy(dtype=float)

        with np.errstate(divide="ignore", invalid="ignore"):
            frac = n / n_pa
            ll   = np.nansum(n * np.log(frac, where=(frac > 0)))

        q_i = 1
        for p in parents:
            q_i *= self.card[p]

        return float(ll - (r_i - 1) * q_i)


def make_score(score_name: str, df: pd.DataFrame):
    score_name = score_name.lower().strip()
    if score_name == "bic":
        return BicScore(df)
    if score_name == "aic":
        return AicScoreCustom(df)
    raise ValueError("score_name must be 'bic' or 'aic'")


# ============================================================
# BLACKLIST  (domain causal ordering)
# ============================================================
def build_blacklist(df):
    """
    Encodes the system's causal ordering as forbidden edges.

    Layer 0 — control parameters  (cores_, data_quality_):
        Set exogenously by the autoscaler. Nothing in the system
        causes them to change. No incoming edges allowed.

    Layer 1 — container resource metrics:
        Intermediate consequences of resource allocation.

    Layer 2 — performance outcomes
        (throughput_, avg_p_latency_, buffer_size_):
        Terminal effects of resource consumption.

    Layer 3 — failure / error indicators:
        Downstream consequences only.

    Justification: this ordering follows directly from the
    elastic-workbench architecture [Sedlak et al. 2025] and
    established performance-modelling conventions [Brosig et al. 2012].
    Encoding it as a blacklist is standard practice when domain
    knowledge is available [Koller & Friedman 2009,
    de Campos & Castellano 2007].
    """
    all_vars = list(df.columns)

    def has_prefix(c, prefixes):
        return any(c.lower().startswith(p) for p in prefixes)

    def has_substring(c, subs):
        return any(s in c.lower() for s in subs)

    layer0 = [v for v in all_vars if has_prefix(v, ["cores_", "data_quality_"])]
    layer1 = [v for v in all_vars if has_prefix(
        v, ["container_cpu_", "container_memory_", "container_network_",
            "container_fs_", "container_blkio_"])]
    layer2 = [v for v in all_vars if has_prefix(
        v, ["throughput_", "avg_p_latency_", "buffer_size_"])]
    layer3 = [v for v in all_vars if has_substring(
        v, ["fail", "oom", "scrape_error"])]

    layers      = [layer0, layer1, layer2, layer3]
    layer_index = {v: i for i, L in enumerate(layers) for v in L}

    black = []

    for child in layer0:
        for parent in all_vars:
            if parent != child:
                black.append((parent, child))

    for p in all_vars:
        for c in all_vars:
            if p == c:
                continue
            if (p in layer_index and c in layer_index
                    and layer_index[p] > layer_index[c]):
                black.append((p, c))

    return black


# ============================================================
# INTRA-SLICE EDGES
# ============================================================
def learn_intra_edges(
    df,
    score_name="bic",
    max_indegree=MAX_INDEGREE,
    max_iter=HC_MAX_ITER,
    tabu_length=HC_TABU_LENGTH,
    epsilon=HC_EPSILON,
    use_cache=HC_USE_CACHE,
):
    """
    Learn within-slice dependencies from single-slice data.
    Uses hill-climbing with the domain blacklist.

    Note: hill climbing is a greedy local search and may converge
    to a local optimum [Chickering 2002].
    """
    bl  = build_blacklist(df)
    est = HillClimbSearch(df, use_cache=use_cache)
    best = est.estimate(
        scoring_method=make_score(score_name, df),
        max_indegree=max_indegree,
        black_list=bl,
        max_iter=max_iter,
        tabu_length=tabu_length,
        epsilon=epsilon,
        show_progress=False,
    )
    return list(best.edges())


# ============================================================
# INTER-SLICE EDGES
# ============================================================
def build_inter_only_blacklist(nodes):
    """
    Only allow edges of the form X_t -> Y_t1.
    Block all intra-t, intra-t1, and backward t1->t edges.
    """
    black = []
    for a in nodes:
        for b in nodes:
            if a == b:
                continue
            black.append((f"{a}_t",  f"{b}_t"))
            black.append((f"{a}_t1", f"{b}_t1"))
            black.append((f"{a}_t1", f"{b}_t"))
    return black


def learn_inter_edges_only(
    df,
    score_name="bic",
    max_indegree=MAX_INDEGREE,
    max_iter=HC_MAX_ITER,
    tabu_length=HC_TABU_LENGTH,
    epsilon=HC_EPSILON,
    use_cache=HC_USE_CACHE,
):
    nodes = list(df.columns)
    df_t  = df.iloc[:-1].reset_index(drop=True).add_suffix("_t")
    df_t1 = df.iloc[1:].reset_index(drop=True).add_suffix("_t1")
    df_2s = pd.concat([df_t, df_t1], axis=1)

    black = build_inter_only_blacklist(nodes)
    est   = HillClimbSearch(df_2s, use_cache=use_cache)
    best  = est.estimate(
        scoring_method=make_score(score_name, df_2s),
        max_indegree=max_indegree,
        black_list=black,
        max_iter=max_iter,
        tabu_length=tabu_length,
        epsilon=epsilon,
        show_progress=False,
    )
    return [
        (u, v) for (u, v) in best.edges()
        if u.endswith("_t") and v.endswith("_t1")
    ]


# ============================================================
# FIT 2-SLICE DBN
# ============================================================
def fit_consistent_2slice_bn(df, intra_edges, inter_edges):
    """
    Build and fit the formal 2-slice DBN.

    Structure
    ---------
    A proper 2-slice DBN has three edge sets:
      1. intra_t  : X_t  -> Y_t   (within-slice at t)
      2. intra_t1 : X_t1 -> Y_t1  (same structure at t1, stationarity)
      3. inter    : X_t  -> Y_t1  (temporal transition dynamics)

    CPD fitting
    -----------
    Intra CPDs are re-fitted from single-slice data to avoid
    conflation with inter-slice parent signals.
    State space is validated before replacing any CPD to prevent
    the 'CPD does not have proper parents' crash.
    """
    intra_t  = [(f"{u}_t",  f"{v}_t")  for (u, v) in intra_edges]
    intra_t1 = [(f"{u}_t1", f"{v}_t1") for (u, v) in intra_edges]
    edges_2s = intra_t + intra_t1 + inter_edges

    df_t  = df.iloc[:-1].reset_index(drop=True).add_suffix("_t")
    df_t1 = df.iloc[1:].reset_index(drop=True).add_suffix("_t1")
    df_2s = pd.concat([df_t, df_t1], axis=1)

    df_single_t  = df.add_suffix("_t")
    df_single_t1 = df.add_suffix("_t1")

    model_2s = BayesianNetwork(edges_2s)

    # Step 1: fit everything from df_2s as baseline
    model_2s.fit(
        df_2s,
        estimator=BayesianEstimator,
        prior_type="BDeu",
        equivalent_sample_size=10,
    )

    # Step 2: re-fit intra nodes from single-slice data only
    intra_children_t  = list({v for (_, v) in intra_t})
    intra_children_t1 = list({v for (_, v) in intra_t1})

    def refit_intra_node(node, df_single, suffix, all_edges):
        parents_intra = [
            p for (p, c) in all_edges
            if c == node and p.endswith(suffix)
        ]
        if not parents_intra:
            return

        cols_available = [
            c for c in parents_intra + [node]
            if c in df_single.columns
        ]
        if node not in cols_available:
            return

        tiny_edges = [
            (p, node) for p in parents_intra
            if p in cols_available
        ]
        if not tiny_edges:
            return

        sub = df_single[cols_available].copy()

        try:
            existing_cpd = model_2s.get_cpds(node)
        except Exception:
            return

        if existing_cpd is None:
            return

        expected_states = set(existing_cpd.state_names[node])
        actual_states   = set(sub[node].unique())

        if not actual_states.issubset(expected_states):
            return

        tiny_model = BayesianNetwork(tiny_edges)
        try:
            tiny_model.fit(
                sub,
                estimator=BayesianEstimator,
                prior_type="BDeu",
                equivalent_sample_size=10,
            )
            new_cpd = tiny_model.get_cpds(node)

            if new_cpd is None:
                return
            if set(new_cpd.state_names[node]) != expected_states:
                return

            model_2s.cpds = [
                cpd for cpd in model_2s.cpds
                if cpd.variable != node
            ]
            model_2s.add_cpds(new_cpd)

        except Exception:
            pass

    for node in intra_children_t:
        refit_intra_node(node, df_single_t,  "_t",  edges_2s)
    for node in intra_children_t1:
        refit_intra_node(node, df_single_t1, "_t1", edges_2s)

    return model_2s, edges_2s


# ============================================================
# MAIN BUILDER
# ============================================================
def build_dbn_model_2s(
    df_ready,
    score_name="bic",
    max_indegree=MAX_INDEGREE,
    max_iter=HC_MAX_ITER,
    tabu_length=HC_TABU_LENGTH,
    epsilon=HC_EPSILON,
    use_cache=HC_USE_CACHE,
):
    intra = learn_intra_edges(
        df_ready, score_name, max_indegree, max_iter,
        tabu_length, epsilon, use_cache
    )
    inter = learn_inter_edges_only(
        df_ready, score_name, max_indegree, max_iter,
        tabu_length, epsilon, use_cache
    )
    model_2s, edges = fit_consistent_2slice_bn(df_ready, intra, inter)
    return model_2s, edges, intra, inter