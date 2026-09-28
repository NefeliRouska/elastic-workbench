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
    Discrete AIC score: maximize LL - k.
    Equivalent to minimizing AIC = -2LL + 2k.
    """

    def __init__(self, data):
        super().__init__(data)
        self.data = data
        self.state_names = {c: sorted(data[c].unique()) for c in data.columns}
        self.card = {c: len(self.state_names[c]) for c in data.columns}

    def local_score(self, variable, parents):
        df = self.data
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
                p = counts / total if total > 0 else np.zeros_like(counts, dtype=float)
                # CHANGE (bug fix): np.log(p, where=(p > 0)) without out= leaves
                # uninitialized/garbage memory at positions where p == 0, and
                # nansum does NOT catch that garbage (it only skips real NaNs).
                # Explicitly zero-init the output array so skipped positions are
                # a defined 0 (log of a zero-probability outcome contributes
                # nothing to the sum, which is the correct behavior).
                p_log = np.zeros_like(p)
                np.log(p, where=(p > 0), out=p_log)
                ll = np.sum(counts * p_log)

            return float(ll - (r_i - 1))

        group_cols = list(parents) + [variable]
        ct = df.groupby(group_cols).size().reset_index(name="n")
        pa_counts = ct.groupby(list(parents))["n"].sum().reset_index(name="n_pa")
        merged = ct.merge(pa_counts, on=list(parents), how="left")

        n = merged["n"].to_numpy(dtype=float)
        n_pa = merged["n_pa"].to_numpy(dtype=float)

        with np.errstate(divide="ignore", invalid="ignore"):
            frac = n / n_pa
            # CHANGE (bug fix): same issue as above, with-parents branch.
            frac_log = np.zeros_like(frac)
            np.log(frac, where=(frac > 0), out=frac_log)
            ll = np.sum(n * frac_log)

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
# BLACKLIST: DOMAIN CAUSAL ORDERING
# ============================================================
def build_blacklist(df):
    """
    Encodes domain causal ordering as forbidden edges.

    Layer 0: control parameters
      cores_, data_quality_

    Layer 1: container resource metrics
      container_cpu_, container_memory_, container_network_,
      container_fs_, container_blkio_

    Layer 2: performance outcomes
      throughput_, avg_p_latency_, buffer_size_

    Layer 3: failure/error indicators
      fail, oom, scrape_error

    Rules:
      - Layer 0 has no incoming edges.
      - Later layers cannot cause earlier layers.
      - Reverse inter-slice throughput edges are not blacklisted here,
        because they may represent backpressure.
    """
    all_vars = list(df.columns)

    def has_prefix(c, prefixes):
        c = c.lower()
        return any(c.startswith(p) for p in prefixes)

    def has_substring(c, subs):
        c = c.lower()
        return any(s in c for s in subs)

    layer0 = [
        v for v in all_vars
        if has_prefix(v, ["cores_", "data_quality_"])
    ]

    layer1 = [
        v for v in all_vars
        if has_prefix(
            v,
            [
                "container_cpu_",
                "container_memory_",
                "container_network_",
                "container_fs_",
                "container_blkio_",
            ],
        )
    ]

    layer2 = [
        v for v in all_vars
        if has_prefix(v, ["throughput_", "avg_p_latency_", "buffer_size_"])
    ]

    layer3 = [
        v for v in all_vars
        if has_substring(v, ["fail", "oom", "scrape_error"])
    ]

    layers = [layer0, layer1, layer2, layer3]
    layer_index = {v: i for i, layer in enumerate(layers) for v in layer}

    black = []

    # Layer 0 variables are exogenous: no incoming edges.
    for child in layer0:
        for parent in all_vars:
            if parent != child:
                black.append((parent, child))

    # Forbid later-layer variables causing earlier-layer variables.
    for parent in all_vars:
        for child in all_vars:
            if parent == child:
                continue

            if (
                parent in layer_index
                and child in layer_index
                and layer_index[parent] > layer_index[child]
            ):
                black.append((parent, child))

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
    """
    blacklist = build_blacklist(df)

    est = HillClimbSearch(df, use_cache=use_cache)
    best = est.estimate(
        scoring_method=make_score(score_name, df),
        max_indegree=max_indegree,
        black_list=blacklist,
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

    Forbid:
      - intra-t edges during inter-edge learning
      - intra-t1 edges during inter-edge learning
      - backward t1 -> t edges

    Self-transition edges X_t -> X_t1 are allowed.
    """
    black = []

    for a in nodes:
        for b in nodes:
            if a == b:
                continue

            black.append((f"{a}_t", f"{b}_t"))
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
    """
    Learn inter-slice edges X_t -> Y_t1 from shifted two-slice data.
    """
    nodes = list(df.columns)

    df_t = df.iloc[:-1].reset_index(drop=True).add_suffix("_t")
    df_t1 = df.iloc[1:].reset_index(drop=True).add_suffix("_t1")
    df_2s = pd.concat([df_t, df_t1], axis=1)

    blacklist = build_inter_only_blacklist(nodes)

    est = HillClimbSearch(df_2s, use_cache=use_cache)
    best = est.estimate(
        scoring_method=make_score(score_name, df_2s),
        max_indegree=max_indegree,
        black_list=blacklist,
        max_iter=max_iter,
        tabu_length=tabu_length,
        epsilon=epsilon,
        show_progress=False,
    )

    return [
        (u, v)
        for (u, v) in best.edges()
        if u.endswith("_t") and v.endswith("_t1")
    ]


# ============================================================
# CHANGE: GUARANTEE TARGET'S SELF-LOOP EDGE
# ============================================================
def ensure_self_loops(columns, inter_edges, target=None):
    """
    CHANGE: force X_t -> X_t1 for a specific variable (target), or for
    every variable if target is not given.

    Why: verification on the sweep results showed that the greedy
    inter-edge search frequently drops the direct self-transition
    edge for whichever variable ends up being TARGET downstream --
    e.g. throughput_3_t1 was sometimes only reachable via
    throughput_2_t/throughput_2_t1, with no throughput_3_t parent at
    all. That makes the model structurally unable to reproduce even
    the "predict same as last step" baseline for that variable, since
    it never sees its own last value directly. Separately confirmed
    on Sachs data (CHANGES_TO_APPLY.md #4/#7): HillClimbSearch reliably
    returns zero edges for hub-like nodes regardless of K or bin count
    -- this is a documented, reproducible weakness of this search
    setup, not a one-off quirk of the telemetry data.

    Why scoped to target rather than every variable: forcing a
    self-loop adds a parent to that variable's _t1 CPD, which shrinks
    the effective data per cell (CPD sparsity). Doing this for every
    variable pays that cost everywhere, even for variables nobody is
    predicting. Scoping to target pays the cost only where it buys
    something.

    target=None keeps this backward compatible: existing calls that
    don't pass target fall back to forcing every self-loop. Pass
    target=TARGET from the calling script to scope it -- this makes
    the fix work for whatever variable TARGET is set to (throughput_1,
    throughput_3, anything else) with no other changes needed.

    Self-loops are never blacklisted by build_inter_only_blacklist,
    so adding one here never conflicts with the existing search
    constraints -- this only adds an edge the search was already
    allowed to pick, for cases where it didn't.
    """
    existing = set(inter_edges)
    forced = []

    vars_to_force = [target] if target is not None else list(columns)

    for v in vars_to_force:
        edge = (f"{v}_t", f"{v}_t1")
        if edge not in existing:
            forced.append(edge)

    return inter_edges + forced


# ============================================================
# FIT 2-SLICE DBN
# ============================================================
def fit_consistent_2slice_bn(df, intra_edges, inter_edges):
    """
    Build and fit a two-slice DBN.

    Structure:
      1. intra_t  : X_t  -> Y_t
      2. intra_t1 : X_t1 -> Y_t1
      3. inter    : X_t  -> Y_t1

    Important:
      CPDs are fitted once from the full two-slice dataframe.
      We do NOT refit intra CPDs separately, because t1 nodes may have
      both intra-slice parents and inter-slice parents. Replacing their
      CPDs with intra-only CPDs would make the CPD parent sets inconsistent
      with the graph.
    """
    intra_t = [(f"{u}_t", f"{v}_t") for (u, v) in intra_edges]
    intra_t1 = [(f"{u}_t1", f"{v}_t1") for (u, v) in intra_edges]

    edges_2s = intra_t + intra_t1 + inter_edges

    df_t = df.iloc[:-1].reset_index(drop=True).add_suffix("_t")
    df_t1 = df.iloc[1:].reset_index(drop=True).add_suffix("_t1")
    df_2s = pd.concat([df_t, df_t1], axis=1)

    model_2s = BayesianNetwork(edges_2s)

    # Explicit state names prevent silent state-order or missing-bin issues.
    state_names_2s = {}
    for v in df.columns:
        max_val = int(df[v].max())
        states = list(range(max_val + 1))
        state_names_2s[f"{v}_t"] = states
        state_names_2s[f"{v}_t1"] = states

    model_2s.fit(
        df_2s,
        estimator=BayesianEstimator,
        prior_type="BDeu",
        equivalent_sample_size=10,
        state_names=state_names_2s,
    )

    return model_2s, edges_2s


# ============================================================
# MAIN BUILDER
# ============================================================
def build_dbn_model_2s(
    df_ready,
    score_name="bic",
    target=None,
    duration_col=None,
    max_indegree=MAX_INDEGREE,
    max_iter=HC_MAX_ITER,
    tabu_length=HC_TABU_LENGTH,
    epsilon=HC_EPSILON,
    use_cache=HC_USE_CACHE,
):
    # The counter is excluded from the within-slice search on purpose.
    intra_df = df_ready.drop(columns=[duration_col]) if duration_col is not None else df_ready

    intra = learn_intra_edges(
        intra_df,
        score_name=score_name,
        max_indegree=max_indegree,
        max_iter=max_iter,
        tabu_length=tabu_length,
        epsilon=epsilon,
        use_cache=use_cache,
    )

    inter = learn_inter_edges_only(
        df_ready,
        score_name=score_name,
        max_indegree=max_indegree,
        max_iter=max_iter,
        tabu_length=tabu_length,
        epsilon=epsilon,
        use_cache=use_cache,
    )

    inter = ensure_self_loops(df_ready.columns, inter, target=target)

    # Force counter_t -> target_t1
    if duration_col is not None and target is not None:
        forced = (f"{duration_col}_t", f"{target}_t1")
        if forced not in inter:
            inter = inter + [forced]

    model_2s, edges = fit_consistent_2slice_bn(df_ready, intra, inter)

    return model_2s, edges, intra, inter