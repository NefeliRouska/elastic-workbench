"""
final_joint_dbn_fixed.py
========================

Train and evaluate a single joint 2-slice DBN for all three throughput targets:
    throughput_1, throughput_2, throughput_3

Main fixes compared with the previous version:
1. Preserves temporal order before dropping timestamp columns.
2. Aggregates by run_tag + time windows, not by raw row number.
3. Prevents fake t -> t+1 transitions across different runs.
4. Performs multi-target feature selection and takes the union of predictors selected for all throughput targets.
5. Keeps all three throughput targets in the same joint model.
6. Uses a run-safe 2-slice DBN builder.
7. Evaluates each throughput target using only current-time evidence X_t, avoiding future leakage.

Usage:
    python final_joint_dbn_fixed.py
"""

import ast
import os
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.feature_selection import mutual_info_regression
from sklearn.cluster import DBSCAN

from pgmpy.estimators import HillClimbSearch, BayesianEstimator, StructureScore, BicScore
from pgmpy.inference import VariableElimination
from pgmpy.models import BayesianNetwork


# ============================================================
# CONFIG
# ============================================================

CSV_PATH = "share/metrics/dbn_wide_20260310_160246_seed161312.csv"

TRAIN_FRAC = 0.8
MODELING_GRANULARITY_SEC = 30

TARGETS = ["throughput_1", "throughput_2", "throughput_3"]

# Best hyperparameters from sweep
BEST_K = 8
BEST_FS = "markov"       # markov | mrmr | mi
BEST_DISC = "kmeans"     # classic_uniform | classic_quantile | kmeans | dbscan
BEST_N_BINS = 4
BEST_SCORE = "bic"       # bic | aic

MAX_INDEGREE = 4
HC_MAX_ITER = 25000
HC_TABU_LENGTH = 100
HC_EPSILON = 1e-4
HC_USE_CACHE = True

DBSCAN_EPS = 0.30
DBSCAN_MIN_SAMPLES = 10
MB_QUICK_MAX_ITER = 8000

MODEL_SAVE_PATH = "saved_dbn_models_top5_20260310_160246_seed161312/final_joint_dbn_fixed.pkl"
EDGES_SAVE_PATH = "saved_dbn_models_top5_20260310_160246_seed161312/final_joint_dbn_fixed_edges.csv"
RESULTS_SAVE_PATH = "final_joint_dbn_fixed_results.csv"
SELECTED_FEATURES_SAVE_PATH = "final_joint_dbn_fixed_selected_features.csv"


# ============================================================
# CUSTOM AIC SCORE
# ============================================================

class AicScoreCustom(StructureScore):
    """
    Discrete AIC-style score for pgmpy structure learning.
    Maximizes LL - k, equivalent to minimizing AIC = -2LL + 2k.
    """

    def __init__(self, data):
        super().__init__(data)
        self.data = data
        self.state_names = {c: sorted(data[c].dropna().unique()) for c in data.columns}
        self.card = {c: len(self.state_names[c]) for c in data.columns}

    def local_score(self, variable, parents):
        df = self.data
        r_i = self.card[variable]

        if r_i <= 1:
            return 0.0

        if not parents:
            counts = (
                df[variable]
                .value_counts()
                .reindex(self.state_names[variable], fill_value=0)
                .to_numpy(dtype=float)
            )
            total = counts.sum()
            if total <= 0:
                return 0.0
            p = counts / total
            ll = np.sum(counts[p > 0] * np.log(p[p > 0]))
            k = r_i - 1
            return float(ll - k)

        group_cols = list(parents) + [variable]
        ct = df.groupby(group_cols).size().reset_index(name="n")
        pa_counts = ct.groupby(list(parents))["n"].sum().reset_index(name="n_pa")
        merged = ct.merge(pa_counts, on=list(parents), how="left")

        n = merged["n"].to_numpy(dtype=float)
        n_pa = merged["n_pa"].to_numpy(dtype=float)
        frac = n / n_pa
        ll = np.sum(n[frac > 0] * np.log(frac[frac > 0]))

        q_i = 1
        for p in parents:
            q_i *= max(self.card[p], 1)

        k = (r_i - 1) * q_i
        return float(ll - k)


def make_score(score_name, df):
    score_name = score_name.lower().strip()
    if score_name == "bic":
        return BicScore(df)
    if score_name == "aic":
        return AicScoreCustom(df)
    raise ValueError("score_name must be 'bic' or 'aic'")


# ============================================================
# LOAD, SORT, CLEAN, AGGREGATE
# ============================================================

def load_sort_clean(path):
    """
    Loads wide DBN table.
    Important: sorts by run_tag/timestamp before dropping time columns.
    Returns:
        numeric_df: numeric modeling data
        meta_df: run_tag and timestamp/window metadata used for safe transitions
    """
    df = pd.read_csv(path)

    # Expand s_config if present.
    if "s_config" in df.columns:
        def parse_config(val):
            try:
                d = ast.literal_eval(val)
                return d if isinstance(d, dict) else {}
            except Exception:
                return {}

        cfg = df["s_config"].apply(parse_config).apply(pd.Series)
        cfg = cfg.rename(columns=lambda c: f"s_config_{c}")
        df = pd.concat([df.drop(columns=["s_config"]), cfg], axis=1)

    run_col = "run_tag" if "run_tag" in df.columns else None
    time_col = "timestamp" if "timestamp" in df.columns else None

    # Sort before dropping timestamp.
    if run_col and time_col:
        df = df.sort_values([run_col, time_col]).reset_index(drop=True)
    elif time_col:
        df = df.sort_values(time_col).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)

    # Keep metadata separately.
    meta = pd.DataFrame(index=df.index)
    if run_col:
        meta["run_tag"] = df[run_col].astype(str)
    else:
        meta["run_tag"] = "single_run"

    if time_col:
        meta["timestamp"] = pd.to_numeric(df[time_col], errors="coerce")
    else:
        meta["timestamp"] = np.arange(len(df), dtype=float)

    # Drop explicit time columns from modeling variables, but only after sorting.
    drop_cols = []
    for col in df.columns:
        if col == run_col:
            drop_cols.append(col)
        elif "time" in col.lower():
            drop_cols.append(col)

    df_model = df.drop(columns=drop_cols, errors="ignore")

    # Convert to numeric where possible.
    for col in df_model.columns:
        df_model[col] = pd.to_numeric(df_model[col], errors="coerce")

    df_model = df_model.select_dtypes(include=[np.number])

    # Align rows and remove rows with missing values.
    combined = pd.concat([meta.reset_index(drop=True), df_model.reset_index(drop=True)], axis=1)
    combined = combined.dropna().reset_index(drop=True)

    meta_clean = combined[["run_tag", "timestamp"]].copy()
    numeric_clean = combined.drop(columns=["run_tag", "timestamp"]).copy()

    return numeric_clean, meta_clean


def aggregate_by_run_and_time(df, meta, granularity_sec):
    """
    Aggregates observations into time windows per run.
    This avoids mixing different runs and preserves a valid temporal sequence.
    """
    tmp = pd.concat([meta.reset_index(drop=True), df.reset_index(drop=True)], axis=1)

    # If timestamps are real seconds, create time bins relative to start of each run.
    tmp["t0"] = tmp.groupby("run_tag")["timestamp"].transform("min")
    tmp["window"] = np.floor((tmp["timestamp"] - tmp["t0"]) / granularity_sec).astype(int)

    value_cols = [c for c in tmp.columns if c not in {"run_tag", "timestamp", "t0", "window"}]

    agg = (
        tmp.groupby(["run_tag", "window"], as_index=False)[value_cols]
        .mean()
        .sort_values(["run_tag", "window"])
        .reset_index(drop=True)
    )

    meta_agg = agg[["run_tag", "window"]].copy()
    df_agg = agg.drop(columns=["run_tag", "window"]).copy()

    return df_agg, meta_agg


def temporal_train_test_split(df, meta, train_frac):
    """
    Splits each run temporally, then concatenates train/test across runs.
    This avoids train/test mixing within local temporal order.
    """
    tmp = pd.concat([meta.reset_index(drop=True), df.reset_index(drop=True)], axis=1)
    train_parts = []
    test_parts = []

    for _, sub in tmp.groupby("run_tag", sort=False):
        sub = sub.sort_values("window").reset_index(drop=True)
        if len(sub) < 3:
            continue
        split = int(train_frac * len(sub))
        split = max(1, min(split, len(sub) - 1))
        train_parts.append(sub.iloc[:split].copy())
        test_parts.append(sub.iloc[split:].copy())

    if not train_parts or not test_parts:
        raise ValueError("Not enough data after per-run train/test split.")

    train = pd.concat(train_parts, ignore_index=True)
    test = pd.concat(test_parts, ignore_index=True)

    train_meta = train[["run_tag", "window"]].copy()
    test_meta = test[["run_tag", "window"]].copy()
    train_df = train.drop(columns=["run_tag", "window"]).reset_index(drop=True)
    test_df = test.drop(columns=["run_tag", "window"]).reset_index(drop=True)

    return train_df, train_meta, test_df, test_meta


# ============================================================
# DISCRETIZATION
# ============================================================

class Discretizer:
    def __init__(self, method, n_bins=4, dbscan_eps=0.3, dbscan_min_samples=10):
        self.method = method
        self.n_bins = n_bins
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples
        self.kbins = {}
        self.dbscan_centers = {}
        self.bin_centers_ = {}
        self.columns_ = None

    def fit(self, df):
        self.columns_ = list(df.columns)

        if self.method in {"classic_uniform", "classic_quantile", "kmeans"}:
            strategy = {
                "classic_uniform": "uniform",
                "classic_quantile": "quantile",
                "kmeans": "kmeans",
            }[self.method]

            for c in self.columns_:
                # Constant variables should already be removed, but keep this safe.
                if df[c].nunique() <= 1:
                    self.bin_centers_[c] = np.array([float(df[c].iloc[0])])
                    continue

                n_bins_eff = min(self.n_bins, df[c].nunique())
                kbd = KBinsDiscretizer(n_bins=n_bins_eff, encode="ordinal", strategy=strategy)
                kbd.fit(df[[c]])
                self.kbins[c] = kbd
                edges = kbd.bin_edges_[0]
                self.bin_centers_[c] = (edges[:-1] + edges[1:]) / 2

        elif self.method == "dbscan":
            for c in self.columns_:
                x = df[c].to_numpy(dtype=float).reshape(-1, 1)
                labels = DBSCAN(eps=self.dbscan_eps, min_samples=self.dbscan_min_samples).fit_predict(x)
                uniq = sorted(set(labels) - {-1})
                if len(uniq) == 0:
                    centers = np.array([float(np.mean(x))])
                else:
                    centers = np.sort(np.array([float(np.mean(x[labels == lab])) for lab in uniq]))
                self.dbscan_centers[c] = centers
                self.bin_centers_[c] = centers
        else:
            raise ValueError("Unknown discretization method.")

    def transform(self, df):
        out = df[self.columns_].copy()

        if self.method in {"classic_uniform", "classic_quantile", "kmeans"}:
            for c in self.columns_:
                if c in self.kbins:
                    out[c] = self.kbins[c].transform(out[[c]]).astype(int).flatten()
                else:
                    out[c] = 0
            return out.astype(int)

        for c in self.columns_:
            centers = self.dbscan_centers[c]
            x = out[c].to_numpy(dtype=float).reshape(-1, 1)
            out[c] = np.argmin(np.abs(x - centers.reshape(1, -1)), axis=1).astype(int)
        return out.astype(int)

    def get_bin_centers(self, col):
        return self.bin_centers_.get(col, None)


# ============================================================
# FEATURE SELECTION
# ============================================================

def drop_constant_cols(df):
    return df[[c for c in df.columns if df[c].nunique(dropna=True) > 1]].copy()


def split_X_y(df, target):
    X = df.drop(columns=[target]).copy()
    y = df[target].copy()
    return X, y


def top_k_by_mi(train_df, target, k):
    X, y = split_X_y(train_df, target)
    if X.shape[1] == 0:
        return []
    mi = mutual_info_regression(X.to_numpy(), y.to_numpy(), random_state=0)
    order = np.argsort(mi)[::-1]
    return [X.columns[i] for i in order[:min(k, len(order))]]


def mrmr_select(train_df, target, k):
    X, y = split_X_y(train_df, target)
    cols = list(X.columns)
    if not cols:
        return []

    Xv = X.to_numpy()
    rel = mutual_info_regression(Xv, y.to_numpy(), random_state=0)

    selected = [int(np.argmax(rel))]
    remaining = set(range(len(cols))) - {selected[0]}

    while len(selected) < min(k, len(cols)) and remaining:
        best_j, best_score = None, -1e18
        for j in remaining:
            redundancies = []
            for s in selected:
                red = mutual_info_regression(Xv[:, [j]], Xv[:, s], random_state=0)[0]
                redundancies.append(float(red))
            score = float(rel[j]) - float(np.mean(redundancies))
            if score > best_score:
                best_score, best_j = score, j
        selected.append(best_j)
        remaining.remove(best_j)

    return [cols[i] for i in selected]


def markov_blanket_select(train_raw, target, k, score_name, n_bins, disc_method):
    quick_method = "classic_uniform" if disc_method == "dbscan" else disc_method
    quick_disc = Discretizer(method=quick_method, n_bins=n_bins)

    train_small = drop_constant_cols(train_raw)
    if target not in train_small.columns:
        return top_k_by_mi(train_raw, target, k)

    quick_disc.fit(train_small)
    train_disc = quick_disc.transform(train_small)

    est = HillClimbSearch(train_disc)
    best = est.estimate(
        scoring_method=make_score(score_name, train_disc),
        max_iter=MB_QUICK_MAX_ITER,
        show_progress=False,
    )

    mb_model = BayesianNetwork(best.edges())
    try:
        mb = [c for c in mb_model.get_markov_blanket(target) if c != target]
    except Exception:
        mb = []

    if len(mb) > k:
        X_mb = train_raw[mb + [target]]
        mi = mutual_info_regression(
            X_mb.drop(columns=[target]).to_numpy(),
            X_mb[target].to_numpy(),
            random_state=0,
        )
        order = np.argsort(mi)[::-1]
        mb = [mb[i] for i in order[:k]]
    elif len(mb) < k:
        all_mi = top_k_by_mi(train_raw, target, len(train_raw.columns))
        extra = [c for c in all_mi if c not in mb and c != target]
        mb = (mb + extra)[:k]

    return mb


def select_features_for_target(train_raw, target, k, fs_method, score_name, n_bins, disc_method):
    fs_method = fs_method.lower().strip()

    if fs_method == "mi":
        return top_k_by_mi(train_raw, target, k)

    if fs_method == "mrmr":
        return mrmr_select(train_raw, target, k)

    if fs_method == "markov":
        return markov_blanket_select(train_raw, target, k, score_name, n_bins, disc_method)

    raise ValueError("fs_method must be one of: markov | mrmr | mi")


def select_joint_features(train_raw, targets, k, fs_method, score_name, n_bins, disc_method):
    """
    Multi-target feature selection:
    select K predictors per target, then take the union.
    Always include all targets.
    """
    selected_by_target = {}
    union_predictors = []

    for target in targets:
        predictors = select_features_for_target(
            train_raw=train_raw,
            target=target,
            k=k,
            fs_method=fs_method,
            score_name=score_name,
            n_bins=n_bins,
            disc_method=disc_method,
        )
        selected_by_target[target] = predictors
        for p in predictors:
            if p not in union_predictors and p not in targets:
                union_predictors.append(p)

    keep = union_predictors + list(targets)
    keep = [c for c in keep if c in train_raw.columns]

    return keep, selected_by_target


# ============================================================
# STRUCTURE LEARNING BLACKLISTS
# ============================================================

def build_intra_blacklist(df):
    all_vars = list(df.columns)

    def has_prefix(c, prefixes):
        c = c.lower()
        return any(c.startswith(p) for p in prefixes)

    def has_substring(c, subs):
        c = c.lower()
        return any(s in c for s in subs)

    layer0 = [v for v in all_vars if has_prefix(v, ["cores_", "data_quality_", "s_config_"])]
    layer1 = [v for v in all_vars if has_prefix(v, [
        "container_cpu_", "container_memory_", "container_network_",
        "container_fs_", "container_blkio_",
    ])]
    layer2 = [v for v in all_vars if has_prefix(v, ["throughput_", "avg_p_latency_", "buffer_size_"])]
    layer3 = [v for v in all_vars if has_substring(v, ["fail", "oom", "scrape_error"])]

    layers = [layer0, layer1, layer2, layer3]
    layer_index = {v: i for i, layer in enumerate(layers) for v in layer}

    black = []

    # Configuration/control variables should not have parents.
    for child in layer0:
        for parent in all_vars:
            if parent != child:
                black.append((parent, child))

    # Forbid later causal layers from pointing back to earlier ones.
    for p in all_vars:
        for c in all_vars:
            if p == c:
                continue
            if p in layer_index and c in layer_index and layer_index[p] > layer_index[c]:
                black.append((p, c))

    return black


def learn_intra_edges(df, score_name="bic"):
    bl = build_intra_blacklist(df)
    est = HillClimbSearch(df, use_cache=HC_USE_CACHE)
    best = est.estimate(
        scoring_method=make_score(score_name, df),
        max_indegree=MAX_INDEGREE,
        black_list=bl,
        max_iter=HC_MAX_ITER,
        tabu_length=HC_TABU_LENGTH,
        epsilon=HC_EPSILON,
        show_progress=False,
    )
    return list(best.edges())


def build_inter_only_blacklist(nodes):
    black = []
    for a in nodes:
        for b in nodes:
            if a == b:
                continue
            # No same-slice learning in inter phase.
            black.append((f"{a}_t", f"{b}_t"))
            black.append((f"{a}_t1", f"{b}_t1"))
            # No backward temporal edges.
            black.append((f"{a}_t1", f"{b}_t"))
    return black


def make_two_slice_data_by_run(df, meta):
    """
    Creates valid row pairs X_t, X_t1 only within the same run.
    Prevents fake transitions across run boundaries.
    """
    tmp = pd.concat([meta.reset_index(drop=True), df.reset_index(drop=True)], axis=1)
    pieces = []

    for _, sub in tmp.groupby("run_tag", sort=False):
        sub = sub.sort_values("window").reset_index(drop=True)
        if len(sub) < 2:
            continue

        value_df = sub.drop(columns=["run_tag", "window"])
        df_t = value_df.iloc[:-1].reset_index(drop=True).add_suffix("_t")
        df_t1 = value_df.iloc[1:].reset_index(drop=True).add_suffix("_t1")
        pieces.append(pd.concat([df_t, df_t1], axis=1))

    if not pieces:
        raise ValueError("No valid within-run temporal pairs available.")

    return pd.concat(pieces, ignore_index=True)


def learn_inter_edges_only(df, meta, score_name="bic"):
    nodes = list(df.columns)
    df_2s = make_two_slice_data_by_run(df, meta)
    black = build_inter_only_blacklist(nodes)

    est = HillClimbSearch(df_2s, use_cache=HC_USE_CACHE)
    best = est.estimate(
        scoring_method=make_score(score_name, df_2s),
        max_indegree=MAX_INDEGREE,
        black_list=black,
        max_iter=HC_MAX_ITER,
        tabu_length=HC_TABU_LENGTH,
        epsilon=HC_EPSILON,
        show_progress=False,
    )

    return [(u, v) for (u, v) in best.edges() if u.endswith("_t") and v.endswith("_t1")]


def fit_consistent_2slice_bn(df, meta, intra_edges, inter_edges):
    intra_t = [(f"{u}_t", f"{v}_t") for (u, v) in intra_edges]
    intra_t1 = [(f"{u}_t1", f"{v}_t1") for (u, v) in intra_edges]
    edges_2s = intra_t + intra_t1 + inter_edges

    df_2s = make_two_slice_data_by_run(df, meta)

    # Ensure nodes with no edges are still included.
    all_nodes = list(df_2s.columns)
    model_2s = BayesianNetwork(edges_2s)
    model_2s.add_nodes_from(all_nodes)

    model_2s.fit(
        df_2s,
        estimator=BayesianEstimator,
        prior_type="BDeu",
        equivalent_sample_size=10,
    )

    return model_2s, edges_2s


def build_dbn_model_2s_run_safe(df_ready, meta_ready, score_name="bic"):
    intra = learn_intra_edges(df_ready, score_name=score_name)
    inter = learn_inter_edges_only(df_ready, meta_ready, score_name=score_name)
    model_2s, edges = fit_consistent_2slice_bn(df_ready, meta_ready, intra, inter)
    return model_2s, edges, intra, inter


# ============================================================
# EVALUATION
# ============================================================

def evaluate_target(model_2s, test_ready, test_meta, target, bin_centers):
    """
    Evaluates P(target_t1 | X_t) using valid within-run transitions only.
    Evidence contains current-time variables only.
    """
    infer = VariableElimination(model_2s)
    model_nodes = set(model_2s.nodes())
    target_t1 = f"{target}_t1"

    if target_t1 not in model_nodes:
        return {
            "accuracy": float("nan"),
            "mean_prob_true": float("nan"),
            "mae": float("nan"),
            "n_queries": 0,
            "cache_size": 0,
        }

    tmp = pd.concat([test_meta.reset_index(drop=True), test_ready.reset_index(drop=True)], axis=1)
    target_centers = bin_centers.get(target, None)

    correct = 0
    probs_true = []
    abs_errors = []
    n_queries = 0
    inference_cache = {}

    for _, sub in tmp.groupby("run_tag", sort=False):
        sub = sub.sort_values("window").reset_index(drop=True)
        if len(sub) < 2:
            continue

        values = sub.drop(columns=["run_tag", "window"])
        ev_cols = [c for c in values.columns if c != target and f"{c}_t" in model_nodes]

        for i in range(len(values) - 1):
            evidence = {f"{c}_t": int(values.iloc[i][c]) for c in ev_cols}
            true_next = int(values.iloc[i + 1][target])
            evidence_key = tuple(sorted(evidence.items()))

            if evidence_key in inference_cache:
                probs = inference_cache[evidence_key]
            else:
                q = infer.query([target_t1], evidence=evidence, show_progress=False)
                probs = q.values / q.values.sum()
                inference_cache[evidence_key] = probs

            pred = int(np.argmax(probs))
            correct += int(pred == true_next)
            n_queries += 1

            if true_next < len(probs):
                probs_true.append(float(probs[true_next]))
            else:
                probs_true.append(0.0)

            if target_centers is not None:
                pred_val = float(target_centers[pred]) if pred < len(target_centers) else float(pred)
                true_val = float(target_centers[true_next]) if true_next < len(target_centers) else float(true_next)
                abs_errors.append(abs(pred_val - true_val))

    if n_queries == 0:
        return {
            "accuracy": float("nan"),
            "mean_prob_true": float("nan"),
            "mae": float("nan"),
            "n_queries": 0,
            "cache_size": len(inference_cache),
        }

    return {
        "accuracy": correct / n_queries,
        "mean_prob_true": float(np.mean(probs_true)),
        "mae": float(np.mean(abs_errors)) if abs_errors else float("nan"),
        "n_queries": n_queries,
        "cache_size": len(inference_cache),
    }


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("FINAL JOINT DBN — FIXED")
    print("=" * 70)
    print(f"CSV:       {CSV_PATH}")
    print(f"Targets:   {TARGETS}")
    print(f"K:         {BEST_K}")
    print(f"FS:        {BEST_FS}")
    print(f"Disc:      {BEST_DISC}")
    print(f"n_bins:    {BEST_N_BINS}")
    print(f"Score:     {BEST_SCORE}")
    print()

    raw, meta = load_sort_clean(CSV_PATH)
    print(f"Raw cleaned shape: {raw.shape}")

    for target in TARGETS:
        if target not in raw.columns:
            raise ValueError(f"Target {target} not found in data columns.")

    raw = drop_constant_cols(raw)
    for target in TARGETS:
        if target not in raw.columns:
            raise ValueError(f"Target {target} became constant or missing after cleaning.")

    agg, meta_agg = aggregate_by_run_and_time(raw, meta, MODELING_GRANULARITY_SEC)
    agg = drop_constant_cols(agg)
    meta_agg = meta_agg.loc[agg.index].reset_index(drop=True)
    agg = agg.reset_index(drop=True)

    print(f"Aggregated shape: {agg.shape}")
    print(f"Number of runs:   {meta_agg['run_tag'].nunique()}")
    print()

    for target in TARGETS:
        if target not in agg.columns:
            raise ValueError(f"Target {target} not found after aggregation/constant-column removal.")

    train_raw, train_meta, test_raw, test_meta = temporal_train_test_split(agg, meta_agg, TRAIN_FRAC)
    print(f"Train raw shape: {train_raw.shape}")
    print(f"Test raw shape:  {test_raw.shape}")
    print()

    print("Running multi-target feature selection...")
    keep_cols, selected_by_target = select_joint_features(
        train_raw=train_raw,
        targets=TARGETS,
        k=BEST_K,
        fs_method=BEST_FS,
        score_name=BEST_SCORE,
        n_bins=BEST_N_BINS,
        disc_method=BEST_DISC,
    )

    train_fs = train_raw[keep_cols].copy()
    test_fs = test_raw[keep_cols].copy()

    print("Selected predictors by target:")
    for target, predictors in selected_by_target.items():
        print(f"  {target}: {predictors}")
    print()
    print(f"Joint feature set ({len(keep_cols)} variables):")
    print(keep_cols)
    print()

    pd.DataFrame([
        {"target": t, "selected_predictors": repr(preds)}
        for t, preds in selected_by_target.items()
    ]).to_csv(SELECTED_FEATURES_SAVE_PATH, index=False)

    train_fs = drop_constant_cols(train_fs)
    test_fs = test_fs[train_fs.columns]

    for target in TARGETS:
        if target not in train_fs.columns:
            raise RuntimeError(f"{target} missing after feature selection/constant-column removal.")

    print("Discretizing...")
    disc = Discretizer(
        method=BEST_DISC,
        n_bins=BEST_N_BINS,
        dbscan_eps=DBSCAN_EPS,
        dbscan_min_samples=DBSCAN_MIN_SAMPLES,
    )
    disc.fit(train_fs)
    train_ready = disc.transform(train_fs)
    test_ready = disc.transform(test_fs)

    train_ready = drop_constant_cols(train_ready)
    test_ready = test_ready[train_ready.columns]

    for target in TARGETS:
        if target not in train_ready.columns:
            raise RuntimeError(f"{target} missing after discretization/constant-column removal.")

    bin_centers = {c: disc.get_bin_centers(c) for c in train_fs.columns}

    print(f"Train ready shape: {train_ready.shape}")
    print(f"Test ready shape:  {test_ready.shape}")
    print(f"Variables: {list(train_ready.columns)}")
    print()

    print("Training run-safe joint 2-slice DBN...")
    t0 = time.perf_counter()
    model_2s, edges, intra, inter = build_dbn_model_2s_run_safe(
        train_ready,
        train_meta.reset_index(drop=True),
        score_name=BEST_SCORE,
    )
    t1 = time.perf_counter()
    print(f"Training time: {t1 - t0:.2f}s")
    print()

    print("Learned temporal edges:")
    for u, v in inter:
        print(f"  {u} -> {v}")
    print()

    print("Learned intra-slice edges:")
    for u, v in intra:
        print(f"  {u} -> {v}")
    print()

    print("Evaluating all throughput targets...")
    results = []

    for target in TARGETS:
        print(f"Target: {target}")
        te0 = time.perf_counter()
        res = evaluate_target(model_2s, test_ready, test_meta.reset_index(drop=True), target, bin_centers)
        te1 = time.perf_counter()

        print(f"  Accuracy:       {res['accuracy']:.3f}")
        print(f"  Mean prob true: {res['mean_prob_true']:.3f}")
        print(f"  MAE:            {res['mae']:.3f}")
        print(f"  Queries:        {res['n_queries']}")
        print(f"  Cache size:     {res['cache_size']}")
        print(f"  Eval time:      {te1 - te0:.2f}s")
        print()

        results.append({
            "target": target,
            "accuracy": res["accuracy"],
            "mean_prob_true": res["mean_prob_true"],
            "mae": res["mae"],
            "n_queries": res["n_queries"],
            "cache_size": res["cache_size"],
            "K": BEST_K,
            "fs": BEST_FS,
            "disc": BEST_DISC,
            "n_bins": BEST_N_BINS,
            "score": BEST_SCORE,
            "n_train": len(train_ready),
            "n_test": len(test_ready),
            "n_variables": len(train_ready.columns),
            "n_inter_edges": len(inter),
            "n_intra_edges": len(intra),
            "training_time_sec": t1 - t0,
        })

    pd.DataFrame(results).to_csv(RESULTS_SAVE_PATH, index=False)
    print(f"Results saved to: {RESULTS_SAVE_PATH}")

    save_dir = os.path.dirname(MODEL_SAVE_PATH)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    with open(MODEL_SAVE_PATH, "wb") as f:
        pickle.dump({
            "model": model_2s,
            "discretizer": disc,
            "bin_centers": bin_centers,
            "targets": TARGETS,
            "variables": list(train_ready.columns),
            "selected_by_target": selected_by_target,
            "config": {
                "K": BEST_K,
                "fs": BEST_FS,
                "disc": BEST_DISC,
                "n_bins": BEST_N_BINS,
                "score": BEST_SCORE,
                "granularity_sec": MODELING_GRANULARITY_SEC,
            },
        }, f)

    with open(MODEL_SAVE_PATH, "rb") as f:
        pickle.load(f)

    edges_df = pd.DataFrame(list(model_2s.edges()), columns=["parent", "child"])
    edges_df.to_csv(EDGES_SAVE_PATH, index=False)

    print(f"Model saved to:   {MODEL_SAVE_PATH}")
    print(f"Edges saved to:   {EDGES_SAVE_PATH}")
    print(f"Features saved to:{SELECTED_FEATURES_SAVE_PATH}")
    print()

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for r in results:
        print(
            f"{r['target']:15s} "
            f"acc={r['accuracy']:.3f}  "
            f"prob={r['mean_prob_true']:.3f}  "
            f"mae={r['mae']:.3f}"
        )


if __name__ == "__main__":
    main()
