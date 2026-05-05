import ast
import os
import pickle
import time
import gc
import heapq
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression
from sklearn.cluster import DBSCAN
from pgmpy.inference import VariableElimination
from pgmpy.estimators import HillClimbSearch, BayesianEstimator
from pgmpy.models import BayesianNetwork
from full_dynamic_bn_new_new_new import build_dbn_model_2s, make_score
from pathlib import Path
import argparse


# ============================================================
# CONFIG
# ============================================================
# --- Single CSV (currently active) ---
#CSV_PATH = "share/metrics/dbn_wide_20260310_160246_seed161312.csv"
def parse_args():
    parser = argparse.ArgumentParser(
        description="DBN hyperparameter sweep"
    )
    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Path to wide DBN CSV file"
    )
    return parser.parse_args()

args = parse_args()
CSV_PATH = args.csv

TRAIN_FRAC = 0.8
#TARGETS    = ["throughput_1", "throughput_2", "throughput_3"]
TARGETS    = ["throughput_3"]

K_VALUES                  = [8, 12, 20]
FEATURE_SELECTION_METHODS = ["markov", "mrmr", "pca"]
DISCRETIZATION_METHODS    = ["classic_uniform", "classic_quantile", "kmeans", "dbscan"]
SCORES                    = ["bic", "aic"]

N_BINS_VALUES      = [2, 3, 4, 5, 6, 8]
DBSCAN_EPS         = 0.30
DBSCAN_MIN_SAMPLES = 10

MB_QUICK_MAX_ITER = 8000

# Evidence modes: "full" | "full_no_target" | "subset"
EVIDENCE_MODE   = "full_no_target"
EVIDENCE_SUBSET = ["avg_p_latency_", "cores_", "data_quality_", "buffer_size_"]

EXCLUDE_OTHER_THROUGHPUTS = False

TOP_N_MODELS_TO_SAVE = 5

_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
_stem = Path(CSV_PATH).stem.replace("dbn_wide_", "") + f"_run{_timestamp}"

MODEL_SAVE_DIR   = f"saved_dbn_models_top5_{_stem}"
RESULTS_CSV_PATH = f"dbn_k_sweep_results_{_stem}.csv"
#MODEL_SAVE_DIR       = "saved_dbn_models_top5_20260310_160246_seed161312"
#RESULTS_CSV_PATH     = "dbn_k_sweep_results_20260310_160246_seed161312.csv"

# Temporal aggregation:
# Raw data is 1-second resolution. We average every
# MODELING_GRANULARITY_SEC rows into one snapshot so that
# consecutive rows reflect meaningful state transitions and
# 1-second noise is smoothed out.
MODELING_GRANULARITY_SEC = 30

MAX_ROWS = None  # set to integer for debug runs


# ============================================================
# LOAD + CLEAN
# ============================================================
def load_and_clean(path, target):
    """
    Load the raw wide CSV and clean it.
    Returns dataframe at original 1-second resolution.
    Call aggregate_to_modeling_granularity() after this.
    """
    df = pd.read_csv(path)

    if "s_config" in df.columns:
        def parse(val):
            try:
                d = ast.literal_eval(val)
                return d if isinstance(d, dict) else {}
            except Exception:
                return {}
        cfg = df["s_config"].apply(parse).apply(pd.Series)
        cfg = cfg.rename(columns=lambda c: f"s_config_{c}")
        df  = pd.concat([df.drop(columns=["s_config"]), cfg], axis=1)

    for col in list(df.columns):
        if "time" in col.lower():
            df.drop(columns=[col], inplace=True)

    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except Exception:
            pass

    df = df.select_dtypes(include=[np.number])
    df = df.dropna().reset_index(drop=True)

    if target not in df.columns:
        raise ValueError(f"TARGET '{target}' not found after cleaning.")

    return df


# ============================================================
# TEMPORAL AGGREGATION
# ============================================================
def aggregate_to_modeling_granularity(df, every_n_seconds=MODELING_GRANULARITY_SEC):
    """
    Aggregate from 1-second to N-second windows by averaging.

    Kept separate from load_and_clean() because this is a
    modeling decision, not a data cleaning decision.

    Output rows: floor(original_rows / every_n_seconds)
    """
    group_ids = np.arange(len(df)) // every_n_seconds
    return df.groupby(group_ids).mean().reset_index(drop=True)


# ============================================================
# DISCRETIZATION
# ============================================================
class Discretizer:
    """
    Fits on train only; transforms train/test.

    Methods:
      - classic_uniform  : KBins strategy="uniform"
      - classic_quantile : KBins strategy="quantile"
      - kmeans           : KBins strategy="kmeans"
      - dbscan           : density clusters, n_bins ignored

    Also stores bin centers for MAE computation in real units.
    """

    def __init__(self, method, n_bins=4, dbscan_eps=0.3, dbscan_min_samples=10):
        self.method             = method
        self.n_bins             = n_bins
        self.dbscan_eps         = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples
        self.kbins              = {}
        self.dbscan_centers     = {}
        self.bin_centers_       = {}  # stores bin centers for MAE
        self.columns_           = None

        valid = {"classic_uniform", "classic_quantile", "kmeans", "dbscan"}
        if self.method not in valid:
            raise ValueError(f"Unknown discretizer method: {self.method}")

    def fit(self, df):
        self.columns_ = list(df.columns)

        if self.method in {"classic_uniform", "classic_quantile", "kmeans"}:
            strategy = {
                "classic_uniform":  "uniform",
                "classic_quantile": "quantile",
                "kmeans":           "kmeans",
            }[self.method]

            for c in self.columns_:
                kbd = KBinsDiscretizer(
                    n_bins=self.n_bins, encode="ordinal", strategy=strategy
                )
                kbd.fit(df[[c]])
                self.kbins[c] = kbd

                # Store bin centers as midpoints between edges
                edges = kbd.bin_edges_[0]
                self.bin_centers_[c] = (edges[:-1] + edges[1:]) / 2

        else:  # dbscan
            for c in self.columns_:
                x      = df[c].to_numpy(dtype=float).reshape(-1, 1)
                labels = DBSCAN(
                    eps=self.dbscan_eps,
                    min_samples=self.dbscan_min_samples
                ).fit_predict(x)

                uniq = sorted(set(labels) - {-1})
                if len(uniq) == 0:
                    centers = np.array([float(np.mean(x))])
                else:
                    centers = np.sort(
                        np.array([float(np.mean(x[labels == lab]))
                                  for lab in uniq])
                    )
                self.dbscan_centers[c] = centers
                self.bin_centers_[c]   = centers

    def transform(self, df):
        if self.columns_ is None:
            raise RuntimeError("Discretizer must be fitted before transform.")

        out = df[self.columns_].copy()

        if self.method in {"classic_uniform", "classic_quantile", "kmeans"}:
            for c in self.columns_:
                out[c] = self.kbins[c].transform(out[[c]]).astype(int).flatten()
            return out.astype(int)

        for c in self.columns_:
            centers = self.dbscan_centers[c]
            x       = out[c].to_numpy(dtype=float).reshape(-1, 1)
            out[c]  = np.argmin(
                np.abs(x - centers.reshape(1, -1)), axis=1
            ).astype(int)

        return out.astype(int)

    def get_bin_centers(self, col):
        """Return bin center values for a column (for MAE computation)."""
        return self.bin_centers_.get(col, None)


# ============================================================
# FEATURE SELECTION HELPERS
# ============================================================
def drop_other_throughputs_from_X(X):
    if not EXCLUDE_OTHER_THROUGHPUTS:
        return X
    return X.drop(
        columns=[c for c in X.columns if c.startswith("throughput_")],
        errors="ignore"
    )


def _split_X_y(df):
    X = drop_other_throughputs_from_X(df.drop(columns=[TARGET]).copy())
    y = df[TARGET]
    return X, y


def _top_k_by_mi(train_df, k):
    X, y = _split_X_y(train_df)
    if X.shape[1] == 0:
        return []
    mi    = mutual_info_regression(X.to_numpy(), y.to_numpy(), random_state=0)
    order = np.argsort(mi)[::-1]
    return [X.columns[i] for i in order[:min(k, len(order))]]


def select_mrmr(train_df, k):
    X, y = _split_X_y(train_df)
    cols = list(X.columns)
    if not cols:
        return []

    Xv  = X.to_numpy()
    rel = mutual_info_regression(Xv, y.to_numpy(), random_state=0)

    selected  = [int(np.argmax(rel))]
    remaining = set(range(len(cols))) - {selected[0]}

    while len(selected) < min(k, len(cols)) and remaining:
        best_j, best_score = None, -1e18
        for j in remaining:
            red = float(np.mean([
                float(mutual_info_regression(
                    Xv[:, [j]], Xv[:, s], random_state=0
                )[0])
                for s in selected
            ]))
            score = float(rel[j]) - red
            if score > best_score:
                best_score, best_j = score, j
        selected.append(best_j)
        remaining.remove(best_j)

    return [cols[i] for i in selected]


def _build_blacklist_single_slice(df):
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
            if p != c and p in layer_index and c in layer_index \
                    and layer_index[p] > layer_index[c]:
                black.append((p, c))
    
    # Enforce pipeline order: throughput edges only go forward (S1 → S2 → S3)
    throughputs = ["throughput_1", "throughput_2", "throughput_3"]
    for i, t_child in enumerate(throughputs):
        for t_parent in throughputs[i+1:]:  # downstream services can't cause upstream
            if t_parent in all_vars and t_child in all_vars:
                black.append((t_parent, t_child))
    
    return black


def select_markov_blanket(train_df, k, score_name, n_bins, disc_method):
    filtered = train_df.copy()

    if EXCLUDE_OTHER_THROUGHPUTS:
        filtered = filtered.drop(
            columns=[c for c in filtered.columns
                     if c != TARGET and c.startswith("throughput_")],
            errors="ignore"
        )

    filtered = filtered[[
        c for c in filtered.columns
        if c == TARGET or filtered[c].nunique() > 1
    ]].copy()

    quick_method = "classic_uniform" if disc_method == "dbscan" else disc_method
    quick_disc   = Discretizer(method=quick_method, n_bins=n_bins)
    quick_disc.fit(filtered)
    train_disc = quick_disc.transform(filtered)

    bl   = _build_blacklist_single_slice(train_disc)
    est  = HillClimbSearch(train_disc)
    best = est.estimate(
        scoring_method=make_score(score_name, train_disc),
        black_list=bl,
        max_iter=MB_QUICK_MAX_ITER,
        show_progress=False,
    )

    mb_model = BayesianNetwork(best.edges())
    mb       = [c for c in mb_model.get_markov_blanket(TARGET) if c != TARGET]
    mb_size  = len(mb)

    if len(mb) > k:
        top = _top_k_by_mi(filtered[mb + [TARGET]], k)
        return top, mb_size, "trimmed"

    if len(mb) < k:
        mi_ranked   = [c for c in _top_k_by_mi(filtered, len(filtered.columns))
                       if c not in mb]
        mb_extended = mb + mi_ranked
        return mb_extended[:min(k, len(mb_extended))], mb_size, "padded"

    return mb, mb_size, "none"


def apply_feature_selection(train_raw, test_raw, fs_method, k,
                             score_name, n_bins, disc_method):
    fs_method = fs_method.lower().strip()

    if fs_method == "mrmr":
        predictors = select_mrmr(train_raw, k)
        #keep = predictors + [TARGET]
        always_keep = [c for c in train_raw.columns if c.startswith("throughput_")]
        keep = list(dict.fromkeys(predictors + [TARGET] + always_keep))  # dedup, preserve order
        
        return train_raw[keep].copy(), test_raw[keep].copy(), None, None

    if fs_method == "markov":
        predictors, mb_size, mb_fallback = select_markov_blanket(
            train_raw, k, score_name=score_name,
            n_bins=n_bins, disc_method=disc_method
        )
        #keep = predictors + [TARGET]
        always_keep = [c for c in train_raw.columns if c.startswith("throughput_")]
        keep = list(dict.fromkeys(predictors + [TARGET] + always_keep))  # dedup, preserve order

        return train_raw[keep].copy(), test_raw[keep].copy(), mb_size, mb_fallback

    if fs_method == "pca":
        Xtr, ytr = _split_X_y(train_raw)
        Xte, yte = _split_X_y(test_raw)

        if Xtr.shape[1] == 0:
            raise RuntimeError("No predictors available for PCA after filtering.")

        ncomp = min(k, Xtr.shape[1])
        pca   = PCA(n_components=ncomp, random_state=0)
        pca.fit(Xtr.to_numpy())

        Ztr = pca.transform(Xtr.to_numpy())
        Zte = pca.transform(Xte.to_numpy())

        cols     = [f"pca_{i+1}" for i in range(Ztr.shape[1])]
        train_fs = pd.DataFrame(Ztr, columns=cols).reset_index(drop=True)
        test_fs  = pd.DataFrame(Zte, columns=cols).reset_index(drop=True)
        train_fs[TARGET] = ytr.reset_index(drop=True)
        test_fs[TARGET]  = yte.reset_index(drop=True)

        return train_fs, test_fs, None, None

    raise ValueError("fs_method must be one of: markov | mrmr | pca")


# ============================================================
# MODEL SAVING  (live top-N per target)
# ============================================================
def save_dbn_model(model_2s, target, k, fs_method, disc_method,
                   score_name, n_bins, rank=None):
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

    rank_prefix = f"rank{rank}_" if rank is not None else ""
    model_base  = (f"{rank_prefix}dbn_{target}_K{k}_{fs_method}_"
                   f"{disc_method}_{score_name}_bins{n_bins}")

    pkl_path   = os.path.join(MODEL_SAVE_DIR, model_base + ".pkl")
    edges_path = os.path.join(MODEL_SAVE_DIR, model_base + "_edges.csv")

    with open(pkl_path, "wb") as f:
        pickle.dump(model_2s, f)

    # Verify the saved model is loadable
    try:
        with open(pkl_path, "rb") as f:
            pickle.load(f)
    except Exception as e:
        raise RuntimeError(f"Saved model failed verification: {e}")

    edges_df = pd.DataFrame(list(model_2s.edges()), columns=["parent", "child"])
    edges_df.to_csv(edges_path, index=False)

    return pkl_path, edges_path


class TopNModelTracker:
    """
    Maintains the top-N models by accuracy for a single target.
    Uses a min-heap to efficiently evict the worst model
    when a better one arrives.
    """

    def __init__(self, n, target):
        self.n        = n
        self.target   = target
        self.heap     = []
        self._counter = 0

    def offer(self, accuracy, row_dict, model_2s,
              k, fs_method, disc_method, score_name, n_bins):
        if np.isnan(accuracy):
            return row_dict

        entry = (accuracy, self._counter, row_dict, model_2s,
                 k, fs_method, disc_method, score_name, n_bins)
        self._counter += 1

        if len(self.heap) < self.n:
            heapq.heappush(self.heap, entry)
            self._save_all()
        elif accuracy > self.heap[0][0]:
            evicted = heapq.heapreplace(self.heap, entry)
            self._delete_saved(evicted)
            self._save_all()

        for _, _, rd, _, _, _, _, _, _ in self.heap:
            if rd is row_dict:
                return rd
        return row_dict

    def _save_all(self):
        # Clean up all existing rank files for this target before rewriting
        if os.path.exists(MODEL_SAVE_DIR):
            for f in os.listdir(MODEL_SAVE_DIR):
                if f.startswith("rank") and f"_{self.target}_" in f:
                    try:
                        os.remove(os.path.join(MODEL_SAVE_DIR, f))
                    except OSError:
                        pass

        sorted_entries = sorted(self.heap, key=lambda e: e[0], reverse=True)
        for rank, (acc, uid, rd, mdl, k, fs, disc, sc, nb) in enumerate(
                sorted_entries, start=1):
            pkl, edges = save_dbn_model(
                mdl, self.target, k, fs, disc, sc, nb, rank=rank
            )
            rd["model_pkl_path"]  = pkl
            rd["edges_csv_path"]  = edges

    def _delete_saved(self, evicted_entry):
        _, _, rd, _, _, _, _, _, _ = evicted_entry
        for key in ("model_pkl_path", "edges_csv_path"):
            path = rd.get(key)
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass


# ============================================================
# BASELINES
# ============================================================
def persistence_baseline(test_ready):
    y_prev = test_ready[TARGET].iloc[:-1].to_numpy()
    y_true = test_ready[TARGET].iloc[1:].to_numpy()
    return float(np.mean(y_prev == y_true))


def majority_baseline(train_ready, test_ready):
    majority = int(train_ready[TARGET].mode().iloc[0])
    y_true   = test_ready[TARGET].iloc[1:].to_numpy()
    acc      = float(np.mean(np.full_like(y_true, majority) == y_true))
    return acc, majority


# ============================================================
# NAIVE DBN BASELINE
# ============================================================
def naive_dbn_baseline(train_ready, test_ready, bin_centers):
    """
    Naive DBN: each variable connects only to itself at t+1.
    No structure learning. Baseline to check if structure
    learning added any value beyond simple auto-regression.
    """
    nodes = list(train_ready.columns)
    edges = [(f"{v}_t", f"{v}_t1") for v in nodes]

    df_t  = train_ready.iloc[:-1].reset_index(drop=True).add_suffix("_t")
    df_t1 = train_ready.iloc[1:].reset_index(drop=True).add_suffix("_t1")
    df_2s = pd.concat([df_t, df_t1], axis=1)

    model = BayesianNetwork(edges)
    model.fit(
        df_2s,
        estimator=BayesianEstimator,
        prior_type="BDeu",
        equivalent_sample_size=10,
    )

    res = evaluate(model, test_ready, bin_centers)
    return res["accuracy"], res["mean_prob_true"], res["mae"]


# ============================================================
# EVALUATION  (with inference cache and MAE)
# ============================================================
def evaluate(model_2s, test_df, bin_centers):
    """
    Evaluate model on test data.

    bin_centers: dict mapping column name to array of bin center
                 values in original units, used for MAE computation.

    Inference cache: stores results for previously seen evidence
    combinations to avoid redundant computation. With few bins
    and few features, many rows share identical evidence — the
    cache can give 10-50x speedup.
    """
    test_df = test_df.reset_index(drop=True)
    infer   = VariableElimination(model_2s)

    if EVIDENCE_MODE == "subset":
        ev_cols = [c for c in test_df.columns
                   if any(c.startswith(p) for p in EVIDENCE_SUBSET)]
    elif EVIDENCE_MODE == "full_no_target":
        ev_cols = [c for c in test_df.columns if c != TARGET]
    else:
        ev_cols = list(test_df.columns)

    model_nodes = set(model_2s.nodes())
    valid_cols  = [c for c in ev_cols if f"{c}_t" in model_nodes]

    n = len(test_df) - 1
    if n <= 0:
        return {
            "accuracy":       float("nan"),
            "mean_prob_true": float("nan"),
            "mae":            float("nan"),
        }

    # Get bin centers for target (for MAE in real units)
    target_centers = bin_centers.get(TARGET, None)

    correct, probs_true, abs_errors = 0, [], []

    # Inference cache: evidence tuple -> query result
    inference_cache = {}
    cache_hits      = 0

    for t in range(n):
        evidence     = {f"{c}_t": int(test_df.iloc[t][c]) for c in valid_cols}
        evidence_key = tuple(sorted(evidence.items()))
        true_next    = int(test_df.iloc[t + 1][TARGET])

        # Use cached result if available
        if evidence_key in inference_cache:
            probs = inference_cache[evidence_key]
            cache_hits += 1
        else:
            q     = infer.query([f"{TARGET}_t1"], evidence=evidence,
                                show_progress=False)
            probs = q.values / q.values.sum()
            inference_cache[evidence_key] = probs

        pred = int(np.argmax(probs))

        if pred == true_next:
            correct += 1
        probs_true.append(float(probs[true_next]))

        # MAE in real units
        if target_centers is not None:
            pred_val = float(target_centers[pred]) \
                if pred < len(target_centers) else float(pred)
            true_val = float(target_centers[true_next]) \
                if true_next < len(target_centers) else float(true_next)
            abs_errors.append(abs(pred_val - true_val))

    mae = float(np.mean(abs_errors)) if abs_errors else float("nan")

    return {
        "accuracy":       correct / n,
        "mean_prob_true": float(np.mean(probs_true)),
        "mae":            mae,
        "cache_hits":     cache_hits,
        "cache_size":     len(inference_cache),
        "n_queries":      n,
    }


# ============================================================
# TWO-STEP AHEAD EVALUATION  (commented out — uncomment when needed)
# ============================================================
# def evaluate_two_step(model_2s, test_df, bin_centers):
#     """
#     Evaluate two-step ahead prediction (t -> t+2).
#     Chains two inference steps:
#       1. Predict t+1 from evidence at t
#       2. Use predicted t+1 as evidence to predict t+2
#
#     Uncomment the call in run_one() to populate
#     accuracy_t2, mae_t2 columns in results CSV.
#     """
#     test_df = test_df.reset_index(drop=True)
#     infer   = VariableElimination(model_2s)
#
#     if EVIDENCE_MODE == "full_no_target":
#         ev_cols = [c for c in test_df.columns if c != TARGET]
#     else:
#         ev_cols = list(test_df.columns)
#
#     model_nodes    = set(model_2s.nodes())
#     valid_cols     = [c for c in ev_cols if f"{c}_t" in model_nodes]
#     target_centers = bin_centers.get(TARGET, None)
#
#     n = len(test_df) - 2
#     if n <= 0:
#         return {"accuracy": float("nan"), "mae": float("nan")}
#
#     correct, abs_errors = 0, []
#
#     for t in range(n):
#         # Step 1: predict t+1 from t
#         evidence_t = {f"{c}_t": int(test_df.iloc[t][c]) for c in valid_cols}
#         q1    = infer.query([f"{TARGET}_t1"], evidence=evidence_t,
#                             show_progress=False)
#         probs1 = q1.values / q1.values.sum()
#         pred_t1 = int(np.argmax(probs1))
#
#         # Step 2: predict t+2 from predicted t+1 as evidence
#         evidence_t1 = {f"{c}_t": int(test_df.iloc[t+1][c])
#                        for c in valid_cols if c != TARGET}
#         evidence_t1[f"{TARGET}_t"] = pred_t1
#         q2     = infer.query([f"{TARGET}_t1"], evidence=evidence_t1,
#                              show_progress=False)
#         probs2  = q2.values / q2.values.sum()
#         pred_t2 = int(np.argmax(probs2))
#         true_t2 = int(test_df.iloc[t + 2][TARGET])
#
#         if pred_t2 == true_t2:
#             correct += 1
#
#         if target_centers is not None:
#             pred_val = float(target_centers[pred_t2]) \
#                 if pred_t2 < len(target_centers) else float(pred_t2)
#             true_val = float(target_centers[true_t2]) \
#                 if true_t2 < len(target_centers) else float(true_t2)
#             abs_errors.append(abs(pred_val - true_val))
#
#     return {
#         "accuracy": correct / n,
#         "mae":      float(np.mean(abs_errors)) if abs_errors else float("nan"),
#     }


# ============================================================
# ONE RUN
# ============================================================
def run_one(raw_df, fs_method, disc_method, score_name, k, n_bins):
    split     = int(TRAIN_FRAC * len(raw_df))
    train_raw = raw_df.iloc[:split].reset_index(drop=True)
    test_raw  = raw_df.iloc[split:].reset_index(drop=True)

    train_fs, test_fs, mb_size, mb_fallback = apply_feature_selection(
        train_raw, test_raw, fs_method, k, score_name, n_bins, disc_method
    )

    def drop_constant_cols(df):
        return df[[c for c in df.columns
                   if c == TARGET or df[c].nunique() > 1]]

    train_fs = drop_constant_cols(train_fs)
    test_fs  = test_fs[train_fs.columns]

    disc = Discretizer(
        method=disc_method, n_bins=n_bins,
        dbscan_eps=DBSCAN_EPS, dbscan_min_samples=DBSCAN_MIN_SAMPLES,
    )
    disc.fit(train_fs)
    train_ready = disc.transform(train_fs)
    test_ready  = disc.transform(test_fs)

    train_ready = drop_constant_cols(train_ready)
    test_ready  = test_ready[train_ready.columns]

    if TARGET not in train_ready.columns or TARGET not in test_ready.columns:
        raise RuntimeError(
            f"TARGET missing after FS={fs_method} DISC={disc_method} K={k}"
        )

    # Explicit throughput leakage guard
    if EXCLUDE_OTHER_THROUGHPUTS:
        other_thr = [c for c in train_ready.columns
                     if c.startswith("throughput_") and c != TARGET]
        if other_thr:
            train_ready = train_ready.drop(columns=other_thr)
            test_ready  = test_ready[[c for c in test_ready.columns
                                       if c not in other_thr]]

    # Bin centers for MAE computation
    bin_centers = {c: disc.get_bin_centers(c) for c in train_fs.columns}

    persistence_acc              = persistence_baseline(test_ready)
    majority_acc, majority_class = majority_baseline(train_ready, test_ready)

    print("TRAIN_READY COLS:", train_ready.columns.tolist())
    print("TRAIN_READY LEN:", len(train_ready))
    print("PERSISTENCE ACC:", persistence_acc)
    print("MAJORITY ACC:", majority_acc, "MAJORITY CLASS:", majority_class)

    # Naive DBN baseline
    naive_acc, naive_mean_prob, naive_mae = naive_dbn_baseline(
        train_ready, test_ready, bin_centers
    )

    # Learned DBN
    t_train_start = time.perf_counter()
    model_2s, *_ = build_dbn_model_2s(train_ready, score_name=score_name)
    t_train_end   = time.perf_counter()

    t_eval_start = time.perf_counter()
    res          = evaluate(model_2s, test_ready, bin_centers)
    t_eval_end   = time.perf_counter()

    train_time_sec = t_train_end - t_train_start
    eval_time_sec  = t_eval_end  - t_eval_start

    print(f"TRAIN TIME: {train_time_sec:.2f}s  EVAL TIME: {eval_time_sec:.2f}s")
    print(f"CACHE hits={res['cache_hits']} / {res['n_queries']} queries "
          f"({100*res['cache_hits']/max(res['n_queries'],1):.1f}%) "
          f"unique evidence combos={res['cache_size']}")

    # ── Evaluate all three throughputs on the same model ──────────────────
    all_tput_results = {}
    model_nodes = set(model_2s.nodes())
    for eval_tgt in ["throughput_1", "throughput_2", "throughput_3"]:
        t1_node = f"{eval_tgt}_t1"
        if t1_node in model_nodes and eval_tgt in test_ready.columns:
            saved_target = TARGET
            globals()["TARGET"] = eval_tgt
            try:
                r = evaluate(model_2s, test_ready, bin_centers)
                all_tput_results[eval_tgt] = r
                print(f"  [{eval_tgt}] acc={r['accuracy']:.3f} "
                      f"mae={r['mae']:.3f} prob={r['mean_prob_true']:.3f}")
            except Exception as e:
                print(f"  [{eval_tgt}] eval failed: {e}")
                all_tput_results[eval_tgt] = None
            finally:
                globals()["TARGET"] = saved_target
        else:
            print(f"  [{eval_tgt}] not in model, skipping")
            all_tput_results[eval_tgt] = None
    # ──────────────────────────────────────────────────────────────────────

    accuracy_t2 = np.nan
    mae_t2      = np.nan
    cols_used   = list(train_ready.columns)

    del train_raw, test_raw, train_fs, test_fs, disc
    gc.collect()

    return (
        res, cols_used, model_2s,
        persistence_acc, majority_acc, majority_class,
        naive_acc, naive_mean_prob, naive_mae,
        mb_size, mb_fallback,
        train_time_sec, eval_time_sec,
        accuracy_t2, mae_t2,
        all_tput_results,
    )


# ============================================================
# MAIN
# ============================================================
def main():
    global TARGET

    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    all_results = []

    for tgt in TARGETS:
        TARGET = tgt
        print("\n==============================")
        print(f" TARGET = {TARGET}")
        print("==============================\n")

        # --- Load single CSV (currently active) ---
        raw = load_and_clean(CSV_PATH, TARGET)

        raw = aggregate_to_modeling_granularity(raw)

        if MAX_ROWS is not None:
            raw = raw.iloc[:MAX_ROWS].reset_index(drop=True)

        print(f"[DATA] rows after aggregation: {len(raw)}")

        tracker = TopNModelTracker(n=TOP_N_MODELS_TO_SAVE, target=TARGET)
        rows    = []

        for k in K_VALUES:
            for fs in FEATURE_SELECTION_METHODS:
                for disc in DISCRETIZATION_METHODS:
                    for sc in SCORES:
                        for n_bins in N_BINS_VALUES:

                            # Skip redundant n_bins for dbscan
                            if disc == "dbscan" and n_bins != N_BINS_VALUES[0]:
                                continue

                            print(
                                f"[START] target={TARGET} K={k} "
                                f"n_bins={n_bins!s:>4} fs={fs} "
                                f"disc={disc} score={sc}",
                                flush=True,
                            )
                            start = time.perf_counter()
                            try:
                                (
                                    res, cols, model_2s,
                                    persistence_acc, majority_acc, majority_class,
                                    naive_acc, naive_mean_prob, naive_mae,
                                    mb_size, mb_fallback,
                                    train_time_sec, eval_time_sec,
                                    accuracy_t2, mae_t2,
                                    all_tput_results,
                                ) = run_one(raw, fs, disc, sc, k, n_bins)

                                elapsed = time.perf_counter() - start

                                row = {
                                    "target":                    TARGET,
                                    "K":                         k,
                                    "n_bins":                    n_bins if disc != "dbscan" else None,
                                    "fs":                        fs,
                                    "disc":                      disc,
                                    "score":                     sc,
                                    "accuracy":                  res["accuracy"],
                                    "mean_prob_true":            res["mean_prob_true"],
                                    "mae":                       res["mae"],
                                    "accuracy_t2":               accuracy_t2,
                                    "mae_t2":                    mae_t2,
                                    "naive_dbn_accuracy":        naive_acc,
                                    "naive_dbn_mean_prob":       naive_mean_prob,
                                    "naive_dbn_mae":             naive_mae,
                                    "persistence_accuracy":      persistence_acc,
                                    "majority_accuracy":         majority_acc,
                                    "majority_class":            majority_class,
                                    "n_features_incl_target":    len(cols),
                                    "features":                  ",".join(cols),
                                    "mb_size":                   mb_size,
                                    "mb_fallback":               mb_fallback,
                                    "train_time_sec":            train_time_sec,
                                    "eval_time_sec":             eval_time_sec,
                                    "total_time_sec":            elapsed,
                                    "cache_hits":                res["cache_hits"],
                                    "cache_size":                res["cache_size"],
                                    "model_pkl_path":            None,
                                    "edges_csv_path":            None,
                                    "exclude_other_throughputs": EXCLUDE_OTHER_THROUGHPUTS,
                                    "modeling_granularity_sec":  MODELING_GRANULARITY_SEC,
                                    "error":                     None,
                                    "acc_tput1": all_tput_results.get("throughput_1", {}).get("accuracy", np.nan) if all_tput_results.get("throughput_1") else np.nan,
                                    "acc_tput2": all_tput_results.get("throughput_2", {}).get("accuracy", np.nan) if all_tput_results.get("throughput_2") else np.nan,
                                    "acc_tput3": all_tput_results.get("throughput_3", {}).get("accuracy", np.nan) if all_tput_results.get("throughput_3") else np.nan,
                                    "mae_tput1": all_tput_results.get("throughput_1", {}).get("mae", np.nan) if all_tput_results.get("throughput_1") else np.nan,
                                    "mae_tput2": all_tput_results.get("throughput_2", {}).get("mae", np.nan) if all_tput_results.get("throughput_2") else np.nan,
                                    "mae_tput3": all_tput_results.get("throughput_3", {}).get("mae", np.nan) if all_tput_results.get("throughput_3") else np.nan,
                                }

                                # Live top-N saving
                                row = tracker.offer(
                                    res["accuracy"], row, model_2s,
                                    k, fs, disc, sc, n_bins
                                )

                                rows.append(row)

                                print(
                                    f"[OK] target={TARGET} K={k:2d} "
                                    f"n_bins={str(n_bins if disc != 'dbscan' else '-'):>4} "
                                    f"fs={fs:6s} disc={disc:15s} score={sc:3s} "
                                    f"total={elapsed:.2f}s "
                                    f"(train={train_time_sec:.1f}s "
                                    f"eval={eval_time_sec:.1f}s) "
                                    f"| acc={res['accuracy']:.3f} "
                                    f"| prob={res['mean_prob_true']:.3f} "
                                    f"| mae={res['mae']:.3f} "
                                    f"| naive={naive_acc:.3f} "
                                    f"| pers={persistence_acc:.3f} "
                                    f"| delta_naive={res['accuracy']-naive_acc:.3f}"
                                    + (f" | mb_size={mb_size} mb_fb={mb_fallback}"
                                       if fs == "markov" else ""),
                                    flush=True,
                                )

                                del model_2s
                                gc.collect()

                            except Exception as e:
                                elapsed = time.perf_counter() - start
                                rows.append({
                                    "target":                    TARGET,
                                    "K":                         k,
                                    "n_bins":                    n_bins if disc != "dbscan" else None,
                                    "fs":                        fs,
                                    "disc":                      disc,
                                    "score":                     sc,
                                    "accuracy":                  np.nan,
                                    "mean_prob_true":            np.nan,
                                    "mae":                       np.nan,
                                    "accuracy_t2":               np.nan,
                                    "mae_t2":                    np.nan,
                                    "naive_dbn_accuracy":        np.nan,
                                    "naive_dbn_mean_prob":       np.nan,
                                    "naive_dbn_mae":             np.nan,
                                    "persistence_accuracy":      np.nan,
                                    "majority_accuracy":         np.nan,
                                    "majority_class":            np.nan,
                                    "n_features_incl_target":    np.nan,
                                    "features":                  None,
                                    "mb_size":                   None,
                                    "mb_fallback":               None,
                                    "train_time_sec":            np.nan,
                                    "eval_time_sec":             np.nan,
                                    "total_time_sec":            elapsed,
                                    "cache_hits":                np.nan,
                                    "cache_size":                np.nan,
                                    "model_pkl_path":            None,
                                    "edges_csv_path":            None,
                                    "exclude_other_throughputs": EXCLUDE_OTHER_THROUGHPUTS,
                                    "modeling_granularity_sec":  MODELING_GRANULARITY_SEC,
                                    "error":                     str(e),
                                    "acc_tput1": np.nan,
                                    "acc_tput2": np.nan,
                                    "acc_tput3": np.nan,
                                    "mae_tput1": np.nan,
                                    "mae_tput2": np.nan,
                                    "mae_tput3": np.nan,
                                })
                                print(
                                    f"[FAIL] target={TARGET} K={k} n_bins={n_bins} "
                                    f"fs={fs} disc={disc} score={sc} "
                                    f"time={elapsed:.2f}s: {e}"
                                )
                                gc.collect()

        out = pd.DataFrame(rows)

        out_sorted = out.sort_values(
            by=["accuracy", "mean_prob_true"],
            ascending=False, na_position="last"
        )

        print("\n=== TOP RESULTS ===")
        with pd.option_context("display.max_colwidth", 120):
            print(
                out_sorted.head(50)[[
                    "target", "K", "n_bins", "fs", "disc", "score",
                    "accuracy", "mean_prob_true", "mae",
                    "naive_dbn_accuracy", "persistence_accuracy",
                    "n_features_incl_target", "mb_size", "mb_fallback",
                    "train_time_sec", "eval_time_sec",
                ]].to_string(index=False)
            )

        all_results.append(out)

        pd.concat(all_results, ignore_index=True).to_csv(
            RESULTS_CSV_PATH, index=False
        )

        del raw, rows, out, out_sorted
        gc.collect()

    pd.concat(all_results, ignore_index=True).to_csv(RESULTS_CSV_PATH, index=False)
    print(f"\nSaved results CSV: {RESULTS_CSV_PATH}")


if __name__ == "__main__":
    main()