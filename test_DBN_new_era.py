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
from sklearn.feature_selection import mutual_info_regression, VarianceThreshold
from sklearn.cluster import DBSCAN
from sklearn.metrics import f1_score, log_loss, precision_score, recall_score
from pgmpy.inference import VariableElimination
from pgmpy.estimators import HillClimbSearch, BayesianEstimator
from pgmpy.models import BayesianNetwork
from full_dynamic_bn_new_new_new import build_dbn_model_2s, make_score
from pathlib import Path
import argparse


# ============================================================
# CONFIG
# ============================================================
def parse_args():
    parser = argparse.ArgumentParser(description="DBN hyperparameter sweep")
    parser.add_argument("--csv", type=str, required=True,
                        help="Path to wide DBN CSV file")
    return parser.parse_args()

args = parse_args()
CSV_PATH = args.csv

# K-fold cross-validation config
# Expanding window: fold i trains on rows 0..split_i, tests on split_i..split_i+1
# TRAIN_FRAC kept for backward compatibility but overridden when K_FOLDS > 1
K_FOLDS    = 5
TRAIN_FRAC = 0.8
TARGETS    = ["throughput_3"]

K_VALUES                  = [4, 8, 12, 16, 20]
FEATURE_SELECTION_METHODS = ["markov", "mrmr"]
DISCRETIZATION_METHODS    = ["classic_uniform", "classic_quantile", "kmeans", "fixed"]

# Domain-driven boundaries based on natural modes observed in the throughput
# distribution histograms. Boundaries sit at the valleys between modes:
#   bin 0:   0–100  degraded   (below minimum viable throughput)
#   bin 1: 100–200  baseline   (normal operation around 200 req/s)
#   bin 2: 200–300  elevated   (moderate-high load)
#   bin 3: 300+     high/spike (peak periodic load and spike events)
# For all other features, fixed method falls back to quantile binning.
FIXED_BOUNDARIES = {
    "throughput_1": [100, 200, 300],
    "throughput_2": [100, 200, 300],
    "throughput_3": [100, 200, 300],
}
SCORES                    = ["bic", "aic"]

N_BINS_VALUES      = [2, 3, 4, 6, 8, 10]
DBSCAN_EPS         = 0.30
DBSCAN_MIN_SAMPLES = 10

MB_QUICK_MAX_ITER = 8000

EVIDENCE_MODE   = "full_no_target"
EVIDENCE_SUBSET = ["avg_p_latency_", "cores_", "data_quality_", "buffer_size_"]

EXCLUDE_OTHER_THROUGHPUTS = False

# ============================================================
# CHANGE 1: variance threshold for dropping near-constant features
# Features where almost all values are the same (e.g. process_open_fds
# which had 99.9% of observations in one bin) carry no information and
# only add noise to the structure search. We drop them before doing
# anything else. 0.01 is a very low bar — only truly near-constant
# columns get dropped.
# ============================================================
VARIANCE_THRESHOLD = 0.01

_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
_stem = Path(CSV_PATH).stem.replace("dbn_wide_", "") + f"_run{_timestamp}"

MODEL_SAVE_DIR   = f"saved_dbn_models_top5_{_stem}"
RESULTS_CSV_PATH = f"dbn_k_sweep_results_{_stem}.csv"

MODELING_GRANULARITY_SEC = 30
MAX_ROWS = None


# ============================================================
# LOAD + CLEAN
# ============================================================
def load_and_clean(path, target):
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

    # ============================================================
    # CHANGE 1 APPLIED HERE: drop near-constant columns
    # We always keep the target column regardless of its variance,
    # then apply the threshold to everything else.
    # ============================================================
    non_target_cols = [c for c in df.columns if c != target]
    if non_target_cols:
        selector = VarianceThreshold(threshold=VARIANCE_THRESHOLD)
        selector.fit(df[non_target_cols])
        kept = [c for c, s in zip(non_target_cols, selector.get_support()) if s]
        dropped = [c for c, s in zip(non_target_cols, selector.get_support()) if not s]
        if dropped:
            print(f"[VARIANCE FILTER] Dropped {len(dropped)} near-constant columns: {dropped}")
        df = df[kept + [target]]

    return df


# ============================================================
# TEMPORAL AGGREGATION
# ============================================================
def aggregate_to_modeling_granularity(df, every_n_seconds=MODELING_GRANULARITY_SEC):
    group_ids = np.arange(len(df)) // every_n_seconds
    return df.groupby(group_ids).mean().reset_index(drop=True)


# ============================================================
# TEMPORAL K-FOLD SPLITS
# ============================================================
def make_temporal_folds(df, k=K_FOLDS):
    """
    Expanding-window temporal CV with k valid test folds.

    Splits data into k+1 equally-sized blocks:
      fold 1: train block 1,        test block 2
      fold 2: train blocks 1-2,     test block 3
      ...
      fold k: train blocks 1-k,     test block k+1

    This ensures every fold has a non-empty test set and the last
    fold always uses all remaining data as test.
    Any fold with fewer than 2 rows in train or test is skipped.

    Returns list of (train_df, test_df) pairs.
    """
    n          = len(df)
    boundaries = np.linspace(0, n, k + 2, dtype=int)
    folds      = []
    for i in range(k):
        train_end  = boundaries[i + 1]
        test_start = boundaries[i + 1]
        test_end   = boundaries[i + 2]
        train_df   = df.iloc[0:train_end].reset_index(drop=True)
        test_df    = df.iloc[test_start:test_end].reset_index(drop=True)
        if len(train_df) < 2 or len(test_df) < 2:
            continue
        folds.append((train_df, test_df))
    return folds


# ============================================================
# DISCRETIZATION
# ============================================================
class Discretizer:
    def __init__(self, method, n_bins=4, dbscan_eps=0.3,
                 dbscan_min_samples=10, fixed_boundaries=None):
        self.method             = method
        self.n_bins             = n_bins
        self.dbscan_eps         = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples
        # fixed_boundaries: dict of {column_name: [cut1, cut2, ...]}
        # only used when method="fixed". columns not in this dict
        # fall back to quantile binning with n_bins.
        self.fixed_boundaries   = fixed_boundaries or {}
        self.kbins              = {}
        self.dbscan_centers     = {}
        self.fixed_cuts         = {}
        self.columns_           = None

        valid = {"classic_uniform", "classic_quantile", "kmeans",
                 "dbscan", "fixed"}
        if self.method not in valid:
            raise ValueError(f"Unknown discretizer method: {self.method}")

    def fit(self, df):
        self.columns_ = list(df.columns)

        if self.method == "fixed":
            for c in self.columns_:
                if c in self.fixed_boundaries:
                    # store sorted cut points — np.digitize will use these
                    self.fixed_cuts[c] = sorted(self.fixed_boundaries[c])
                else:
                    # fallback: quantile binning for features without
                    # explicit boundaries (all non-throughput features)
                    kbd = KBinsDiscretizer(
                        n_bins=self.n_bins, encode="ordinal",
                        strategy="quantile"
                    )
                    kbd.fit(df[[c]])
                    self.kbins[c] = kbd
            return

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
            return

        # dbscan
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

    def transform(self, df):
        if self.columns_ is None:
            raise RuntimeError("Discretizer must be fitted before transform.")

        out = df[self.columns_].copy()

        if self.method == "fixed":
            for c in self.columns_:
                if c in self.fixed_cuts:
                    # np.digitize returns 0 for values below first cut,
                    # 1 between cut[0] and cut[1], etc. — exactly what we want
                    out[c] = np.digitize(
                        out[c].to_numpy(dtype=float),
                        bins=self.fixed_cuts[c]
                    ).astype(int)
                else:
                    out[c] = self.kbins[c].transform(
                        out[[c]]
                    ).astype(int).flatten()
            return out.astype(int)

        if self.method in {"classic_uniform", "classic_quantile", "kmeans"}:
            for c in self.columns_:
                out[c] = self.kbins[c].transform(out[[c]]).astype(int).flatten()
            return out.astype(int)

        # dbscan
        for c in self.columns_:
            centers = self.dbscan_centers[c]
            x       = out[c].to_numpy(dtype=float).reshape(-1, 1)
            out[c]  = np.argmin(
                np.abs(x - centers.reshape(1, -1)), axis=1
            ).astype(int)

        return out.astype(int)


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

    throughputs = ["throughput_1", "throughput_2", "throughput_3"]
    for i, t_child in enumerate(throughputs):
        for t_parent in throughputs[i+1:]:
            if t_parent in all_vars and t_child in all_vars:
                black.append((t_parent, t_child))

    return black


def _build_blacklist_two_slice(train_ready):
    """
    Blacklist for the two-slice DBN structure (t -> t+1).
    Prevents reverse pipeline edges across time slices:
      throughput_3_t -> throughput_2_t1  (downstream cannot cause upstream)
      throughput_3_t -> throughput_1_t1
      throughput_2_t -> throughput_1_t1
    These edges are statistically possible due to confounding but
    causally implausible given the pipeline direction S1 -> S2 -> S3.
    """
    nodes = list(train_ready.columns)
    black = []

    throughputs  = ["throughput_1", "throughput_2", "throughput_3"]
    tput_present = [t for t in throughputs if t in nodes]

    for i, t_parent in enumerate(tput_present):
        for t_child in tput_present[:i]:
            black.append((f"{t_parent}_t", f"{t_child}_t1"))

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

    quick_method = "classic_uniform" if disc_method in {"dbscan", "fixed"} else disc_method
    quick_disc   = Discretizer(
        method=quick_method, n_bins=n_bins,
        fixed_boundaries=FIXED_BOUNDARIES if quick_method == "fixed" else {},
    )
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
        predictors  = select_mrmr(train_raw, k)
        always_keep = [c for c in train_raw.columns if c.startswith("throughput_")]
        keep        = list(dict.fromkeys(predictors + [TARGET] + always_keep))
        return train_raw[keep].copy(), test_raw[keep].copy(), None, None

    if fs_method == "markov":
        predictors, mb_size, mb_fallback = select_markov_blanket(
            train_raw, k, score_name=score_name,
            n_bins=n_bins, disc_method=disc_method
        )
        always_keep = [c for c in train_raw.columns if c.startswith("throughput_")]
        keep        = list(dict.fromkeys(predictors + [TARGET] + always_keep))
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
# MODEL SAVING
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

    try:
        with open(pkl_path, "rb") as f:
            pickle.load(f)
    except Exception as e:
        raise RuntimeError(f"Saved model failed verification: {e}")

    edges_df = pd.DataFrame(list(model_2s.edges()), columns=["parent", "child"])
    edges_df.to_csv(edges_path, index=False)

    return pkl_path, edges_path


# ============================================================
# CHANGE 2: BestPerBinsTracker now selects best model by F1
# instead of accuracy.
#
# Previously the tracker kept the model with the highest accuracy.
# Because throughput_3 is heavily imbalanced (85% of observations
# in one bin), a model can get high accuracy just by always predicting
# the majority bin — without actually learning anything useful.
# F1 (macro) is a better criterion because it requires the model to
# predict minority bins correctly too.
# ============================================================
class BestPerBinsTracker:
    """
    Saves the best model (by macro F1) for each distinct n_bins value.
    Produces one saved model per bin count, allowing direct comparison
    of causal structure and performance across discretization granularities.
    """

    def __init__(self, target):
        self.target   = target
        self.best     = {}
        self._counter = 0

    def offer(self, f1, row_dict, model_2s,
              k, fs_method, disc_method, score_name, n_bins):
        # CHANGE 2: parameter renamed from accuracy to f1
        if np.isnan(f1):
            return row_dict

        self._counter += 1
        key = n_bins

        if key not in self.best or f1 > self.best[key][0]:
            if key in self.best:
                self._delete_saved(self.best[key][2])
            self.best[key] = (f1, self._counter, row_dict,
                              model_2s, k, fs_method, disc_method,
                              score_name, n_bins)
            self._save_bin(key)

        if key in self.best and self.best[key][2] is row_dict:
            return row_dict
        return row_dict

    def _save_bin(self, key):
        f1, uid, rd, mdl, k, fs, disc, sc, nb = self.best[key]

        if os.path.exists(MODEL_SAVE_DIR):
            for f in os.listdir(MODEL_SAVE_DIR):
                if (f"_{self.target}_" in f and
                        f"_bins{nb}" in f and
                        f.startswith("best_bins")):
                    try:
                        os.remove(os.path.join(MODEL_SAVE_DIR, f))
                    except OSError:
                        pass

        os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
        model_base  = (f"best_bins{nb}_dbn_{self.target}_K{k}_{fs}_"
                       f"{disc}_{sc}_bins{nb}")
        pkl_path    = os.path.join(MODEL_SAVE_DIR, model_base + ".pkl")
        edges_path  = os.path.join(MODEL_SAVE_DIR, model_base + "_edges.csv")

        with open(pkl_path, "wb") as f:
            pickle.dump(mdl, f)
        try:
            with open(pkl_path, "rb") as f:
                pickle.load(f)
        except Exception as e:
            raise RuntimeError(f"Saved model failed verification: {e}")

        edges_df = pd.DataFrame(list(mdl.edges()), columns=["parent", "child"])
        edges_df.to_csv(edges_path, index=False)

        rd["model_pkl_path"] = pkl_path
        rd["edges_csv_path"] = edges_path

    def _delete_saved(self, rd):
        for key in ("model_pkl_path", "edges_csv_path"):
            path = rd.get(key)
            if path and os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass

    def summary(self):
        print(f"\n=== BEST MODEL PER N_BINS ({self.target}) ===")
        for nb in sorted(self.best.keys()):
            f1, _, _, _, k, fs, disc, sc, _ = self.best[nb]
            print(f"  bins={nb}: f1={f1:.4f} K={k} fs={fs} "
                  f"disc={disc} score={sc}")


# ============================================================
# BASELINES
# ============================================================
def persistence_baseline(test_ready):
    y_prev = test_ready[TARGET].iloc[:-1].to_numpy()
    y_true = test_ready[TARGET].iloc[1:].to_numpy()
    acc       = float(np.mean(y_prev == y_true))
    f1        = float(f1_score(y_true, y_prev, average="macro", zero_division=0))
    precision = float(precision_score(y_true, y_prev, average="macro", zero_division=0))
    recall    = float(recall_score(y_true, y_prev, average="macro", zero_division=0))
    return acc, f1, precision, recall


# ============================================================
# EVALUATION
# ============================================================
def evaluate(model_2s, test_df, target_override=None):
    """
    Evaluate a DBN model on test data.

    CHANGE 3: accepts an optional target_override parameter instead
    of reading from the global TARGET variable. This replaces the
    globals()["TARGET"] mutation that was used to evaluate S1/S2
    throughputs on the same model. Mutating a global inside a function
    is fragile — if an exception occurs, the global stays corrupted
    for all subsequent configs in the sweep.
    """
    # CHANGE 3: use override if provided, otherwise fall back to global
    eval_target = target_override if target_override is not None else TARGET

    test_df = test_df.reset_index(drop=True)
    infer   = VariableElimination(model_2s)

    if EVIDENCE_MODE == "subset":
        ev_cols = [c for c in test_df.columns
                   if any(c.startswith(p) for p in EVIDENCE_SUBSET)]
    elif EVIDENCE_MODE == "full_no_target":
        ev_cols = [c for c in test_df.columns if c != eval_target]
    else:
        ev_cols = list(test_df.columns)

    model_nodes = set(model_2s.nodes())
    valid_cols  = [c for c in ev_cols if f"{c}_t" in model_nodes]

    n = len(test_df) - 1
    if n <= 0:
        return {
            "accuracy":  float("nan"),
            "f1":        float("nan"),
            "precision": float("nan"),
            "recall":    float("nan"),
            "log_loss":  float("nan"),
            "cache_hits": 0,
            "cache_size": 0,
            "n_queries":  0,
        }

    target_t1_node = f"{eval_target}_t1"
    try:
        n_classes = len(model_2s.get_cpds(target_t1_node).state_names[target_t1_node])
    except Exception:
        n_classes = None

    try:
        target_states_eval = model_2s.get_cpds(target_t1_node).state_names[target_t1_node]
        labels_eval        = [int(s) for s in target_states_eval]
    except Exception:
        labels_eval = list(range(n_classes)) if n_classes is not None else None

    correct         = 0
    y_true_list     = []
    y_pred_list     = []
    y_prob_list     = []
    inference_cache = {}
    cache_hits      = 0

    for t in range(n):
        evidence     = {f"{c}_t": int(test_df.iloc[t][c]) for c in valid_cols}
        evidence_key = tuple(sorted(evidence.items()))
        true_next    = int(test_df.iloc[t + 1][eval_target])

        if evidence_key in inference_cache:
            probs = inference_cache[evidence_key]
            cache_hits += 1
        else:
            q     = infer.query([target_t1_node], evidence=evidence,
                                show_progress=False)
            probs = q.values / q.values.sum()
            inference_cache[evidence_key] = probs

        pred_idx = int(np.argmax(probs))

        y_true_list.append(true_next)
        y_prob_list.append(probs)

        if labels_eval is not None and pred_idx < len(labels_eval):
            pred = labels_eval[pred_idx]
        else:
            pred = pred_idx
        y_pred_list.append(pred)

        if pred == true_next:
            correct += 1

    y_true_arr = np.array(y_true_list)
    y_pred_arr = np.array(y_pred_list)
    y_prob_arr = np.array(y_prob_list)

    accuracy  = correct / n
    f1        = float(f1_score(y_true_arr, y_pred_arr,
                               average="macro", zero_division=0))
    precision = float(precision_score(y_true_arr, y_pred_arr,
                                      average="macro", zero_division=0))
    recall    = float(recall_score(y_true_arr, y_pred_arr,
                                   average="macro", zero_division=0))

    if labels_eval is not None:
        labels = labels_eval
    else:
        labels = sorted(list(set(y_true_list)))

    if y_prob_arr.shape[1] < len(labels):
        pad = np.zeros((y_prob_arr.shape[0], len(labels) - y_prob_arr.shape[1]))
        y_prob_arr = np.hstack([y_prob_arr, pad])

    eps        = 1e-12
    y_prob_arr = np.clip(y_prob_arr, eps, 1.0)
    y_prob_arr = y_prob_arr / y_prob_arr.sum(axis=1, keepdims=True)

    ll = float(log_loss(y_true_arr, y_prob_arr, labels=labels))

    return {
        "accuracy":   accuracy,
        "f1":         f1,
        "precision":  precision,
        "recall":     recall,
        "log_loss":   ll,
        "cache_hits": cache_hits,
        "cache_size": len(inference_cache),
        "n_queries":  n,
    }


# ============================================================
# AUTOREGRESSIVE DBN BASELINE
# ============================================================
def autoregressive_dbn_baseline(train_ready, test_ready):
    nodes = list(train_ready.columns)
    edges = [(f"{v}_t", f"{v}_t1") for v in nodes]

    df_t  = train_ready.iloc[:-1].reset_index(drop=True).add_suffix("_t")
    df_t1 = train_ready.iloc[1:].reset_index(drop=True).add_suffix("_t1")
    df_2s = pd.concat([df_t, df_t1], axis=1)

    model = BayesianNetwork(edges)
    state_names_2s = {
        f"{v}_t":  list(range(int(max(
            df_t[f"{v}_t"].max(),
            test_ready[v].max() if v in test_ready.columns else 0
        )) + 1)) for v in nodes
    }
    state_names_2s.update({
        f"{v}_t1": list(range(int(max(
            df_t1[f"{v}_t1"].max(),
            test_ready[v].max() if v in test_ready.columns else 0
        )) + 1)) for v in nodes
    })
    model.fit(
        df_2s,
        estimator=BayesianEstimator,
        prior_type="BDeu",
        equivalent_sample_size=10,
        state_names=state_names_2s,
    )

    res = evaluate(model, test_ready)
    return (res["accuracy"], res["f1"], res["precision"],
            res["recall"], res["log_loss"])


# ============================================================
# STATIC BN INFERENCE BASELINE
# ============================================================
def static_bn_inference_baseline(train_ready, test_ready, score_name):
    train_s = train_ready[[c for c in train_ready.columns
                           if train_ready[c].nunique() > 1]].copy()
    if TARGET not in train_s.columns:
        return float("nan"), float("nan"), float("nan"), float("nan"), float("nan")

    test_s = test_ready[[c for c in train_s.columns
                         if c in test_ready.columns]].copy()

    bl  = _build_blacklist_single_slice(train_s)
    est = HillClimbSearch(train_s)

    try:
        best = est.estimate(
            scoring_method=make_score(score_name, train_s),
            black_list=bl,
            max_iter=MB_QUICK_MAX_ITER,
            show_progress=False,
        )
        model = BayesianNetwork(best.edges())
        state_names_s = {
            c: list(range(int(max(
                train_s[c].max(),
                test_s[c].max() if c in test_s.columns else train_s[c].max()
            )) + 1))
            for c in train_s.columns
        }
        model.fit(
            train_s,
            estimator=BayesianEstimator,
            prior_type="BDeu",
            equivalent_sample_size=10,
            state_names=state_names_s,
        )
    except Exception as e:
        print(f"  [static BN inference] structure learning failed: {e}")
        return float("nan"), float("nan"), float("nan"), float("nan"), float("nan")

    infer   = VariableElimination(model)
    ev_cols = [c for c in test_s.columns if c != TARGET
               and c in model.nodes()]
    n       = len(test_s)

    if n <= 0 or TARGET not in model.nodes():
        return float("nan"), float("nan"), float("nan"), float("nan"), float("nan")

    try:
        target_states = model.get_cpds(TARGET).state_names[TARGET]
        labels_si     = [int(s) for s in target_states]
        n_classes     = len(labels_si)
    except Exception:
        target_states = None
        labels_si     = None
        n_classes     = None

    y_true_list, y_pred_list, y_prob_list = [], [], []
    inference_cache = {}

    for t in range(n):
        evidence     = {c: int(test_s.iloc[t][c]) for c in ev_cols
                        if not np.isnan(test_s.iloc[t][c])}
        evidence_key = tuple(sorted(evidence.items()))
        true_val     = int(test_s.iloc[t][TARGET])

        if evidence_key in inference_cache:
            probs = inference_cache[evidence_key]
        else:
            try:
                q     = infer.query([TARGET], evidence=evidence,
                                    show_progress=False)
                probs = q.values / q.values.sum()
            except Exception:
                if n_classes is not None:
                    probs = np.ones(n_classes) / n_classes
                else:
                    probs = np.array([0.5, 0.5])
            inference_cache[evidence_key] = probs

        if labels_si is not None and int(np.argmax(probs)) < len(labels_si):
            pred = labels_si[int(np.argmax(probs))]
        else:
            pred = int(np.argmax(probs))

        y_true_list.append(true_val)
        y_pred_list.append(pred)
        y_prob_list.append(probs)

    y_true_arr = np.array(y_true_list)
    y_pred_arr = np.array(y_pred_list)
    y_prob_arr = np.array(y_prob_list)

    accuracy  = float(np.mean(y_true_arr == y_pred_arr))
    f1        = float(f1_score(y_true_arr, y_pred_arr,
                               average="macro", zero_division=0))
    precision = float(precision_score(y_true_arr, y_pred_arr,
                                      average="macro", zero_division=0))
    recall    = float(recall_score(y_true_arr, y_pred_arr,
                                   average="macro", zero_division=0))

    if target_states is not None:
        labels = [int(s) for s in target_states]
    elif n_classes is not None:
        labels = list(range(n_classes))
    else:
        labels = sorted(list(set(y_true_list)))

    if y_prob_arr.shape[1] < len(labels):
        pad = np.zeros((y_prob_arr.shape[0], len(labels) - y_prob_arr.shape[1]))
        y_prob_arr = np.hstack([y_prob_arr, pad])

    eps        = 1e-12
    y_prob_arr = np.clip(y_prob_arr, eps, 1.0)
    y_prob_arr = y_prob_arr / y_prob_arr.sum(axis=1, keepdims=True)

    ll = float(log_loss(y_true_arr, y_prob_arr, labels=labels))

    return accuracy, f1, precision, recall, ll


# ============================================================
# ONE RUN
# ============================================================
def run_one(raw_df, fs_method, disc_method, score_name, k, n_bins,
            train_raw=None, test_raw=None):
    if train_raw is None or test_raw is None:
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
        fixed_boundaries=FIXED_BOUNDARIES if disc_method == "fixed" else {},
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

    if EXCLUDE_OTHER_THROUGHPUTS:
        other_thr = [c for c in train_ready.columns
                     if c.startswith("throughput_") and c != TARGET]
        if other_thr:
            train_ready = train_ready.drop(columns=other_thr)
            test_ready  = test_ready[[c for c in test_ready.columns
                                       if c not in other_thr]]

    (persistence_acc, persistence_f1,
     persistence_precision, persistence_recall) = persistence_baseline(test_ready)

    print("TRAIN_READY COLS:", train_ready.columns.tolist())
    print("TRAIN_READY LEN:", len(train_ready))
    print(f"PERSISTENCE  acc={persistence_acc:.3f} f1={persistence_f1:.3f} "
          f"prec={persistence_precision:.3f} rec={persistence_recall:.3f}")

    (ar_acc, ar_f1, ar_precision,
     ar_recall, ar_log_loss) = autoregressive_dbn_baseline(train_ready, test_ready)
    print(f"AR-DBN       acc={ar_acc:.3f} f1={ar_f1:.3f} "
          f"prec={ar_precision:.3f} rec={ar_recall:.3f} "
          f"log_loss={ar_log_loss:.4f}")

    (si_acc, si_f1, si_precision,
     si_recall, si_log_loss) = static_bn_inference_baseline(train_ready, test_ready, score_name)
    print(f"Static BN SI acc={si_acc:.3f} f1={si_f1:.3f} "
          f"prec={si_precision:.3f} rec={si_recall:.3f} "
          f"log_loss={si_log_loss:.4f}")

    t_train_start = time.perf_counter()
    model_2s, *_ = build_dbn_model_2s(train_ready, score_name=score_name)
    t_train_end   = time.perf_counter()

    t_eval_start = time.perf_counter()
    res          = evaluate(model_2s, test_ready)
    t_eval_end   = time.perf_counter()

    train_time_sec = t_train_end - t_train_start
    eval_time_sec  = t_eval_end  - t_eval_start

    print(f"TRAIN TIME: {train_time_sec:.2f}s  EVAL TIME: {eval_time_sec:.2f}s")
    print(f"CACHE hits={res['cache_hits']} / {res['n_queries']} queries "
          f"({100*res['cache_hits']/max(res['n_queries'],1):.1f}%) "
          f"unique evidence combos={res['cache_size']}")

    # ============================================================
    # CHANGE 4: evaluate S1/S2/S3 throughputs using target_override
    # instead of mutating the global TARGET variable.
    #
    # Previously the code did globals()["TARGET"] = eval_tgt inside
    # this loop, which is a side effect — if an exception escaped the
    # finally block, TARGET would be wrong for all subsequent configs.
    # Now we pass the target explicitly to evaluate() via target_override,
    # which is clean and safe.
    # ============================================================
    all_tput_results = {}
    model_nodes = set(model_2s.nodes())
    for eval_tgt in ["throughput_1", "throughput_2", "throughput_3"]:
        t1_node = f"{eval_tgt}_t1"
        if t1_node in model_nodes and eval_tgt in test_ready.columns:
            try:
                r = evaluate(model_2s, test_ready, target_override=eval_tgt)
                all_tput_results[eval_tgt] = r
                print(f"  [{eval_tgt}] acc={r['accuracy']:.3f} "
                      f"f1={r['f1']:.3f} prec={r['precision']:.3f} "
                      f"rec={r['recall']:.3f} log_loss={r['log_loss']:.4f}")
            except Exception as e:
                print(f"  [{eval_tgt}] eval failed: {e}")
                all_tput_results[eval_tgt] = None
        else:
            print(f"  [{eval_tgt}] not in model, skipping")
            all_tput_results[eval_tgt] = None

    cols_used = list(train_ready.columns)

    del train_raw, test_raw, train_fs, test_fs, disc
    gc.collect()

    return (
        res, cols_used, model_2s,
        persistence_acc, persistence_f1, persistence_precision, persistence_recall,
        ar_acc, ar_f1, ar_precision, ar_recall, ar_log_loss,
        si_acc, si_f1, si_precision, si_recall, si_log_loss,
        mb_size, mb_fallback,
        train_time_sec, eval_time_sec,
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

        raw = load_and_clean(CSV_PATH, TARGET)
        raw = aggregate_to_modeling_granularity(raw)

        if MAX_ROWS is not None:
            raw = raw.iloc[:MAX_ROWS].reset_index(drop=True)

        print(f"[DATA] rows after aggregation: {len(raw)}")

        folds   = make_temporal_folds(raw, k=K_FOLDS)
        tracker = BestPerBinsTracker(target=TARGET)
        rows    = []

        print(f"[FOLDS] Using {K_FOLDS}-fold expanding window temporal CV")
        for fi, (tr, te) in enumerate(folds):
            print(f"  Fold {fi+1}/{K_FOLDS}: "
                  f"train={len(tr)} rows, test={len(te)} rows")

        for k in K_VALUES:
            for fs in FEATURE_SELECTION_METHODS:
                for disc in DISCRETIZATION_METHODS:
                    for sc in SCORES:
                        for n_bins in N_BINS_VALUES:

                            if disc == "dbscan" and n_bins != N_BINS_VALUES[0]:
                                continue

                            print(
                                f"[START] target={TARGET} K={k} "
                                f"n_bins={n_bins!s:>4} fs={fs} "
                                f"disc={disc} score={sc}",
                                flush=True,
                            )
                            config_start = time.perf_counter()

                            fold_results      = []
                            best_fold_model   = None
                            best_fold_acc     = -1.0

                            for fold_idx, (train_raw_fold, test_raw_fold) in enumerate(folds):
                                try:
                                    (
                                        res, cols, model_2s,
                                        persistence_acc, persistence_f1,
                                        persistence_precision, persistence_recall,
                                        ar_acc, ar_f1, ar_precision,
                                        ar_recall, ar_log_loss,
                                        si_acc, si_f1, si_precision,
                                        si_recall, si_log_loss,
                                        mb_size, mb_fallback,
                                        train_time_sec, eval_time_sec,
                                        all_tput_results,
                                    ) = run_one(raw, fs, disc, sc, k, n_bins,
                                                train_raw=train_raw_fold,
                                                test_raw=test_raw_fold)

                                    fold_results.append({
                                        "fold":               fold_idx + 1,
                                        "accuracy":           res["accuracy"],
                                        "f1":                 res["f1"],
                                        "precision":          res["precision"],
                                        "recall":             res["recall"],
                                        "log_loss":           res["log_loss"],
                                        "persistence_acc":    persistence_acc,
                                        "persistence_f1":     persistence_f1,
                                        "persistence_prec":   persistence_precision,
                                        "persistence_rec":    persistence_recall,
                                        "ar_acc":             ar_acc,
                                        "ar_f1":              ar_f1,
                                        "ar_prec":            ar_precision,
                                        "ar_rec":             ar_recall,
                                        "ar_ll":              ar_log_loss,
                                        "si_acc":             si_acc,
                                        "si_f1":              si_f1,
                                        "si_prec":            si_precision,
                                        "si_rec":             si_recall,
                                        "si_ll":              si_log_loss,
                                        "cols":               cols,
                                        "mb_size":            mb_size,
                                        "mb_fallback":        mb_fallback,
                                        "train_time_sec":     train_time_sec,
                                        "eval_time_sec":      eval_time_sec,
                                        "cache_hits":         res["cache_hits"],
                                        "cache_size":         res["cache_size"],
                                        # CHANGE 4: all_tput_results now
                                        # comes from every fold, not just best
                                        "all_tput_results":   all_tput_results,
                                        "error":              None,
                                    })

                                    is_best_fold = res["accuracy"] > best_fold_acc
                                    if is_best_fold:
                                        best_fold_acc = res["accuracy"]
                                        if best_fold_model is not None:
                                            del best_fold_model
                                            gc.collect()
                                        best_fold_model = model_2s
                                    else:
                                        del model_2s
                                        gc.collect()

                                    print(
                                        f"  [fold {fold_idx+1}/{K_FOLDS}] "
                                        f"acc={res['accuracy']:.3f} "
                                        f"f1={res['f1']:.3f} "
                                        f"ll={res['log_loss']:.4f}",
                                        flush=True,
                                    )

                                except Exception as e:
                                    fold_results.append({
                                        "fold": fold_idx + 1, "error": str(e),
                                        "accuracy": np.nan, "f1": np.nan,
                                        "precision": np.nan, "recall": np.nan,
                                        "log_loss": np.nan,
                                        "persistence_acc": np.nan, "persistence_f1": np.nan,
                                        "persistence_prec": np.nan, "persistence_rec": np.nan,
                                        "ar_acc": np.nan, "ar_f1": np.nan,
                                        "ar_prec": np.nan, "ar_rec": np.nan, "ar_ll": np.nan,
                                        "si_acc": np.nan, "si_f1": np.nan,
                                        "si_prec": np.nan, "si_rec": np.nan, "si_ll": np.nan,
                                        "cols": [], "mb_size": None, "mb_fallback": None,
                                        "train_time_sec": np.nan, "eval_time_sec": np.nan,
                                        "cache_hits": np.nan, "cache_size": np.nan,
                                        "all_tput_results": {}, "error": str(e),
                                    })
                                    print(f"  [fold {fold_idx+1}/{K_FOLDS}] FAIL: {e}")
                                    gc.collect()

                            elapsed = time.perf_counter() - config_start

                            def _mean(key):
                                vals = [r[key] for r in fold_results
                                        if r.get("error") is None
                                        and not np.isnan(float(r[key]))]
                                return float(np.mean(vals)) if vals else np.nan

                            def _std(key):
                                vals = [r[key] for r in fold_results
                                        if r.get("error") is None
                                        and not np.isnan(float(r[key]))]
                                return float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0

                            n_ok          = sum(1 for r in fold_results if r.get("error") is None)
                            n_total_folds = len(folds)
                            n_failed      = n_total_folds - n_ok

                            ok_folds     = [r for r in fold_results if r.get("error") is None]
                            cols         = ok_folds[-1]["cols"]        if ok_folds else []
                            mb_size      = ok_folds[-1]["mb_size"]     if ok_folds else None
                            mb_fallback  = ok_folds[-1]["mb_fallback"] if ok_folds else None

                            best_fold_r = max(
                                [r for r in fold_results if r.get("error") is None],
                                key=lambda r: r["accuracy"],
                                default={}
                            )

                            # ============================================================
                            # CHANGE 4 CONTINUED: aggregate per-throughput results
                            # across all folds instead of taking from best fold only.
                            # Previously all_tput_results came from a single best fold,
                            # making S1/S2 metrics non-comparable to the CV-averaged
                            # main metrics. Now we average across all successful folds.
                            # ============================================================
                            def _mean_tput(tgt, metric):
                                vals = []
                                for r in ok_folds:
                                    tr = r.get("all_tput_results", {})
                                    if tr.get(tgt) and tr[tgt].get(metric) is not None:
                                        v = tr[tgt][metric]
                                        if not np.isnan(float(v)):
                                            vals.append(float(v))
                                return float(np.mean(vals)) if vals else np.nan

                            error_str = None
                            if n_failed == n_total_folds:
                                error_str = fold_results[0].get("error", "all folds failed")

                            row = {
                                "target":                    TARGET,
                                "K":                         k,
                                "n_bins":                    n_bins if disc != "dbscan" else None,
                                "fs":                        fs,
                                "disc":                      disc,
                                "score":                     sc,
                                "n_folds":                   n_total_folds,
                                "n_folds_ok":                n_ok,
                                "accuracy":                  _mean("accuracy"),
                                "accuracy_std":              _std("accuracy"),
                                "f1":                        _mean("f1"),
                                "f1_std":                    _std("f1"),
                                "precision":                 _mean("precision"),
                                "precision_std":             _std("precision"),
                                "recall":                    _mean("recall"),
                                "recall_std":                _std("recall"),
                                "log_loss":                  _mean("log_loss"),
                                "log_loss_std":              _std("log_loss"),
                                "persistence_accuracy":      _mean("persistence_acc"),
                                "persistence_accuracy_std":  _std("persistence_acc"),
                                "persistence_f1":            _mean("persistence_f1"),
                                "persistence_f1_std":        _std("persistence_f1"),
                                "persistence_precision":     _mean("persistence_prec"),
                                "persistence_precision_std": _std("persistence_prec"),
                                "persistence_recall":        _mean("persistence_rec"),
                                "persistence_recall_std":    _std("persistence_rec"),
                                "ar_dbn_accuracy":           _mean("ar_acc"),
                                "ar_dbn_accuracy_std":       _std("ar_acc"),
                                "ar_dbn_f1":                 _mean("ar_f1"),
                                "ar_dbn_f1_std":             _std("ar_f1"),
                                "ar_dbn_precision":          _mean("ar_prec"),
                                "ar_dbn_precision_std":      _std("ar_prec"),
                                "ar_dbn_recall":             _mean("ar_rec"),
                                "ar_dbn_recall_std":         _std("ar_rec"),
                                "ar_dbn_log_loss":           _mean("ar_ll"),
                                "ar_dbn_log_loss_std":       _std("ar_ll"),
                                "static_bn_si_accuracy":     _mean("si_acc"),
                                "static_bn_si_accuracy_std": _std("si_acc"),
                                "static_bn_si_f1":           _mean("si_f1"),
                                "static_bn_si_f1_std":       _std("si_f1"),
                                "static_bn_si_precision":    _mean("si_prec"),
                                "static_bn_si_precision_std":_std("si_prec"),
                                "static_bn_si_recall":       _mean("si_rec"),
                                "static_bn_si_recall_std":   _std("si_rec"),
                                "static_bn_si_log_loss":     _mean("si_ll"),
                                "static_bn_si_log_loss_std": _std("si_ll"),
                                "n_features_incl_target":    len(cols),
                                "features":                  ",".join(cols),
                                "mb_size":                   mb_size,
                                "mb_fallback":               mb_fallback,
                                "train_time_sec":            _mean("train_time_sec"),
                                "eval_time_sec":             _mean("eval_time_sec"),
                                "total_time_sec":            elapsed,
                                "cache_hits":                _mean("cache_hits"),
                                "cache_size":                _mean("cache_size"),
                                "best_fold_accuracy":        best_fold_acc if best_fold_acc > -1.0 else np.nan,
                                "best_fold":                 best_fold_r.get("fold", None),
                                "model_pkl_path":            None,
                                "edges_csv_path":            None,
                                "exclude_other_throughputs": EXCLUDE_OTHER_THROUGHPUTS,
                                "modeling_granularity_sec":  MODELING_GRANULARITY_SEC,
                                "error":                     error_str,
                                # Per-throughput: now CV-averaged across all folds
                                "acc_tput1":  _mean_tput("throughput_1", "accuracy"),
                                "acc_tput2":  _mean_tput("throughput_2", "accuracy"),
                                "acc_tput3":  _mean_tput("throughput_3", "accuracy"),
                                "f1_tput1":   _mean_tput("throughput_1", "f1"),
                                "f1_tput2":   _mean_tput("throughput_2", "f1"),
                                "f1_tput3":   _mean_tput("throughput_3", "f1"),
                                "prec_tput1": _mean_tput("throughput_1", "precision"),
                                "prec_tput2": _mean_tput("throughput_2", "precision"),
                                "prec_tput3": _mean_tput("throughput_3", "precision"),
                                "rec_tput1":  _mean_tput("throughput_1", "recall"),
                                "rec_tput2":  _mean_tput("throughput_2", "recall"),
                                "rec_tput3":  _mean_tput("throughput_3", "recall"),
                                "ll_tput1":   _mean_tput("throughput_1", "log_loss"),
                                "ll_tput2":   _mean_tput("throughput_2", "log_loss"),
                                "ll_tput3":   _mean_tput("throughput_3", "log_loss"),
                            }

                            # CHANGE 2: pass f1 to tracker instead of accuracy
                            if best_fold_model is not None and not np.isnan(_mean("f1")):
                                row = tracker.offer(
                                    _mean("f1"), row, best_fold_model,
                                    k, fs, disc, sc, n_bins
                                )

                            rows.append(row)

                            print(
                                f"[OK] target={TARGET} K={k:2d} "
                                f"n_bins={str(n_bins if disc != 'dbscan' else '-'):>4} "
                                f"fs={fs:6s} disc={disc:15s} score={sc:3s} "
                                f"folds={n_ok}/{n_total_folds} "
                                f"total={elapsed:.1f}s "
                                f"| acc={_mean('accuracy'):.3f}±{_std('accuracy'):.3f} "
                                f"| f1={_mean('f1'):.3f}±{_std('f1'):.3f} "
                                f"| ll={_mean('log_loss'):.4f}±{_std('log_loss'):.4f} "
                                f"| ar_acc={_mean('ar_acc'):.3f} "
                                f"| si_acc={_mean('si_acc'):.3f}"
                                + (f" | mb={mb_size} fb={mb_fallback}"
                                   if fs == "markov" else ""),
                                flush=True,
                            )
                            gc.collect()

        tracker.summary()

        out = pd.DataFrame(rows)

        out_sorted = out.sort_values(
            by=["f1", "accuracy"],
            ascending=False, na_position="last"
        )

        print("\n=== TOP RESULTS ===")
        with pd.option_context("display.max_colwidth", 120):
            print(
                out_sorted.head(50)[[
                    "target", "K", "n_bins", "fs", "disc", "score",
                    "accuracy", "f1", "precision", "recall", "log_loss",
                    "persistence_accuracy", "persistence_f1",
                    "ar_dbn_accuracy", "ar_dbn_f1", "ar_dbn_log_loss",
                    "static_bn_si_accuracy", "static_bn_si_f1", "static_bn_si_log_loss",
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