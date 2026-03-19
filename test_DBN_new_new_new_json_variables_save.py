import ast
import os
import pickle
import time
import gc
import numpy as np
import pandas as pd

from sklearn.preprocessing import KBinsDiscretizer
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression
from sklearn.cluster import DBSCAN

from pgmpy.inference import VariableElimination

# IMPORTANT: change this import to your actual create_dbn filename/module
from full_dynamic_bn_new_new_new import build_dbn_model_2s


# ============================================================
# CONFIG
# ============================================================
CSV_PATH = "share/metrics/dbn_wide_20260310_160246_seed161312.csv"
TRAIN_FRAC = 0.8
TARGETS = ["throughput_1", "throughput_2", "throughput_3"]
#TARGETS = ["throughput_1"]
# Sweep K (number of predictor features; TARGET is always included)
K_VALUES = [3, 8, 12, 20]

# Methods
FEATURE_SELECTION_METHODS = ["markov", "mrmr", "pca"]

# Discretization methods
DISCRETIZATION_METHODS = ["classic_uniform", "classic_quantile", "kmeans", "dbscan"]

# Structure score
SCORES = ["bic", "aic"]

# Discretization params
N_BINS = 4
DBSCAN_EPS = 0.30
DBSCAN_MIN_SAMPLES = 10

# Markov blanket stage (quick BN to estimate MB)
MB_QUICK_MAX_ITER = 8000

# Evidence modes
# - "full": all features at time t INCLUDING current TARGET
# - "full_no_target": all features at time t EXCLUDING current TARGET
# - "subset": specific columns only
EVIDENCE_MODE = "full_no_target"
EVIDENCE_SUBSET = ["avg_p_latency_", "cores_", "data_quality_", "buffer_size_"]

# Save only top N models per target
TOP_N_MODELS_TO_SAVE = 5
MODEL_SAVE_DIR = "saved_dbn_models_top5_20260310_160246_seed161312_5000"

# Output CSV
RESULTS_CSV_PATH = "dbn_k_sweep_results_top5_saved_20260310_160246_seed161312_5000.csv"


# ============================================================
# LOAD + CLEAN
# ============================================================
def load_and_clean(path):
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
        df = pd.concat([df.drop(columns=["s_config"]), cfg], axis=1)

    # Drop timestamp-like columns
    for col in list(df.columns):
        if "time" in col.lower():
            df.drop(columns=[col], inplace=True)

    #df = df.apply(pd.to_numeric, errors="ignore")
    #df = df.apply(lambda col: pd.to_numeric(col, errors="coerce")) #Try to convert everything to numbers. If something cannot be converted → replace it with NaN
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col])
        except Exception:
            pass
    # Keep numeric only
    df = df.select_dtypes(include=[np.number])
    df = df.dropna().reset_index(drop=True)

    if TARGET not in df.columns:
        raise ValueError(f"TARGET '{TARGET}' not found after cleaning.")

    return df


# ============================================================
# DISCRETIZATION
# ============================================================
class Discretizer:
    """
    Fits on train only; transforms train/test.

    Methods:
      - classic_uniform (KBins strategy="uniform")
      - classic_quantile (KBins strategy="quantile")
      - kmeans (KBins strategy="kmeans")
      - dbscan (DBSCAN per feature, then nearest-center binning)
    """

    def __init__(self, method, n_bins=4, dbscan_eps=0.3, dbscan_min_samples=10):
        self.method = method
        self.n_bins = n_bins
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_samples = dbscan_min_samples

        self.kbins = {}
        self.dbscan_centers = {}
        self.columns_ = None

        valid = {"classic_uniform", "classic_quantile", "kmeans", "dbscan"}
        if self.method not in valid:
            raise ValueError(f"Unknown discretizer method: {self.method}")

    def fit(self, df):
        self.columns_ = list(df.columns)

        if self.method in {"classic_uniform", "classic_quantile", "kmeans"}:
            if self.method == "classic_uniform":
                strategy = "uniform"
            elif self.method == "classic_quantile":
                strategy = "quantile"
            else:
                strategy = "kmeans"

            for c in self.columns_:
                kbd = KBinsDiscretizer(
                    n_bins=self.n_bins,
                    encode="ordinal",
                    strategy=strategy
                )
                kbd.fit(df[[c]])
                self.kbins[c] = kbd

        else:  # dbscan
            for c in self.columns_:
                x = df[c].to_numpy(dtype=float).reshape(-1, 1)
                db = DBSCAN(eps=self.dbscan_eps, min_samples=self.dbscan_min_samples)
                labels = db.fit_predict(x)

                uniq = sorted(set(labels) - {-1})
                if len(uniq) == 0:
                    self.dbscan_centers[c] = np.array([float(np.mean(x))])
                    continue

                centers = []
                for lab in uniq:
                    centers.append(float(np.mean(x[labels == lab])))
                centers = np.sort(np.array(centers))
                self.dbscan_centers[c] = centers

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
            x = out[c].to_numpy(dtype=float).reshape(-1, 1)
            diffs = np.abs(x - centers.reshape(1, -1))
            out[c] = np.argmin(diffs, axis=1).astype(int)

        return out.astype(int)


# ============================================================
# FEATURE SELECTION HELPERS
# ============================================================
def _split_X_y(df):
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    return X, y


def _top_k_by_mi(train_df, k):
    X, y = _split_X_y(train_df)
    mi = mutual_info_regression(X.to_numpy(), y.to_numpy(), random_state=0)
    order = np.argsort(mi)[::-1]
    cols = [X.columns[i] for i in order[:min(k, len(order))]]
    return cols


def select_mrmr(train_df, k):
    """
    Simple greedy mRMR using mutual_info_regression:
      score(f) = I(f; y) - mean_{s in selected} I(f; s)
    """
    X, y = _split_X_y(train_df)
    cols = list(X.columns)
    Xv = X.to_numpy()
    yv = y.to_numpy()

    rel = mutual_info_regression(Xv, yv, random_state=0)

    if len(cols) == 0:
        return []

    selected = []
    remaining = set(range(len(cols)))

    first = int(np.argmax(rel))
    selected.append(first)
    remaining.remove(first)

    while len(selected) < min(k, len(cols)) and remaining:
        best_j = None
        best_score = -1e18

        for j in list(remaining):
            red_vals = []
            for s in selected:
                mi_js = mutual_info_regression(Xv[:, [j]], Xv[:, s], random_state=0)
                red_vals.append(float(mi_js[0]))
            redundancy = float(np.mean(red_vals)) if red_vals else 0.0
            score = float(rel[j]) - redundancy
            if score > best_score:
                best_score = score
                best_j = j

        selected.append(best_j)
        remaining.remove(best_j)

    return [cols[i] for i in selected]


def _build_blacklist_single_slice(df):
    """
    Single-slice blacklist for the quick BN used in Markov blanket selection.
    Updated to correctly match your real variable names with suffixes like _1, _2, _3.
    """
    all_vars = list(df.columns)

    def has_prefix(c, prefixes):
        c = c.lower()
        return any(c.startswith(p) for p in prefixes)

    def has_substring(c, subs):
        c = c.lower()
        return any(s in c for s in subs)

    layer0 = [v for v in all_vars if has_prefix(v,["cores_", "data_quality_",])]

    layer1 = [v for v in all_vars if has_prefix(
        v,[ "container_cpu_", "container_memory_", "container_network_", "container_fs_", "container_blkio_",])]

    layer2 = [v for v in all_vars if has_prefix(v,["throughput_", "avg_p_latency_", "buffer_size_",])]

    layer3 = [v for v in all_vars if has_substring(v, ["fail", "oom", "scrape_error"])]

    layers = [layer0, layer1, layer2, layer3]
    layer_index = {v: i for i, L in enumerate(layers) for v in L}

    black = []

    # Nothing can point into layer0
    for child in layer0:
        for parent in all_vars:
            if parent != child:
                black.append((parent, child))

    # No edges from later layers to earlier layers
    for p in all_vars:
        for c in all_vars:
            if p == c:
                continue
            if p in layer_index and c in layer_index and layer_index[p] > layer_index[c]:
                black.append((p, c))

    return black


def select_markov_blanket(train_df, k, score_name):
    """
    Train-only Markov blanket selection:
      1) quick discretize train (uniform bins)
      2) learn single-slice BN
      3) take Markov blanket of TARGET
      4) trim/expand to exactly k using MI fallback
    """
    from pgmpy.estimators import HillClimbSearch, BicScore
    from pgmpy.models import BayesianNetwork

    # Drop constant columns before quick discretization, but always keep TARGET
    keep = []
    for c in train_df.columns:
        if c == TARGET:
            keep.append(c)
        elif train_df[c].nunique() > 1:
            keep.append(c)
    train_df = train_df[keep].copy()

    quick_disc = Discretizer(method="classic_uniform", n_bins=N_BINS)
    quick_disc.fit(train_df)
    train_disc = quick_disc.transform(train_df)

    bl = _build_blacklist_single_slice(train_disc)

    est = HillClimbSearch(train_disc)
    best = est.estimate(
        scoring_method=BicScore(train_disc),
        black_list=bl,
        max_iter=MB_QUICK_MAX_ITER,
        show_progress=False
    )

    mb_model = BayesianNetwork(best.edges())
    mb = mb_model.get_markov_blanket(TARGET)
    mb = [c for c in mb if c != TARGET]

    if len(mb) > k:
        mb_df = train_df[mb + [TARGET]].copy()
        top = _top_k_by_mi(mb_df, k)
        return top

    if len(mb) < k:
        mi_ranked = _top_k_by_mi(train_df, k=len(train_df.columns))
        mi_ranked = [c for c in mi_ranked if c not in mb]
        mb_extended = mb + mi_ranked
        return mb_extended[:min(k, len(mb_extended))]

    return mb


def apply_feature_selection(train_raw, test_raw, fs_method, k, score_name):
    """
    Returns train_fs, test_fs (continuous) with exactly k predictors + TARGET.
    """
    fs_method = fs_method.lower().strip()

    if fs_method == "mrmr":
        predictors = select_mrmr(train_raw, k)
        keep = predictors + [TARGET]
        return train_raw[keep].copy(), test_raw[keep].copy()

    if fs_method == "markov":
        predictors = select_markov_blanket(train_raw, k, score_name=score_name)
        keep = predictors + [TARGET]
        return train_raw[keep].copy(), test_raw[keep].copy()

    if fs_method == "pca":
        Xtr, ytr = _split_X_y(train_raw)
        Xte, yte = _split_X_y(test_raw)

        ncomp = min(k, Xtr.shape[1])
        pca = PCA(n_components=ncomp, random_state=0)
        pca.fit(Xtr.to_numpy())

        Ztr = pca.transform(Xtr.to_numpy())
        Zte = pca.transform(Xte.to_numpy())

        cols = [f"pca_{i+1}" for i in range(Ztr.shape[1])]
        train_fs = pd.DataFrame(Ztr, columns=cols).reset_index(drop=True)
        test_fs = pd.DataFrame(Zte, columns=cols).reset_index(drop=True)

        train_fs[TARGET] = ytr.reset_index(drop=True)
        test_fs[TARGET] = yte.reset_index(drop=True)

        return train_fs, test_fs

    raise ValueError("fs_method must be one of: markov | mrmr | pca")


# ============================================================
# MODEL SAVING
# ============================================================
def save_dbn_model(model_2s, target, k, fs_method, disc_method, score_name, rank=None):
    """
    Save:
      1) full trained DBN object as .pkl
      2) graph edges as .csv
    """
    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

    rank_prefix = f"rank{rank}_" if rank is not None else ""
    model_base = f"{rank_prefix}dbn_{target}_K{k}_{fs_method}_{disc_method}_{score_name}"

    model_pkl_path = os.path.join(MODEL_SAVE_DIR, model_base + ".pkl")
    edges_csv_path = os.path.join(MODEL_SAVE_DIR, model_base + "_edges.csv")

    with open(model_pkl_path, "wb") as f:
        pickle.dump(model_2s, f)

    edges = list(model_2s.edges())
    edges_df = pd.DataFrame(edges, columns=["parent", "child"])
    edges_df.to_csv(edges_csv_path, index=False)

    return model_pkl_path, edges_csv_path


# ============================================================
# EVALUATION
# ============================================================
def evaluate(model_2s, test_df):
    test_df = test_df.reset_index(drop=True)
    infer = VariableElimination(model_2s)

    if EVIDENCE_MODE == "subset":
        def has_prefix(c, prefixes):
            return any(c.startswith(p) for p in prefixes)
        ev_cols = [c for c in test_df.columns if has_prefix(c, EVIDENCE_SUBSET)]
    elif EVIDENCE_MODE == "full_no_target":
        #ev_cols = [c for c in test_df.columns if not c.startswith("throughput_")]
        ev_cols = [c for c in test_df.columns if c != TARGET]
    else:
        ev_cols = list(test_df.columns)

    model_nodes = set(model_2s.nodes())
    valid_cols = [c for c in ev_cols if f"{c}_t" in model_nodes]

    n = len(test_df) - 1
    if n <= 0:
        return {"accuracy": float("nan"), "mean_prob_true": float("nan")}

    correct = 0
    probs_true = []

    for t in range(n):
        evidence = {f"{c}_t": int(test_df.iloc[t][c]) for c in valid_cols}
        true_next = int(test_df.iloc[t + 1][TARGET])

        q = infer.query([f"{TARGET}_t1"], evidence=evidence, show_progress=False)
        probs = q.values / q.values.sum()

        pred = int(np.argmax(probs))
        probs_true.append(float(probs[true_next]))
        if pred == true_next:
            correct += 1

    return {
        "accuracy": correct / n,
        "mean_prob_true": float(np.mean(probs_true))
    }


# ============================================================
# ONE RUN
# ============================================================
def run_one(raw_df, fs_method, disc_method, score_name, k):
    split = int(TRAIN_FRAC * len(raw_df))
    train_raw = raw_df.iloc[:split].reset_index(drop=True)
    test_raw = raw_df.iloc[split:].reset_index(drop=True)

    # Feature selection (continuous, train-only)
    train_fs, test_fs = apply_feature_selection(train_raw, test_raw, fs_method, k, score_name)

    # Remove constant columns but always keep TARGET
    def drop_constant_cols(df):
        keep = []
        for c in df.columns:
            if c == TARGET:
                keep.append(c)
            elif df[c].nunique() > 1:
                keep.append(c)
        return df[keep]

    train_fs = drop_constant_cols(train_fs)
    test_fs = test_fs[train_fs.columns]

    # Discretization (train-only)
    disc = Discretizer(
        method=disc_method,
        n_bins=N_BINS,
        dbscan_eps=DBSCAN_EPS,
        dbscan_min_samples=DBSCAN_MIN_SAMPLES
    )
    disc.fit(train_fs)

    train_ready = disc.transform(train_fs)
    test_ready = disc.transform(test_fs)

    train_ready = drop_constant_cols(train_ready)
    test_ready = test_ready[train_ready.columns]

    # Safety checks
    if TARGET not in train_ready.columns or TARGET not in test_ready.columns:
        raise RuntimeError(f"TARGET missing after FS={fs_method} DISC={disc_method} K={k}")

    # Debug prints
    print("TRAIN_READY COLS:", train_ready.columns.tolist())
    print("TRAIN_READY LEN:", len(train_ready))

    # DBN training
    model_2s, *_ = build_dbn_model_2s(train_ready, score_name=score_name)

    # Evaluate
    res = evaluate(model_2s, test_ready)

    return res, list(train_ready.columns), model_2s


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
        print(f" RUNNING EXPERIMENTS FOR TARGET = {TARGET}")
        print("==============================\n")

        raw = load_and_clean(CSV_PATH)
        raw = raw.iloc[:5000].reset_index(drop=True) #### not full file
        rows = []

        for k in K_VALUES:
            for fs in FEATURE_SELECTION_METHODS:
                for disc in DISCRETIZATION_METHODS:
                    for sc in SCORES:
                        start = time.perf_counter()
                        try:
                            res, cols, model_2s = run_one(raw, fs, disc, sc, k)
                            elapsed = time.perf_counter() - start

                            rows.append({
                                "target": TARGET,
                                "K": k,
                                "fs": fs,
                                "disc": disc,
                                "score": sc,
                                "accuracy": res["accuracy"],
                                "mean_prob_true": res["mean_prob_true"],
                                "n_features_including_target": len(cols),
                                "features": cols,
                                "time_sec": elapsed,
                                "model_pkl_path": None,
                                "edges_csv_path": None,
                            })

                            print(
                                f"[OK] target={TARGET} K={k:2d} fs={fs:6s} "
                                f"disc={disc:15s} score={sc:3s} time={elapsed:.2f}s -> {res}"
                            )

                            del model_2s
                            gc.collect()

                        except Exception as e:
                            elapsed = time.perf_counter() - start

                            rows.append({
                                "target": TARGET,
                                "K": k,
                                "fs": fs,
                                "disc": disc,
                                "score": sc,
                                "accuracy": np.nan,
                                "mean_prob_true": np.nan,
                                "n_features_including_target": np.nan,
                                "features": None,
                                "time_sec": elapsed,
                                "model_pkl_path": None,
                                "edges_csv_path": None,
                                "error": str(e),
                            })

                            print(
                                f"[FAIL] target={TARGET} K={k} fs={fs} disc={disc} "
                                f"score={sc} time={elapsed:.2f}s: {e}"
                            )

                            gc.collect()

        out = pd.DataFrame(rows)

        out_sorted = out.sort_values(
            by=["accuracy", "mean_prob_true"],
            ascending=False,
            na_position="last"
        )

        print("\n=== TOP RESULTS ===")
        with pd.option_context("display.max_colwidth", 120):
            print(
                out_sorted.head(50)[
                    [
                        "target",
                        "K",
                        "fs",
                        "disc",
                        "score",
                        "accuracy",
                        "mean_prob_true",
                        "n_features_including_target",
                    ]
                ].to_string(index=False)
            )

        partial = pd.concat(all_results + [out], ignore_index=True)
        partial.to_csv(RESULTS_CSV_PATH, index=False)

        all_results.append(out)

    final = pd.concat(all_results, ignore_index=True)
    final.to_csv(RESULTS_CSV_PATH, index=False)

    print(f"\nSaved results CSV: {RESULTS_CSV_PATH}")

if __name__ == "__main__":
    main() 


""" # ============================================================
# MAIN
# ============================================================
def main():
    global TARGET

    os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
    all_results = []

    for tgt in TARGETS:
        TARGET = tgt
        print("\n==============================")
        print(f" RUNNING EXPERIMENTS FOR TARGET = {TARGET}")
        print("==============================\n")

        raw = load_and_clean(CSV_PATH)
        rows = []

        for k in K_VALUES:
            for fs in FEATURE_SELECTION_METHODS:
                for disc in DISCRETIZATION_METHODS:
                    for sc in SCORES:
                        start = time.perf_counter()
                        try:
                            res, cols, model_2s = run_one(raw, fs, disc, sc, k)
                            elapsed = time.perf_counter() - start

                            rows.append({
                                "target": TARGET,
                                "K": k,
                                "fs": fs,
                                "disc": disc,
                                "score": sc,
                                "accuracy": res["accuracy"],
                                "mean_prob_true": res["mean_prob_true"],
                                "n_features_including_target": len(cols),
                                "features": cols,
                                "time_sec": elapsed,
                                "model_pkl_path": None,
                                "edges_csv_path": None,
                                "_model_obj": model_2s,
                            })

                            print(
                                f"[OK] target={TARGET} K={k:2d} fs={fs:6s} "
                                f"disc={disc:15s} score={sc:3s} time={elapsed:.2f}s -> {res}"
                            )

                        except Exception as e:
                            elapsed = time.perf_counter() - start

                            rows.append({
                                "target": TARGET,
                                "K": k,
                                "fs": fs,
                                "disc": disc,
                                "score": sc,
                                "accuracy": np.nan,
                                "mean_prob_true": np.nan,
                                "n_features_including_target": np.nan,
                                "features": None,
                                "time_sec": elapsed,
                                "model_pkl_path": None,
                                "edges_csv_path": None,
                                "error": str(e),
                                "_model_obj": None,
                            })

                            print(
                                f"[FAIL] target={TARGET} K={k} fs={fs} disc={disc} "
                                f"score={sc} time={elapsed:.2f}s: {e}"
                            )

        out = pd.DataFrame(rows)

        # Rank in the same way you already print:
        # first accuracy, then mean_prob_true
        out_sorted = out.sort_values(
            by=["accuracy", "mean_prob_true"],
            ascending=False,
            na_position="last"
        )

        # Save only top N successful models for this target
        top_candidates = out_sorted[out_sorted["_model_obj"].notna()].head(TOP_N_MODELS_TO_SAVE)

        for rank, (idx, row) in enumerate(top_candidates.iterrows(), start=1):
            model_pkl_path, edges_csv_path = save_dbn_model(
                model_2s=row["_model_obj"],
                target=row["target"],
                k=int(row["K"]),
                fs_method=row["fs"],
                disc_method=row["disc"],
                score_name=row["score"],
                rank=rank
            )

            out.loc[idx, "model_pkl_path"] = model_pkl_path
            out.loc[idx, "edges_csv_path"] = edges_csv_path

        print("\n=== TOP RESULTS ===")
        with pd.option_context("display.max_colwidth", 120):
            print(
                out_sorted.head(50)[
                    [
                        "target",
                        "K",
                        "fs",
                        "disc",
                        "score",
                        "accuracy",
                        "mean_prob_true",
                        "n_features_including_target",
                    ]
                ].to_string(index=False)
            )

        # Drop model objects before saving CSV
        out = out.drop(columns=["_model_obj"])
        all_results.append(out)

    final = pd.concat(all_results, ignore_index=True)
    final.to_csv(RESULTS_CSV_PATH, index=False)

    print(f"\nSaved results CSV: {RESULTS_CSV_PATH}")
    print(f"Saved top {TOP_N_MODELS_TO_SAVE} models per target in: {MODEL_SAVE_DIR}")


"""