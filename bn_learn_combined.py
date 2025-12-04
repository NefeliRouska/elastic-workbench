import pandas as pd
import numpy as np
import random
import ast
import itertools

from pgmpy.estimators import HillClimbSearch, BicScore
from pgmpy.models import BayesianModel
from pgmpy.estimators import BayesianEstimator
from sklearn.preprocessing import KBinsDiscretizer
from pgmpy.estimators import MmhcEstimator
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# -------------------------------------------------------------------
# MASTER CONFIG
# -------------------------------------------------------------------
# Choose which pipeline to run:
#   "normal"  -> baseline BN (full data)
#   "markov"  -> Markov blanket preselection
#   "mrmr"    -> mRMR feature selection
#   "pca"     -> PCA-based feature selection
METHOD = "normal"   # change this to "markov" / "mrmr" / "pca"

INPUT_CSV = "prom_all_metrics_wide.csv"

# shared config
DISCRETIZE = True
BINS = 3
MAX_INDEGREE = 4

# use the SAME names everywhere: "hillclimb" or "mmhc"
SEARCH_METHOD_NORMAL = "hillclimb"   # "hillclimb" or "mmhc"
SEARCH_METHOD_MARKOV = "hillclimb"   # "hillclimb" or "mmhc"
SEARCH_METHOD_MRMR  = "hillclimb"    # "hillclimb" or "mmhc"
SEARCH_METHOD_PCA   = "hillclimb"    # "hillclimb" or "mmhc"

# Markov-blanket method targets
MARKOV_TARGETS = ["throughput", "avg_p_latency", "cores", "data_quality"]

# mRMR settings
MRMR_TARGETS = ["throughput", "avg_p_latency", "cores", "data_quality"]
MRMR_K = 15

# PCA feature-selection settings
PCA_VAR_THRESHOLD = 0.90
MUST_HAVE = ["throughput", "avg_p_latency", "cores", "data_quality"]

# Output files per method (same names as your original scripts)
OUTPUT_EDGES_BY_METHOD = {
    "normal": "BN_edges_contrained_hill_normal.csv",
    "markov": "BN_edges_contrained_hill_markov.csv",
    "mrmr":   "BN_edges_contrained_hill_mrmr.csv",
    "pca":    "BN_edges_contrained_hill_pca.csv",
}

OUTPUT_CPDS_BY_METHOD = {
    "normal": "BN_edges_contrained_hill_normal_ALL_CPDS.csv",
    "markov": "BN_edges_contrained_hill_markov_ALL_CPDS.csv",
    "mrmr":   "BN_edges_contrained_hill_mrmr_ALL_CPDS.csv",
    "pca":    "BN_edges_contrained_hill_pca_ALL_CPDS.csv",
}

# -------------------------------------------------------------------
# GLOBAL SEEDS
# -------------------------------------------------------------------
np.random.seed(42)
random.seed(42)

# -------------------------------------------------------------------
# LOAD + BASIC CLEANING (shared by all methods)
# -------------------------------------------------------------------
def load_and_basic_clean(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Expand s_config if present
    if "s_config" in df.columns:
        def parse_config(val):
            try:
                d = ast.literal_eval(val)
                if isinstance(d, dict):
                    return d
            except Exception:
                return {}
            return {}

        configs = df["s_config"].apply(parse_config).apply(pd.Series)
        configs = configs.rename(columns=lambda c: f"s_config_{c}")
        df = pd.concat([df.drop(columns=["s_config"]), configs], axis=1)

    # Drop time columns / if I have a timestamp as a variable, the model could learn "Time causes Latency," which is true, but it obscures the root cause.
    for col in list(df.columns):
        if "time" in col.lower():
            df = df.drop(columns=[col])

    # Drop constant or empty columns, and rows with NaN
    df = df.loc[:, df.nunique(dropna=True) > 1].dropna()

    # Convert to numeric where possible
    df = df.apply(pd.to_numeric, errors="ignore")

    return df

# -------------------------------------------------------------------
# COMMON: DISCRETIZATION HELPERS
# -------------------------------------------------------------------
def discretize_fd(df: pd.DataFrame) -> pd.DataFrame:
    """
    Freedman–Diaconis-based: It looks at the spread of the data (Interquartile Range - IQR).
    It creates adaptive bin widths. If a variable has a huge spike at 0 and a long tail, FD handles it better than simple splitting.
    It is statistically the "smartest" method because it preserves the natural distribution of your data.
    """
    if not DISCRETIZE:
        return df

    for c in df.columns:
        if not pd.api.types.is_numeric_dtype(df[c]):  # Only apply discretization to numeric types
            continue
        col = df[c].dropna()
        if col.nunique() <= 5: # If the variable already has 5 or fewer unique values, treat it as categorical and leave it untouched.
            continue

        vmin, vmax = col.min(), col.max() # Compute min and max to check if the variable is constant
        if vmin == vmax: #no variation-skip
            continue

        n = len(col) # Number of observations
        q75, q25 = np.percentile(col, [75, 25]) # Compute IQR (75th percentile − 25th percentile) / measures spread while ignoring extreme outliers.
        iqr = q75 - q25

        if iqr == 0:
            k = min(5, col.nunique())
        else:
            bin_width = 2 * iqr / (n ** (1 / 3)) # Compute bin width using FD formula
            if bin_width <= 0: # Defensive fallback (should rarely happen)
                k = min(5, col.nunique())
            else:
                k = int(np.ceil((vmax - vmin) / bin_width)) # Raw estimated number of bins
                k = max(2, min(k, 5)) # Choose between [2, 5] to avoid too many categories. 2 = minimum reasonable, 5 = keeps BN efficient

        bins = np.linspace(vmin, vmax, k + 1)

        df[c] = pd.cut( #disretization pd.cut assigns each value to one of the k intervals labels=range(k) gives categories 0,1,2,...,k-1 include_lowest=True ensures the min value is included
            df[c],
            bins=bins,
            labels=range(k),
            include_lowest=True #(bin_start, bin_end]
        ).astype(int)

    return df


def discretize_kbins(df: pd.DataFrame) -> pd.DataFrame:
    """
    KBinsDiscretizer-based, like in Markov/mRMR/PCA scripts.
    Modifies and returns df.
    """
    if not DISCRETIZE:
        return df

    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            if df[c].nunique() <= 5:
                continue
            enc = KBinsDiscretizer(
                n_bins=BINS,
                encode="ordinal",
                strategy="quantile",
            )
            df[c] = enc.fit_transform(df[[c]]).astype(int)
    return df

# -------------------------------------------------------------------
# BLACKLIST BUILDER
# -------------------------------------------------------------------
def build_blacklist(df_local: pd.DataFrame):
    all_vars = list(df_local.columns)

    def has_prefix_any(c, prefixes):
        lc = c.lower()
        return any(lc.startswith(p) for p in prefixes)

    def has_substring_any(c, subs):
        lc = c.lower()
        return any(s in lc for s in subs)

    # ---- LAYER 0: KNOBS & LOAD ----
    # Treat throughput as incoming load, not as a performance result
    layer0 = [v for v in all_vars if v in ["cores", "data_quality"]]

    # ---- LAYER 1: RESOURCES ----
    layer1 = [
        v for v in all_vars
        if has_prefix_any(
            v,
            ["container_cpu_", "container_memory_", "container_network_", "container_fs_"]
        )
    ]

    # ---- LAYER 2: PERFORMANCE METRICS ----
    perf_candidates = ["throughput", "avg_p_latency", "buffer_size"]
    layer2 = [v for v in all_vars if v in perf_candidates]

    # ---- LAYER 3: FAILURES / ERROR FLAGS ----
    layer3 = [
        v for v in all_vars
        if has_substring_any(v, ["fail", "oom", "scrape_error"])
    ]

    layers = [layer0, layer1, layer2, layer3]

    # Map var -> layer index
    layer_index = {}
    for idx, layer in enumerate(layers):
        for v in layer:
            layer_index[v] = idx

    black_list_local = []

    # A) knobs/load must have no parents (no incoming edges into layer0)
    for child in layer0:
        for parent in all_vars:
            if parent != child:
                black_list_local.append((parent, child))

    # B) forbid edges between knobs/load themselves
    # (if you ever want throughput -> cores, you can relax this)
    if "cores" in layer0 and "data_quality" in layer0:
        black_list_local.append(("cores", "data_quality"))
        black_list_local.append(("data_quality", "cores"))

    # C) forbid edges from later layers to earlier layers
    for parent in all_vars:
        for child in all_vars:
            if parent == child:
                continue
            lp = layer_index.get(parent, None)
            lc = layer_index.get(child, None)
            if lp is not None and lc is not None and lp > lc:
                # parent is in a "later" layer than child -> forbid
                black_list_local.append((parent, child))

    print(f"Blacklist size (vars={len(all_vars)}): {len(black_list_local)}")
    return black_list_local

# -------------------------------------------------------------------
# COMMON: STRUCTURE LEARNING
# -------------------------------------------------------------------
def estimate_structure(df_local: pd.DataFrame,
                       search_method: str,
                       black_list=None,
                       method_name: str = None,
                       use_normal_mmhc_skeleton: bool = False):
    """
    Unifies:
    - NORMAL script:
      * 'hillclimb'  -> HillClimb + blacklist
      * 'mmhc' -> MMPC skeleton + HillClimb + blacklist
    - Other scripts:
      * 'hillclimb' -> HillClimb + blacklist
      * 'mmhc' -> MMHC estimator (ignores blacklist)
    """
    sm = search_method.lower()

    if sm == "hillclimb":
        est = HillClimbSearch(df_local)
        return est.estimate(
            scoring_method=BicScore(df_local),
            max_indegree=MAX_INDEGREE,
            black_list=black_list,
        )

    elif sm == "mmhc":
        if method_name == "normal" and use_normal_mmhc_skeleton:
            # your NORMAL script's MMHC-style
            est = MmhcEstimator(df_local)
            skel = est.mmpc(significance_level=0.01)
            hc = HillClimbSearch(df_local)
            return hc.estimate(
                scoring_method=BicScore(df_local),
                max_indegree=MAX_INDEGREE,
                white_list=skel.to_directed().edges(),
                black_list=black_list
            )
        else:
            # other scripts: plain MMHC
            est = MmhcEstimator(df_local)
            return est.estimate()
    else:
        raise ValueError(f"Unknown search method: {search_method}")

# -------------------------------------------------------------------
# COMMON: SAVE CPDs (long format)
# -------------------------------------------------------------------
def save_all_cpds_long(model: BayesianModel, output_path: str):
    all_rows = []

    for cpd in model.get_cpds():
        var = cpd.variable
        values = np.array(cpd.values)

        state_names = getattr(cpd, "state_names", None)
        if state_names and var in state_names and state_names[var] is not None:
            var_states = list(state_names[var])
        else:
            var_states = [f"{var}_{i}" for i in range(values.shape[0])]

        try:
            evidence = list(cpd.get_evidence())
            evidence_card = list(cpd.get_evidence_card())
        except AttributeError:
            evidence = list(getattr(cpd, "evidence", [])) or list(getattr(cpd, "variables", [])[1:])
            card_list = list(getattr(cpd, "cardinality", []))
            evidence_card = card_list[1:] if len(card_list) > 1 else []

        evidence_states = []
        for ev, card in zip(evidence, evidence_card):
            if state_names and ev in state_names and state_names[ev] is not None:
                ev_states = list(state_names[ev])
            else:
                ev_states = [f"{ev}_{i}" for i in range(card)]
            evidence_states.append(ev_states)

        if not evidence:
            for s_idx, s_name in enumerate(var_states):
                prob = float(values[s_idx])
                all_rows.append({
                    "variable": var,
                    "variable_state": s_name,
                    "prob": prob
                })
        else:
            values_flat = values.reshape(len(var_states), -1)
            for col_idx, ev_combo in enumerate(itertools.product(*evidence_states)):
                for s_idx, s_name in enumerate(var_states):
                    prob = float(values_flat[s_idx, col_idx])
                    row = {
                        "variable": var,
                        "variable_state": s_name,
                        "prob": prob
                    }
                    for ev_name, ev_state in zip(evidence, ev_combo):
                        row[ev_name] = ev_state
                    all_rows.append(row)

    if not all_rows:
        print("Warning: no CPDs found; CPD CSV not written.")
    else:
        df_cpds = pd.DataFrame(all_rows)
        df_cpds.to_csv(output_path, index=False)
        print(f"\nSaved ALL CPDs (long format) to: {output_path}")

# -------------------------------------------------------------------
# SHARED TAIL: BLACKLIST -> STRUCTURE -> FIT -> SAVE
# -------------------------------------------------------------------
def bn_tail(df_ready: pd.DataFrame,
            method_name: str,
            search_method: str):
    """
    df_ready must already be:
    - cleaned
    - possibly feature-selected (MB/mRMR/PCA)
    - discretized (if needed)
    """
    edges_path = OUTPUT_EDGES_BY_METHOD[method_name]
    cpds_path = OUTPUT_CPDS_BY_METHOD[method_name]

    black_list = build_blacklist(df_ready)

    best_model = estimate_structure(
        df_local=df_ready,
        search_method=search_method,
        black_list=black_list,
        method_name=method_name,
        use_normal_mmhc_skeleton=True  # only used when method_name=="normal"
    )

    model = BayesianModel(best_model.edges())
    model.fit(
        df_ready,
        estimator=BayesianEstimator,
        prior_type="BDeu",
        equivalent_sample_size=10
    )

    print(f"\nLearned edges ({method_name}):")
    for e in model.edges():
        print(" ", e)

    edges = pd.DataFrame(model.edges(), columns=["from", "to"])
    edges.to_csv(edges_path, index=False)
    print(f"\nSaved edges to: {edges_path}")

    save_all_cpds_long(model, cpds_path)

# -------------------------------------------------------------------
# METHOD-SPECIFIC "BEGINNINGS"
# -------------------------------------------------------------------
def prepare_normal(df_raw: pd.DataFrame):
    df = df_raw.copy()
    df = discretize_fd(df)
    return df, SEARCH_METHOD_NORMAL

def prepare_markov(df_raw: pd.DataFrame):
    df = df_raw.copy()
    # 1) discretize like your Markov script
    df = discretize_fd(df)

    # 2) full BN just to get Markov blankets
    black_list_full = build_blacklist(df)
    best_model_full = estimate_structure(
        df_local=df,
        search_method=SEARCH_METHOD_MARKOV,
        black_list=black_list_full,
        method_name="markov",
        use_normal_mmhc_skeleton=False
    )

    print("\nEdges of full model (for MB discovery):")
    for e in best_model_full.edges():
        print(" ", e)

    mb_vars = set()
    for t in MARKOV_TARGETS:
        if t not in df.columns:
            print(f"Warning: target {t} not in dataframe, skipping.")
            continue
        if t not in best_model_full.nodes():
            print(f"Warning: target {t} not in learned model, skipping.")
            continue
        mb_vars.add(t)
        try:
            mb = best_model_full.get_markov_blanket(t)
            mb_vars.update(mb)
        except ValueError:
            print(f"Warning: could not get Markov blanket for {t}.")

    mb_vars = sorted(mb_vars)
    print("\nMarkov blanket variable set:")
    print(mb_vars)

    df_mb = df[mb_vars].copy()
    return df_mb, SEARCH_METHOD_MARKOV

def prepare_mrmr(df_raw: pd.DataFrame):
    df = df_raw.copy()

    # mRMR library only needed in this method
    try:
        from mrmr import mrmr_classif
    except ImportError:
        raise ImportError(
            "mrmr-selection package not found. Install with:\n"
            "  pip install mrmr-selection"
        )

    # discretize first (as in your mRMR script)
    df = discretize_fd(df)

    selected_union = set()
    for target in MRMR_TARGETS:
        if target not in df.columns:
            print(f"Skipping missing target {target}")
            continue

        X = df.drop(columns=[target])
        y = df[target]
        K_eff = min(MRMR_K, X.shape[1])
        feats = mrmr_classif(X=X, y=y, K=K_eff)

        print(f"\nSelected for target {target}: {feats}")
        selected_union.update(feats)

    vars_mrmr = sorted(selected_union.union(MRMR_TARGETS))
    df_mrmr = df[vars_mrmr].copy()
    print("\nmRMR final feature set:")
    print(vars_mrmr)

    return df_mrmr, SEARCH_METHOD_MRMR

def prepare_pca(df_raw: pd.DataFrame):
    df = df_raw.copy()

    # PCA on continuous (non-discretized) data, like your PCA script
    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    if len(numeric_cols) == 0:
        raise ValueError("No numeric columns available for PCA.")

    X = df[numeric_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)

    cumvar = np.cumsum(pca.explained_variance_ratio_)
    num_components = np.searchsorted(cumvar, PCA_VAR_THRESHOLD) + 1
    num_components = min(num_components, X.shape[1])

    print(f"PCA: using {num_components} components to explain "
          f"{cumvar[num_components-1]:.3f} of variance")

    selected_from_pca = []
    for i in range(num_components):
        comp = pca.components_[i]
        idx = np.argmax(np.abs(comp))
        selected_from_pca.append(numeric_cols[idx])

    selected_from_pca = list(dict.fromkeys(selected_from_pca))
    print("\nPCA-selected features (before MUST_HAVE union):")
    for f in selected_from_pca:
        print("  ", f)

    must_keep = [v for v in MUST_HAVE if v in df.columns]
    selected_vars = list(dict.fromkeys(selected_from_pca + must_keep))
    print("\nFinal feature set for BN (PCA + MUST_HAVE):")
    for f in selected_vars:
        print("  ", f)

    df_pca = df[selected_vars].copy()

    # now discretize (same style as your PCA script)
    df_pca = discretize_fd(df_pca)

    return df_pca, SEARCH_METHOD_PCA

# -------------------------------------------------------------------
# MAIN DISPATCH
# -------------------------------------------------------------------
def main():
    print(f"\n=== Running METHOD = '{METHOD}' ===")
    df_raw = load_and_basic_clean(INPUT_CSV)

    if METHOD == "normal":
        df_ready, search_method = prepare_normal(df_raw)
    elif METHOD == "markov":
        df_ready, search_method = prepare_markov(df_raw)
    elif METHOD == "mrmr":
        df_ready, search_method = prepare_mrmr(df_raw)
    elif METHOD == "pca":
        df_ready, search_method = prepare_pca(df_raw)
    else:
        raise ValueError(f"Unknown METHOD: {METHOD}")

    # Shared tail: blacklist -> structure -> BN -> edges + CPDs
    bn_tail(df_ready, METHOD, search_method)

if __name__ == "__main__":
    main()
