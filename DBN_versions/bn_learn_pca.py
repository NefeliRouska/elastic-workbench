import pandas as pd
import numpy as np
import random
import ast
import itertools 
from pgmpy.estimators import HillClimbSearch, BicScore
from pgmpy.models import BayesianModel
from pgmpy.estimators import BayesianEstimator
from sklearn.preprocessing import KBinsDiscretizer
from pgmpy.estimators import PC, MmhcEstimator
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# -------------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------------
INPUT_CSV = "prom_all_metrics_wide.csv"                  # input data
OUTPUT_EDGES_PCA = "BN_edges_contrained_hill_pca.csv"   # where to save the edges
OUTPUT_CPDS_PCA = "BN_edges_contrained_hill_pca_ALL_CPDS.csv"  

DISCRETIZE = True           # BN needs discrete vars -> we discretize AFTER selection
BINS = 3                    # number of bins for discretization
MAX_INDEGREE = 4            # DAG search limit (same as other scripts)

# PCA feature-selection settings
PCA_VAR_THRESHOLD = 0.90    # keep enough PCs to explain 90% variance
# variables you always want to keep if present
MUST_HAVE = ["throughput", "avg_p_latency", "cores", "data_quality"]

# Which BN structure search algorithm to use:
# "hillclimb" (uses blacklist constraints), "pc", or "mmhc"
SEARCH_METHOD = "hillclimb"

# -------------------------------------------------------------------
# SEEDING (for reproducibility of PCA etc. – HillClimb may still
# have small non-determinism due to internal tie-breaking)
# -------------------------------------------------------------------
np.random.seed(42)
random.seed(42)

# -------------------------------------------------------------------
# 1. Load the data
# -------------------------------------------------------------------
df = pd.read_csv(INPUT_CSV)

# -------------------------------------------------------------------
# 2. Expand s_config (if present) – same logic as your other scripts
# -------------------------------------------------------------------
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

# -------------------------------------------------------------------
# 3. Drop timestamp-like columns
# -------------------------------------------------------------------
for col in list(df.columns):
    if "time" in col.lower():
        df = df.drop(columns=[col])

# -------------------------------------------------------------------
# 4. Drop constant or empty columns, and rows with NaN
# -------------------------------------------------------------------
df = df.loc[:, df.nunique(dropna=True) > 1].dropna()

# -------------------------------------------------------------------
# 5. Convert columns to numeric where possible
# -------------------------------------------------------------------
df = df.apply(pd.to_numeric, errors="ignore")

# -------------------------------------------------------------------
# 6. PCA-BASED FEATURE SELECTION (on continuous data, BEFORE discretization)
# -------------------------------------------------------------------
# Use only numeric columns for PCA
numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

if len(numeric_cols) == 0:
    raise ValueError("No numeric columns available for PCA.")

X = df[numeric_cols].values

# Standardize features for PCA
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Fit PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Determine number of components needed to explain PCA_VAR_THRESHOLD of variance
cumvar = np.cumsum(pca.explained_variance_ratio_)
num_components = np.searchsorted(cumvar, PCA_VAR_THRESHOLD) + 1
num_components = min(num_components, X.shape[1])

print(f"PCA: using {num_components} components to explain "
      f"{cumvar[num_components-1]:.3f} of variance")

# For each of these PCs, pick the original feature with the largest |loading|
selected_from_pca = []
for i in range(num_components):
    comp = pca.components_[i]  # loadings for component i
    idx = np.argmax(np.abs(comp))
    selected_from_pca.append(numeric_cols[idx])

# Deduplicate while preserving order
selected_from_pca = list(dict.fromkeys(selected_from_pca))

print("\nPCA-selected features (before MUST_HAVE union):")
for f in selected_from_pca:
    print("  ", f)

# Ensure domain-important variables are present if they exist in df
must_keep = [v for v in MUST_HAVE if v in df.columns]
selected_vars = list(dict.fromkeys(selected_from_pca + must_keep))

print("\nFinal feature set for BN (PCA + MUST_HAVE):")
for f in selected_vars:
    print("  ", f)

# Restrict dataframe to these variables (still continuous/raw)
df_pca = df[selected_vars].copy()

# -------------------------------------------------------------------
# 7. (Optional) Discretize for BN learning
#    Putting them into a few ordered categories (low / medium / high)
#    makes the structure learner focus on dependencies, not scale.
# -------------------------------------------------------------------
if DISCRETIZE:
    for c in df_pca.columns:
        if pd.api.types.is_numeric_dtype(df_pca[c]):
            # keep small-cardinality variables (e.g., cores=1/2/3) as is
            if df_pca[c].nunique() <= 5:
                continue
            enc = KBinsDiscretizer(
                n_bins=BINS,
                encode='ordinal',
                strategy='quantile'
            )
            df_pca[c] = enc.fit_transform(df_pca[[c]]).astype(int)

# -------------------------------------------------------------------
# 8. Function to build blacklist with your causal layering
# -------------------------------------------------------------------
def build_blacklist(df_local: pd.DataFrame):
    all_vars = list(df_local.columns)

    def has_prefix_any(c, prefixes):
        lc = c.lower()
        return any(lc.startswith(p) for p in prefixes)

    def has_substring_any(c, subs):
        lc = c.lower()
        return any(s in lc for s in subs)

    # 0) KNOBS (roots)
    layer0 = [v for v in all_vars if v in ["cores", "data_quality"]]

    # 1) PERFORMANCE
    perf_candidates = ["avg_p_latency", "throughput", "buffer_size"]
    layer1 = [v for v in all_vars if v in perf_candidates]

    # 2) RESOURCE USAGE: CPU, MEM, NET, FS
    layer2 = [
        v for v in all_vars
        if (
            has_prefix_any(v, ["container_cpu_", "container_memory_",
                               "container_network_", "container_fs_"])
            and v not in layer1
        )
    ]

    # 3) FAILURES / ERROR FLAGS
    layer3 = [
        v for v in all_vars
        if has_substring_any(v, ["fail", "oom", "scrape_error"])
    ]

    layers = [layer0, layer1, layer2, layer3]

    # Map variable -> layer index
    layer_index = {}
    for idx, layer in enumerate(layers):
        for v in layer:
            layer_index[v] = idx

    black_list_local = []

    # A) knobs must have no parents
    for child in layer0:
        for parent in all_vars:
            if parent != child:
                black_list_local.append((parent, child))

    # Forbid edges between knobs themselves
    if "cores" in layer0 and "data_quality" in layer0:
        black_list_local.append(("cores", "data_quality"))
        black_list_local.append(("data_quality", "cores"))

    # B) forbid edges from later layers to earlier layers
    for parent in all_vars:
        for child in all_vars:
            if parent == child:
                continue
            lp = layer_index.get(parent, None)
            lc = layer_index.get(child, None)
            if lp is not None and lc is not None and lp > lc:
                black_list_local.append((parent, child))

    print(f"Blacklist size (vars={len(all_vars)}): {len(black_list_local)}")
    return black_list_local

# -------------------------------------------------------------------
# 8b. Generic structure-learning wrapper
# -------------------------------------------------------------------
def learn_structure(df_local, max_indegree, black_list=None):
    """
    Choose structure learning algorithm via SEARCH_METHOD.
    - 'hillclimb' uses BIC + respects black_list.
    - 'pc' and 'mmhc' IGNORE black_list (pgmpy limitation).
    """
    if SEARCH_METHOD == "hillclimb":
        est = HillClimbSearch(df_local)
        return est.estimate(
            scoring_method=BicScore(df_local),
            max_indegree=max_indegree,
            black_list=black_list,
        )
    elif SEARCH_METHOD == "mmhc":
        est = MmhcEstimator(df_local)
        return est.estimate()                    # no blacklist argument here either
    else:
        raise ValueError(f"Unknown SEARCH_METHOD: {SEARCH_METHOD}")

# -------------------------------------------------------------------
# 9. Learn BN structure on PCA-selected subset
# -------------------------------------------------------------------
black_list_pca = build_blacklist(df_pca)

best_model_pca = learn_structure(
    df_local=df_pca,
    max_indegree=MAX_INDEGREE,
    black_list=black_list_pca
)

# -------------------------------------------------------------------
# 10. Fit Bayesian Network
# -------------------------------------------------------------------
model_pca = BayesianModel(best_model_pca.edges())
model_pca.fit(
    df_pca,
    estimator=BayesianEstimator,
    prior_type="BDeu",
    equivalent_sample_size=10,
)

# -------------------------------------------------------------------
# 11. Show and save learned edges
# -------------------------------------------------------------------
print("\nLearned edges (PCA-based BN):")
for e in model_pca.edges():
    print(" ", e)

edges_pca = pd.DataFrame(model_pca.edges(), columns=["from", "to"])
edges_pca.to_csv(OUTPUT_EDGES_PCA, index=False)
print(f"\nSaved PCA-based BN edges to: {OUTPUT_EDGES_PCA}")

# -------------------------------------------------------------------
# 12. Save ALL CPDs into ONE CSV (long format)  
# -------------------------------------------------------------------
all_rows = []

for cpd in model_pca.get_cpds():
    var = cpd.variable
    values = np.array(cpd.values)

    # state names (if available)
    state_names = getattr(cpd, "state_names", None)
    if state_names and var in state_names and state_names[var] is not None:
        var_states = list(state_names[var])
    else:
        var_states = [f"{var}_{i}" for i in range(values.shape[0])]

    # evidence
    try:
        evidence = list(cpd.get_evidence())
        evidence_card = list(cpd.get_evidence_card())
    except Exception:
        evidence = []
        evidence_card = []

    # evidence state names
    evidence_states = []
    for ev, card in zip(evidence, evidence_card):
        if state_names and ev in state_names and state_names[ev] is not None:
            ev_states = list(state_names[ev])
        else:
            ev_states = [f"{ev}_{i}" for i in range(card)]
        evidence_states.append(ev_states)

    # CPD with no parents
    if not evidence:
        for s_idx, s_name in enumerate(var_states):
            prob = float(values[s_idx])
            all_rows.append({
                "variable": var,
                "variable_state": s_name,
                "prob": prob,
            })
    else:
        # flatten parent states
        values_flat = values.reshape(len(var_states), -1)

        for combo_idx, ev_combo in enumerate(itertools.product(*evidence_states)):
            for s_idx, s_name in enumerate(var_states):
                prob = float(values_flat[s_idx, combo_idx])
                row = {
                    "variable": var,
                    "variable_state": s_name,
                    "prob": prob,
                }
                for ev_name, ev_state in zip(evidence, ev_combo):
                    row[ev_name] = ev_state
                all_rows.append(row)

if not all_rows:
    print("Warning: no CPDs found; CPD file not written.")
else:
    df_cpds = pd.DataFrame(all_rows)
    df_cpds.to_csv(OUTPUT_CPDS_PCA, index=False)
    print(f"\nSaved ALL CPDs (PCA-based BN) to: {OUTPUT_CPDS_PCA}")
