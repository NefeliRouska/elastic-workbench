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

# -------------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------------
INPUT_CSV = "prom_all_metrics_wide.csv"       # path to your input CSV
OUTPUT_EDGES = "BN_edges_contrained_hill_normal.csv"  # where to save the edges
OUTPUT_CPDS = "BN_edges_contrained_hill_normal_ALL_CPDS.csv"

DISCRETIZE = True                             # set False to skip discretization
BINS = 3                                      # number of discrete bins
MAX_INDEGREE = 4                              # limit for DAG search

# Choose structure learning method: "hc" or "mmhc"
SEARCH_METHOD = "hc"                          # change to "mmhc" to use MMHC-style search
# -------------------------------------------------------------------

np.random.seed(42)
random.seed(42)

# -------------------------------------------------------------------
# 1. Load the data
# -------------------------------------------------------------------
df = pd.read_csv(INPUT_CSV)

# -------------------------------------------------------------------
# 2. Expand s_config (if present)
# -------------------------------------------------------------------
if 's_config' in df.columns:
    def parse_config(val):
        try:
            d = ast.literal_eval(val)
            if isinstance(d, dict):
                return d
        except Exception:
            return {}
        return {}

    configs = df['s_config'].apply(parse_config).apply(pd.Series)
    configs = configs.rename(columns=lambda c: f"s_config_{c}")
    df = pd.concat([df.drop(columns=['s_config']), configs], axis=1)

# -------------------------------------------------------------------
# 3. Delete time columns
# -------------------------------------------------------------------
for col in list(df.columns):
    if 'time' in col.lower():
        df = df.drop(columns=[col])

# -------------------------------------------------------------------
# 4. Drop constant or empty columns, and rows with NaN
# -------------------------------------------------------------------
df = df.loc[:, df.nunique(dropna=True) > 1].dropna()

# -------------------------------------------------------------------
# 5. Convert all remaining columns to numeric where possible BN need numeric not strings
# -------------------------------------------------------------------
df = df.apply(pd.to_numeric, errors='ignore')

# -------------------------------------------------------------------
# 6. (Optional) Discretize continuous metrics
#    Putting them into a few ordered categories (e.g. low/medium/high)
#    makes the structure learner focus on dependencies, not exact scale.
# -------------------------------------------------------------------

if DISCRETIZE:
    for c in df.columns:
        # only numeric columns
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue

        col = df[c].dropna()

        # 1) Keep already-discrete small-cardinality vars as they are
        if col.nunique() <= 5:
            continue

        vmin, vmax = col.min(), col.max()
        if vmin == vmax:
            # constant column – nothing to discretize
            continue

        n = len(col)

        # --- Freedman–Diaconis rule to get a data-dependent bin width ---
        q75, q25 = np.percentile(col, [75, 25])
        iqr = q75 - q25

        if iqr == 0:
            # fallback: just use up to 5 bins or number of unique values
            k = min(5, col.nunique())
        else:
            bin_width = 2 * iqr / (n ** (1/3))
            if bin_width <= 0:
                k = min(5, col.nunique())
            else:
                k = int(np.ceil((vmax - vmin) / bin_width))
                # bounds: at least 2 bins, at most 5
                k = max(2, min(k, 5))

        # now build k equal-width bins specific to THIS column
        bins = np.linspace(vmin, vmax, k + 1)

        df[c] = pd.cut(
            df[c],          # use original column (it has same range)
            bins=bins,
            labels=range(k),
            include_lowest=True
        ).astype(int)

"""enc = KBinsDiscretizer(n_bins=BINS, encode='ordinal', strategy='quantile')

if DISCRETIZE:
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            # IMPORTANT: do NOT discretize variables that already have low cardinality
            # (like 'cores' which has only values 1,2,3)
            if df[c].nunique() <= 5:
                continue  # keep original discrete values

            df[c] = enc.fit_transform(df[[c]]).astype(int)"""

# -------------------------------------------------------------------
# 7. GENERAL CAUSAL LAYERING: build a blacklist from name patterns
# -------------------------------------------------------------------
all_vars = list(df.columns)

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

# 2) RESOURCE USAGE: CPU, MEM, NET, FS (only those that appear)
layer2 = [
    v for v in all_vars
    if (
        has_prefix_any(v, ["container_cpu_", "container_memory_", "container_network_", "container_fs_"])
        and v not in layer1
    )
]

# 3) FAILURES / ERROR FLAGS
layer3 = [
    v for v in all_vars
    if has_substring_any(v, ["fail", "oom", "scrape_error"])
]

layers = [layer0, layer1, layer2, layer3]

# Map var -> layer index (if in any layer)
layer_index = {}
for idx, layer in enumerate(layers):
    for v in layer:
        layer_index[v] = idx

black_list = []

# A) knobs must have no parents  (forbids any edge into: cores and data_quality)
for child in layer0:
    for parent in all_vars:
        if parent != child:
            black_list.append((parent, child))

# Forbid edges between knobs themselves
# This additionally forbids: cores → data_quality, data_quality → cores
if "cores" in layer0 and "data_quality" in layer0:
    black_list.append(("cores", "data_quality"))
    black_list.append(("data_quality", "cores"))

# B) forbid edges from later layers to earlier layers
# For any two variables in layers: if parent_layer > child_layer → forbid parent→child
for parent in all_vars:
    for child in all_vars:
        if parent == child:
            continue
        lp = layer_index.get(parent, None)
        lc = layer_index.get(child, None)
        if lp is not None and lc is not None and lp > lc:
            black_list.append((parent, child))

print(f"Blacklist size: {len(black_list)}")

# -------------------------------------------------------------------
# 8. STRUCTURE LEARNING: HillClimb or MMHC
# -------------------------------------------------------------------
if SEARCH_METHOD.lower() == "hc":
    print("\nUsing HillClimbSearch + BIC + blacklist")
    hc = HillClimbSearch(df)
    best_model = hc.estimate(
        scoring_method=BicScore(df),
        max_indegree=MAX_INDEGREE,
        black_list=black_list
    )

elif SEARCH_METHOD.lower() == "mmhc":
    print("\nUsing MMHC-style search (MMPC skeleton + HillClimb + blacklist)")
    # MMPC skeleton using pgmpy's MmhcEstimator
    est = MmhcEstimator(df)
    skel = est.mmpc(significance_level=0.01)

    # Hill-climbing restricted to edges in the skeleton (white_list),
    # still respecting your causal blacklist and max_indegree.
    hc = HillClimbSearch(df)
    best_model = hc.estimate(
        scoring_method=BicScore(df),
        max_indegree=MAX_INDEGREE,
        white_list=skel.to_directed().edges(),
        black_list=black_list
    )
else:
    raise ValueError(f"Unknown SEARCH_METHOD: {SEARCH_METHOD}. Use 'hc' or 'mmhc'.")

# -------------------------------------------------------------------
# 9. Fit Bayesian Network
# -------------------------------------------------------------------
model = BayesianModel(best_model.edges())
model.fit(
    df,
    estimator=BayesianEstimator,
    prior_type="BDeu",
    equivalent_sample_size=10
)

# --- Inspect CPDs (conditional probability tables) ---
print("\n=== CPDs (Conditional Probability Distributions) ===")
for cpd in model.get_cpds():
    print(f"\nCPD of {cpd.variable}:")
    print(cpd)

# -------------------------------------------------------------------
# 10. Show and save learned edges
# -------------------------------------------------------------------
print("\nLearned edges:")
for e in model.edges():
    print(" ", e)

edges = pd.DataFrame(model.edges(), columns=['from', 'to'])
edges.to_csv(OUTPUT_EDGES, index=False)
print(f"\nSaved edges to: {OUTPUT_EDGES}")

# -------------------------------------------------------------------
# 11. Save ALL CPDs into ONE CSV (long format)
# For each variable: List all its states (0 / 1 / 2 or named). 
# List all combinations of parent states.
# For each combination:
# Add one row with: which variable, which state, which parent states, and the final probability number.
# -------------------------------------------------------------------
all_rows = []

for cpd in model.get_cpds():
    var = cpd.variable
    values = np.array(cpd.values)

    # Try to get state_names if they exist
    state_names = getattr(cpd, "state_names", None)

    # Variable states
    if state_names and var in state_names and state_names[var] is not None:
        var_states = list(state_names[var])
    else:
        var_states = [f"{var}_{i}" for i in range(values.shape[0])]

    # Evidence variables + cardinalities
    try:
        evidence = list(cpd.get_evidence())
        evidence_card = list(cpd.get_evidence_card())
    except AttributeError:
        evidence = list(getattr(cpd, "evidence", [])) or list(getattr(cpd, "variables", [])[1:])
        card_list = list(getattr(cpd, "cardinality", []))
        evidence_card = card_list[1:] if len(card_list) > 1 else []

    # Evidence states
    evidence_states = []
    for ev, card in zip(evidence, evidence_card):
        if state_names and ev in state_names and state_names[ev] is not None:
            ev_states = list(state_names[ev])
        else:
            ev_states = [f"{ev}_{i}" for i in range(card)]
        evidence_states.append(ev_states)

    # No parents → just P(var)
    if not evidence:
        for s_idx, s_name in enumerate(var_states):
            prob = float(values[s_idx])
            all_rows.append({
                "variable": var,
                "variable_state": s_name,
                "prob": prob
            })
    else:
        # Parents exist: flatten all parent dims into one
        values_flat = values.reshape(len(var_states), -1)

        for col_idx, ev_config in enumerate(itertools.product(*evidence_states)):
            for s_idx, s_name in enumerate(var_states):
                prob = float(values_flat[s_idx, col_idx])
                row = {
                    "variable": var,
                    "variable_state": s_name,
                    "prob": prob
                }
                for ev_name, ev_state in zip(evidence, ev_config):
                    row[ev_name] = ev_state
                all_rows.append(row)

if not all_rows:
    print("Warning: no CPDs found; CPD CSV not written.")
else:
    df_cpds = pd.DataFrame(all_rows)
    df_cpds.to_csv(OUTPUT_CPDS, index=False)
    print(f"\nSaved ALL CPDs (long format) to: {OUTPUT_CPDS}")
