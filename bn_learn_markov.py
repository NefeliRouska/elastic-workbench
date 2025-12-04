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

# -------------------------------------------------------------------
# CONFIGURATION
# -------------------------------------------------------------------
INPUT_CSV = "prom_all_metrics_wide.csv"                  # path to your input CSV
OUTPUT_EDGES = "BN_edges_contrained_hill_markov.csv"     # where to save the edges
OUTPUT_CPDS = "BN_edges_contrained_hill_markov_ALL_CPDS.csv"

DISCRETIZE = True                                        # set False to skip discretization
BINS = 3                                                 # number of discrete bins
MAX_INDEGREE = 4                                         # limit for DAG search

# Targets whose Markov blankets you care about
TARGETS = ["throughput"]

# Which BN structure search algorithm to use:
# "hillclimb" (uses blacklist) or "mmhc"
SEARCH_METHOD = "hillclimb"
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
# 6. (Optional) Discretize continuous metrics
# -------------------------------------------------------------------
if DISCRETIZE:
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            # keep low-cardinality vars (e.g. cores=1,2,3) as-is
            if df[c].nunique() <= 5:
                continue
            enc = KBinsDiscretizer(
                n_bins=BINS,
                encode='ordinal',
                strategy='quantile'
            )
            df[c] = enc.fit_transform(df[[c]]).astype(int)

# -------------------------------------------------------------------
# 7. Function to build blacklist with causal layering
# -------------------------------------------------------------------
def build_blacklist(df_local: pd.DataFrame):
    all_vars = list(df_local.columns)

    def has_prefix_any(c, prefixes):
        lc = c.lower()
        return any(lc.startswith(p) for p in prefixes)

    def has_substring_any(c, subs):
        lc = c.lower()
        return any(s in lc for s in subs)

    # 0) KNOBS (roots) – only if present
    layer0 = [v for v in all_vars if v in ["cores", "data_quality"]]

    # 1) PERFORMANCE – only if present
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
                # parent is in a "later" layer than child → forbid
                black_list_local.append((parent, child))

    print(f"Blacklist size (vars={len(all_vars)}): {len(black_list_local)}")
    return black_list_local

# -------------------------------------------------------------------
# 7b. Generic structure-learning wrapper
# -------------------------------------------------------------------
def learn_structure(df_local, max_indegree, black_list=None):
    """
    Choose structure learning algorithm via SEARCH_METHOD.
    - 'hillclimb' uses BIC + respects black_list.
    - 'mmhc' ignores the blacklist (pgmpy limitation).
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
        return est.estimate()
    else:
        raise ValueError(f"Unknown SEARCH_METHOD: {SEARCH_METHOD}")

# -------------------------------------------------------------------
# 8. FIRST STRUCTURE LEARNING on full df (to get MBs)
# -------------------------------------------------------------------
black_list_full = build_blacklist(df)

print(f"\n=== Learning FULL BN for Markov blankets with SEARCH_METHOD = '{SEARCH_METHOD}' ===")
best_model_full = learn_structure(
    df_local=df,
    max_indegree=MAX_INDEGREE,
    black_list=black_list_full
)

print("\nEdges of full model (for MB discovery):")
for e in best_model_full.edges():
    print(" ", e)

# -------------------------------------------------------------------
# 9. Build Markov blanket variable set for chosen TARGETS
# -------------------------------------------------------------------
mb_vars = set()

for t in TARGETS:
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

# Subset dataframe to only MB variables
df_mb = df[mb_vars].copy()

# -------------------------------------------------------------------
# 10. SECOND STRUCTURE LEARNING on MB-restricted df
# -------------------------------------------------------------------
black_list_mb = build_blacklist(df_mb)

print(f"\n=== Learning MB-RESTRICTED BN with SEARCH_METHOD = '{SEARCH_METHOD}' ===")
best_model_mb = learn_structure(
    df_local=df_mb,
    max_indegree=MAX_INDEGREE,
    black_list=black_list_mb
)

# -------------------------------------------------------------------
# 11. Fit Bayesian Network on MB data
# -------------------------------------------------------------------
model = BayesianModel(best_model_mb.edges())
model.fit(
    df_mb,
    estimator=BayesianEstimator,
    prior_type="BDeu",
    equivalent_sample_size=10,
)

# -------------------------------------------------------------------
# 12. Show and save learned edges
# -------------------------------------------------------------------
print("\nLearned edges (Markov blanket BN):")
for e in model.edges():
    print(" ", e)

edges = pd.DataFrame(model.edges(), columns=["from", "to"])
edges.to_csv(OUTPUT_EDGES, index=False)
print(f"\nSaved edges to: {OUTPUT_EDGES}")

# -------------------------------------------------------------------
# 13. Save ALL CPDs into ONE CSV (long format)
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
