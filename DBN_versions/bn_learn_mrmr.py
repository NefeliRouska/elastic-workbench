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

# mRMR library: pip install mrmr-selection
try:
    from mrmr import mrmr_classif
except ImportError:
    raise ImportError(
        "mrmr-selection package not found. Install with:\n"
        "  pip install mrmr-selection"
    )

# -------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------
INPUT_CSV = "prom_all_metrics_wide.csv"
OUTPUT_EDGES_MRMR = "BN_edges_contrained_hill_mrmr.csv"
OUTPUT_CPDS_MRMR = "BN_edges_contrained_hill_mrmr_ALL_CPDS.csv"

DISCRETIZE = True
BINS = 3
MAX_INDEGREE = 4

MRMR_TARGETS = ["throughput", "avg_p_latency", "cores", "data_quality"]
MRMR_K = 15

SEARCH_METHOD = "hillclimb"   # hillclimb uses blacklist

np.random.seed(42)
random.seed(42)

# -------------------------------------------------------------
# LOAD + CLEAN
# -------------------------------------------------------------
df = pd.read_csv(INPUT_CSV)

if "s_config" in df.columns:
    def parse_config(val):
        try:
            d = ast.literal_eval(val)
            if isinstance(d, dict):
                return d
        except:
            return {}
        return {}

    configs = df["s_config"].apply(parse_config).apply(pd.Series)
    configs = configs.rename(columns=lambda c: f"s_config_{c}")
    df = pd.concat([df.drop(columns=["s_config"]), configs], axis=1)

# drop timestamps
for col in list(df.columns):
    if "time" in col.lower():
        df = df.drop(columns=[col])

# drop constant cols + NA rows
df = df.loc[:, df.nunique(dropna=True) > 1].dropna()

# numeric conversion
df = df.apply(pd.to_numeric, errors="ignore")

# -------------------------------------------------------------
# DISCRETIZE (same as other scripts)
# -------------------------------------------------------------
if DISCRETIZE:
    for c in df.columns:
        if pd.api.types.is_numeric_dtype(df[c]):
            if df[c].nunique() <= 5:
                continue
            enc = KBinsDiscretizer(
                n_bins=BINS,
                encode='ordinal',
                strategy='quantile'
            )
            df[c] = enc.fit_transform(df[[c]]).astype(int)

# -------------------------------------------------------------
# BLACKLIST BUILDER (same as other scripts)
# -------------------------------------------------------------
def has_prefix_any(c, prefixes):
    lc = c.lower()
    return any(lc.startswith(p) for p in prefixes)

def has_substring_any(c, subs):
    lc = c.lower()
    return any(s in lc for s in subs)

def build_blacklist(vars_subset):
    all_vars = list(vars_subset)

    layer0 = [v for v in all_vars if v in ["cores", "data_quality"]]
    layer1 = [v for v in all_vars if v in ["avg_p_latency", "throughput", "buffer_size"]]

    layer2 = [
        v for v in all_vars
        if has_prefix_any(v, [
            "container_cpu_", "container_memory_", "container_network_", "container_fs_"])
        and v not in layer1
    ]

    layer3 = [v for v in all_vars if has_substring_any(v, ["fail", "oom", "scrape_error"])]

    layers = [layer0, layer1, layer2, layer3]

    layer_index = {}
    for idx, layer in enumerate(layers):
        for v in layer:
            layer_index[v] = idx

    black = []

    for child in layer0:
        for parent in all_vars:
            if parent != child:
                black.append((parent, child))

    if "cores" in layer0 and "data_quality" in layer0:
        black.append(("cores", "data_quality"))
        black.append(("data_quality", "cores"))

    for parent in all_vars:
        for child in all_vars:
            if parent == child:
                continue
            lp = layer_index.get(parent, None)
            lc = layer_index.get(child, None)
            if lp is not None and lc is not None and lp > lc:
                black.append((parent, child))

    return black

# -------------------------------------------------------------
# LEARN STRUCTURE (same wrapper as other scripts)
# -------------------------------------------------------------
def learn_structure(df_local, black_list):
    if SEARCH_METHOD == "hillclimb":
        est = HillClimbSearch(df_local)
        return est.estimate(
            scoring_method=BicScore(df_local),
            max_indegree=MAX_INDEGREE,
            black_list=black_list
        )
    elif SEARCH_METHOD == "mmhc":
        est = MmhcEstimator(df_local)
        return est.estimate()
    else:
        raise ValueError("Unknown SEARCH_METHOD")

# -------------------------------------------------------------
# mRMR feature selection (MULTI-TARGET)
# -------------------------------------------------------------
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

# final feature set
vars_mrmr = sorted(selected_union.union(MRMR_TARGETS))
df_mrmr = df[vars_mrmr].copy()

# -------------------------------------------------------------
# BUILD BLACKLIST + LEARN BN
# -------------------------------------------------------------
black_list_mrmr = build_blacklist(vars_mrmr)

best_model = learn_structure(df_mrmr, black_list_mrmr)

model = BayesianModel(best_model.edges())
model.fit(
    df_mrmr,
    estimator=BayesianEstimator,
    prior_type="BDeu",
    equivalent_sample_size=10
)

# -------------------------------------------------------------
# SAVE EDGES
# -------------------------------------------------------------
edges = pd.DataFrame(model.edges(), columns=["from", "to"])
edges.to_csv(OUTPUT_EDGES_MRMR, index=False)
print(f"\nSaved mRMR BN edges to: {OUTPUT_EDGES_MRMR}")

# -------------------------------------------------------------
# SAVE ALL CPDs IN ONE CSV
# -------------------------------------------------------------
all_rows = []

for cpd in model.get_cpds():
    var = cpd.variable
    values = np.array(cpd.values)

    # resolve state names
    state_names = getattr(cpd, "state_names", None)
    if state_names and var in state_names and state_names[var] is not None:
        var_states = list(state_names[var])
    else:
        var_states = [f"{var}_{i}" for i in range(values.shape[0])]

    try:
        evidence = list(cpd.get_evidence())
        evidence_card = list(cpd.get_evidence_card())
    except:
        evidence, evidence_card = [], []

    evidence_states = []
    for ev, card in zip(evidence, evidence_card):
        if state_names and ev in state_names and state_names[ev] is not None:
            ev_states = list(state_names[ev])
        else:
            ev_states = [f"{ev}_{i}" for i in range(card)]
        evidence_states.append(ev_states)

    if not evidence:
        # unconditional P(var)
        for s_idx, s_name in enumerate(var_states):
            all_rows.append({
                "variable": var,
                "variable_state": s_name,
                "prob": float(values[s_idx]),
            })
    else:
        # conditional P(var | parents)
        values_flat = values.reshape(len(var_states), -1)

        for col_idx, ev_combo in enumerate(itertools.product(*evidence_states)):
            for s_idx, s_name in enumerate(var_states):
                row = {
                    "variable": var,
                    "variable_state": s_name,
                    "prob": float(values_flat[s_idx, col_idx]),
                }
                for ev_name, ev_state in zip(evidence, ev_combo):
                    row[ev_name] = ev_state
                all_rows.append(row)

df_cpds = pd.DataFrame(all_rows)
df_cpds.to_csv(OUTPUT_CPDS_MRMR, index=False)
print(f"Saved ALL CPDs to: {OUTPUT_CPDS_MRMR}")
