# Robust KL between two Bayesian Networks
# The script learns two Bayesian Networks (baseline and perturbed) from your metrics,
# then uses Monte Carlo sampling to estimate how different their probability structures are,
# in a way that is mathematically correct, numerically stable, and independent of your dataset’s quirks.

import numpy as np
import pandas as pd
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import BayesianEstimator
from pgmpy.sampling import BayesianModelSampling

# ------------------ CONFIG ------------------
BASELINE_CSV = "baseline_merged.csv"
PERTURB_CSV  = "baseline_merged_perturbation.csv"
EDGES_EXP    = "bn_edges.csv"                  # expected/baseline structure
EDGES_OBS    = "bn_edges_perturbation.csv"     # observed/perturbed structure

N_BINS     = 2       # 3–4 is stable; fewer bins => denser CPDs
ESS        = 100     # BDeu smoothing (larger => safer)
N_SAMPLES  = 50000   # Monte Carlo samples per direction
RNG_SEED   = 7

# ------------------ HELPERS ------------------
def drop_timestamp_and_non_numeric(df: pd.DataFrame) -> pd.DataFrame:                   #Removes any "timestamp" column (not needed for probability modeling) / Keeps only numeric columns, since Prometheus metrics are numeric
    # drop 'timestamp' and keep ONLY numeric columns (Prometheus metrics are numeric)
    for c in list(df.columns):
        if c.lower() == "timestamp":
            df = df.drop(columns=[c])
    num_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    return df[num_cols].copy()
#clean table of metrics ready to discretize.


def load_edges(path: str):                  #Reads the CSV file that lists your BN edges (source,target pairs). Returns a list of tuples like [("CPU Usage", "Throughput"), ...].
    df = pd.read_csv(path)
    if {"source","target"}.issubset(df.columns):
        uv = df[["source","target"]].astype(str).values
    else:
        uv = df.iloc[:, :2].astype(str).values
    return [(u, v) for u, v in uv if u != v]
#The script knows the BN structure (which variables depend on which).

def strict_edges_from_series(s: pd.Series, n_bins: int) -> np.ndarray:              # strictly increasing quantile edges, even for constant columns
    a = s.to_numpy()
    if np.all(pd.isna(a)):
        return np.array([0.0, 1.0, 2.0], float)
    q = np.linspace(0, 1, n_bins + 1)
    E = np.nanquantile(a, q).astype(float)
    if not np.all(np.diff(E) > 0):
        mn, mx = np.nanmin(a), np.nanmax(a)
        if not np.isfinite(mn): mn = 0.0
        if not np.isfinite(mx): mx = mn + 1.0
        if mn == mx:
            v = float(mn)
            E = np.array([v - 1.0, v, v + 1.0], float)
        else:
            E = np.linspace(mn, mx, max(3, n_bins + 1))
    # enforce strict monotonicity
    eps = np.finfo(float).eps
    for i in range(1, len(E)):
        if E[i] <= E[i-1]:
            E[i] = E[i-1] + eps
    return E
#Prevents errors like “bins must increase monotonically.

def discretize_with_labels_and_scaffold(df_base: pd.DataFrame, df_other: pd.DataFrame, n_bins: int):
    """
    Learns discretization edges (the bin cut points) from the baseline data only.
    Applies the same labeled bins (“bin1”, “bin2”, …) to both datasets, so they are comparable.
    Adds tiny scaffolding rows so every possible label appears at least once in each dataset (so all probabilities exist in the BN).
    """
    cols = list(df_base.columns)  # numeric only by construction
    # 1) learn edges per column on baseline
    bins = {c: strict_edges_from_series(df_base[c], n_bins) for c in cols}
    labels_map = {c: [f"bin{i}" for i in range(1, len(bins[c]))] for c in cols}

    # 2) discretize with fixed labels
    def apply(df):
        out = pd.DataFrame(index=df.index)
        for c in cols:
            edges = bins[c]; labels = labels_map[c]
            out[c] = pd.cut(df[c], bins=edges, include_lowest=True, duplicates="drop", labels=labels).astype(str)
        out = out.dropna(how="any")
        return out

    A = apply(df_base)
    B = apply(df_other)

    # 3) add scaffolding rows per dataset so each label appears at least once for every column
    def scaffold(df):
        if df.empty:
            # Create a single dummy row with the first label for each column
            dummy = {c: labels_map[c][0] for c in cols}
            return pd.DataFrame([dummy])
        base_row = df.iloc[0].to_dict()
        add_rows = []
        for c in cols:
            have = set(df[c].unique())
            need = [lab for lab in labels_map[c] if lab not in have]
            for lab in need:
                row = base_row.copy()
                row[c] = lab
                add_rows.append(row)
        if add_rows:
            df = pd.concat([df, pd.DataFrame(add_rows)], ignore_index=True)
        return df

    A = scaffold(A)
    B = scaffold(B)
    return A, B
#Both datasets have the same structure and label sets

def fit_model(edges, df_disc: pd.DataFrame, ess: int) -> BayesianNetwork:
    # Use only nodes that exist in the discretized data
    nodes = sorted((set(u for u, _ in edges) | set(v for _, v in edges)) & set(df_disc.columns))
    edges_ok = [(u, v) for (u, v) in edges if u in nodes and v in nodes and u != v]
    model = BayesianNetwork(edges_ok)
    if nodes:
        model.fit(df_disc[nodes], estimator=BayesianEstimator,
                  prior_type="BDeu", equivalent_sample_size=ess)
    return model
#a valid, smoothed BN ready for probability computations.

def fast_log_prob(model: BayesianNetwork, X: pd.DataFrame) -> np.ndarray:
    # log P_model(X) assuming X has valid state names (we ensure by sampling)
    n = len(X)
    logs = np.zeros(n, dtype=float)
    cpds = {cpd.variable: cpd for cpd in model.get_cpds()}

    for var in model.nodes():
        if var not in X.columns:
            continue
        parents = list(model.get_parents(var))
        if not set(parents).issubset(X.columns):
            continue

        cpd = cpds[var]
        vals = np.asarray(cpd.values)
        vmap = {s: i for i, s in enumerate(cpd.state_names[var])}
        vi = X[var].astype(str).map(vmap).to_numpy().astype(int)

        if not parents:
            probs = vals[vi] if vals.ndim == 1 else vals[vi, 0]
        else:
            pidx, cards = [], []
            for p in parents:
                pmap = {s: i for i, s in enumerate(cpd.state_names[p])}
                pi = X[p].astype(str).map(pmap).to_numpy().astype(int)
                pidx.append(pi)
                cards.append(len(cpd.state_names[p]))
            pidx = np.vstack(pidx)
            strides = (np.array([1], int) if len(cards) == 1
                       else np.array([int(np.prod(cards[i+1:])) for i in range(len(cards))], int))
            lin = (pidx.T * strides).sum(axis=1).astype(int)

            probs = vals[vi] if vals.ndim == 1 else vals[vi, 0 if vals.shape[1] == 1 else lin]

        logs += np.log(probs + 1e-12)

    return logs
#For each row, you know how “surprised” the model is — lower log-prob means more surprise.

def KL_mc(P: BayesianNetwork, Q: BayesianNetwork, N: int, seed: int) -> float:
    """
    Monte Carlo KL: E_{x~P}[log P(x) - log Q(x)].
    We sample from P (so rows are valid for P), then restrict to variables
    that BOTH models can score given parents present in the sampled frame.
    """
    X = BayesianModelSampling(P).forward_sample(size=N, seed=seed)

    # Variables both can score (variable + its parents present in X)
    colsP = [v for v in P.nodes() if set(P.get_parents(v)).issubset(X.columns)]
    colsQ = [v for v in Q.nodes() if set(Q.get_parents(v)).issubset(X.columns)]
    keep = sorted(set(colsP) & set(colsQ))
    if not keep:
        raise RuntimeError("No common scorable variables between models.")
    X = X[keep]

    lp = fast_log_prob(P, X)
    lq = fast_log_prob(Q, X)
    return float(np.mean(lp - lq))
#standard estimator of KL divergence.

# ------------------ MAIN ------------------
def main():
    # 1) Load data; drop timestamp; keep numeric only; align columns
    df_base_raw = drop_timestamp_and_non_numeric(pd.read_csv(BASELINE_CSV))
    df_pert_raw = drop_timestamp_and_non_numeric(pd.read_csv(PERTURB_CSV))
    common_cols = sorted(set(df_base_raw.columns) & set(df_pert_raw.columns))
    df_base_raw = df_base_raw[common_cols].copy()
    df_pert_raw = df_pert_raw[common_cols].copy()

    # 2) Discretize with SAME labeled bins learned on baseline + scaffold
    df_base, df_pert = discretize_with_labels_and_scaffold(df_base_raw, df_pert_raw, N_BINS)

    # 3) Load structures; fit models on nodes that exist in the data
    edges_exp = load_edges(EDGES_EXP)
    edges_obs = load_edges(EDGES_OBS)
    BN_exp = fit_model(edges_exp, df_base, ESS)  # expected on baseline
    BN_obs = fit_model(edges_obs, df_pert, ESS)  # observed on perturbed

    # 4) KL via Monte Carlo (model-based; independent of raw data overlap)
    kl_obs_exp = KL_mc(BN_obs, BN_exp, N_SAMPLES, RNG_SEED)
    kl_exp_obs = KL_mc(BN_exp, BN_obs, N_SAMPLES, RNG_SEED)

    print(f"[summary] nodes_exp={len(BN_exp.nodes())} edges_exp={len(BN_exp.edges())} | "
          f"nodes_obs={len(BN_obs.nodes())} edges_obs={len(BN_obs.edges())}")
    print("\n=== KL results (nats) ===")
    print(f"KL(BN_obs || BN_exp): {kl_obs_exp:.6f}")
    print(f"KL(BN_exp || BN_obs): {kl_exp_obs:.6f}")
    print(f"Symmetric summary    : {(kl_obs_exp + kl_exp_obs)/2:.6f}")

if __name__ == "__main__":
    main()














# Minimal KL with pgmpy if both models share vars & support log_probability
#from pgmpy.sampling import BayesianModelSampling
#import numpy as np

#def KL_PQ(P, Q, N=20000):
#    # 1) sample x ~ P
#    x = BayesianModelSampling(P).forward_sample(size=N)
#    # 2) log probs under both models
#    lp = P.log_probability(x).to_numpy()
#    lq = Q.log_probability(x).to_numpy()
#    # 3) Monte-Carlo estimate of E_P[log P - log Q]
#    return float(np.mean(lp - lq))

#kl_obs_exp = KL_PQ(BN_obs, BN_exp)
#kl_exp_obs = KL_PQ(BN_exp, BN_obs)
