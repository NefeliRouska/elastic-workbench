import pandas as pd, os
from pathlib import Path

# Load edges
def load_edges(p):
    df = pd.read_csv(p)
    u, v = (df["source"], df["target"]) if {"source","target"}.issubset(df.columns) else (df.iloc[:,0], df.iloc[:,1])
    return set(map(str, u)) | set(map(str, v))

nodes_exp = load_edges("bn_edges.csv")
nodes_obs = load_edges("bn_edges_perturbation.csv")
nodes_all = nodes_exp | nodes_obs

# Find the dataset the autodetect likely picked (widest CSV with many rows)
cands = []
for f in os.listdir("."):
    if f.lower().endswith(".csv") and f not in ["bn_edges.csv","bn_edges_perturbation.csv"]:
        try:
            df = pd.read_csv(f, nrows=200)
            cands.append((f, df.shape[0], df.shape[1]))
        except Exception:
            pass
cands.sort(key=lambda x: (x[2], x[1]), reverse=True)
dataset = cands[0][0] if cands else None
print("[dataset]", dataset)

# Peek columns and overlap with BN nodes
df_full = pd.read_csv(dataset) if dataset else None
cols = set(map(str, df_full.columns)) if df_full is not None else set()
print("Dataset columns (sample):", list(cols)[:10])
print("BN nodes total:", len(nodes_all))
print("Dataset columns total:", len(cols))
print("Overlap count:", len(nodes_all & cols))

missing_for_data = sorted(nodes_all - cols)
if missing_for_data:
    print("BN nodes missing in dataset (first 15):", missing_for_data[:15])
