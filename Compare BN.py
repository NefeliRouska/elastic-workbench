import pandas as pd
from pathlib import Path
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

# ─────────────────────────────────────────────
# 1. Load and compare edges
# ─────────────────────────────────────────────
def load_edges(csv_path):
    df = pd.read_csv(csv_path)
    # assume first two columns are source, target if no standard names
    if {"source","target"}.issubset(df.columns):
        u, v = df["source"], df["target"]
    else:
        u, v = df.iloc[:,0], df.iloc[:,1]
    # keep as tuples for set ops; drop NaNs just in case
    edges = {(str(a), str(b)) for a, b in zip(u, v) if pd.notna(a) and pd.notna(b)}
    return edges

A = load_edges("share/metrics/bn_edges_nominal.csv")
B = load_edges("share/metrics/bn_edges_perturbated.csv")

only_in_A = A - B
only_in_B = B - A
in_both   = A & B

print(f"Edges in bn_edges.csv: {len(A)}")
print(f"Edges in bn_edges_perturbation.csv: {len(B)}")
print(f"Common edges: {len(in_both)}")
print(f"Only in bn_edges.csv: {len(only_in_A)}")
print(f"Only in bn_edges_perturbation.csv: {len(only_in_B)}")

jaccard = len(in_both) / len(A | B) if (A or B) else 1.0
print(f"Jaccard similarity (edge sets): {jaccard:.3f}")

# If you want to see which edges differ:
#if only_in_A:
#    print("\nEdges only in bn_edges.csv:")
#    for e in sorted(only_in_A):
#        print("  ", e)
#if only_in_B:
#    print("\nEdges only in bn_edges_perturbation.csv:")
#    for e in sorted(only_in_B):
#        print("  ", e)

'''# ─────────────────────────────────────────────
# 2. Visualize differences between the two BNs
# ─────────────────────────────────────────────
G_all = nx.DiGraph()
G_all.add_edges_from(A | B)

pos = nx.spring_layout(G_all, k=0.6, seed=7)

plt.figure(figsize=(14, 11))
nx.draw_networkx_nodes(G_all, pos, node_size=1200, node_color="#f7faff", edgecolors="#444")

# Common edges (gray)
nx.draw_networkx_edges(
    G_all, pos,
    edgelist=list(in_both),
    arrows=True, arrowstyle='-|>', arrowsize=24,
    width=1.2, edge_color="#999",
    min_source_margin=10, min_target_margin=12
)

# Only in bn_edges.csv (blue)
nx.draw_networkx_edges(
    G_all, pos,
    edgelist=list(only_in_A),
    arrows=True, arrowstyle='-|>', arrowsize=24,
    width=1.8, edge_color="#2b6cb0",
    min_source_margin=10, min_target_margin=12
)

# Only in bn_edges_perturbation.csv (red)
nx.draw_networkx_edges(
    G_all, pos,
    edgelist=list(only_in_B),
    arrows=True, arrowstyle='-|>', arrowsize=24,
    width=1.8, edge_color="#c53030",
    min_source_margin=10, min_target_margin=12
)

nx.draw_networkx_labels(G_all, pos, font_size=8)

legend_handles = [
    mlines.Line2D([], [], color="#999", linewidth=2, label="Common edges"),
    mlines.Line2D([], [], color="#2b6cb0", linewidth=2, label="Only in bn_edges.csv"),
    mlines.Line2D([], [], color="#c53030", linewidth=2, label="Only in bn_edges_perturbation.csv"),
]
plt.legend(handles=legend_handles, loc="upper left")

plt.title("BN Edge Comparison: baseline vs perturbation")
plt.axis("off")
plt.tight_layout()
plt.show()

'''
