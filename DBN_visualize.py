"""
DBN Sweep Results Visualizer
=============================
Generates all figures needed for the paper from a completed sweep.

Fixes applied:
  - Fig 1: throughput columns kept as features (configurable)
  - Fig 2: persistence taken from best config row, not first row
  - Fig 3: n_bins cast to int before groupby; dbscan correctly excluded
  - Fig 4: edge file pattern matches actual saved filenames
  - Fig 5 (new): MAE distribution by discretisation method and K
  - Table 1: handles None n_bins for dbscan correctly
  - Works with single target (throughput_3) or all three

Usage:
    python dbn_visualize.py \
        --csv dbn_k_sweep_results_<stamp>.csv \
        --edges_dir saved_dbn_models_top5_<stamp> \
        --out figures/

    Optional flags:
        --min_edge_frac 0.5   min fraction of models edge must appear in
        --show_tput_features  include throughput cols in feature importance plot

Outputs (all PDF, LNCS-ready):
    fig1_feature_importance.pdf
    fig2_accuracy_baselines.pdf
    fig3_hyperparam_sensitivity.pdf
    fig4_consensus_graph_<tgt>.pdf   (one per target)
    fig5_mae_distribution.pdf
    table1_results.tex
"""

import argparse
import os
import glob
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import networkx as nx

matplotlib.use("Agg")

plt.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size":          9,
    "axes.titlesize":     9,
    "axes.labelsize":     9,
    "xtick.labelsize":    8,
    "ytick.labelsize":    8,
    "legend.fontsize":    8,
    "figure.dpi":         300,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.05,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.linewidth":     0.6,
    "xtick.major.width":  0.6,
    "ytick.major.width":  0.6,
    "pdf.fonttype":       42,
    "ps.fonttype":        42,
})

COL1 = 8.5  / 2.54
COL2 = 17.0 / 2.54

C = {
    "blue":   "#1F77B4",
    "gray":   "#7F7F7F",
    "red":    "#D62728",
    "green":  "#2CA02C",
    "orange": "#FF7F0E",
}

# Distinct solid colors per service — no hatches in Fig 1
SVC_COLORS = {
    "1": "#2196F3",   # vivid blue  — Service 1
    "2": "#4CAF50",   # vivid green — Service 2
    "3": "#FF5722",   # vivid red-orange — Service 3
    "other": "#9E9E9E",
}

TARGET_LABELS = {
    "throughput_1": "Service 1",
    "throughput_2": "Service 2",
    "throughput_3": "Service 3",
}

FEATURE_LABELS = {
    "avg_p_latency_1":                     "latency (S1)",
    "avg_p_latency_2":                     "latency (S2)",
    "avg_p_latency_3":                     "latency (S3)",
    "cores_1":                             "cores (S1)",
    "cores_2":                             "cores (S2)",
    "cores_3":                             "cores (S3)",
    "data_quality_1":                      "data quality (S1)",
    "data_quality_2":                      "data quality (S2)",
    "data_quality_3":                      "data quality (S3)",
    "buffer_size_1":                       "buffer size (S1)",
    "buffer_size_2":                       "buffer size (S2)",
    "buffer_size_3":                       "buffer size (S3)",
    "throughput_1":                        "throughput (S1)",
    "throughput_2":                        "throughput (S2)",
    "python_gc_objects_collected_total_1": "GC objects (S1)",
    "python_gc_objects_collected_total_2": "GC objects (S2)",
    "python_gc_objects_collected_total_3": "GC objects (S3)",
    "python_gc_collections_total_1":       "GC coll. (S1)",
    "python_gc_collections_total_2":       "GC coll. (S2)",
    "python_gc_collections_total_3":       "GC coll. (S3)",
    "process_cpu_seconds_total_1":         "CPU time (S1)",
    "process_cpu_seconds_total_2":         "CPU time (S2)",
    "process_virtual_memory_bytes_1":      "virtual mem (S1)",
    "process_resident_memory_bytes_1":     "resident mem (S1)",
    "scrape_duration_seconds_1":           "scrape dur. (S1)",
}


def svc_color(feat):
    """Return a solid color for a feature based on its service suffix. No hatches."""
    svc_palette = {
        "1": "#2196F3",   # vivid blue  — Service 1
        "2": "#43A047",   # vivid green — Service 2
        "3": "#E53935",   # vivid red   — Service 3
    }
    for s, color in svc_palette.items():
        if feat.endswith(f"_{s}"):
            return color
    return "#9E9E9E"  # neutral gray for unknown


def short(feat):
    return FEATURE_LABELS.get(feat, feat)


# ─────────────────────────────────────────────────────────────────────────────
# FIG 1  Feature importance
# ─────────────────────────────────────────────────────────────────────────────
def fig_feature_importance(ok, out_dir, top_n=10, show_tput_features=False):
    """
    Shows which features appear most often in the top 10% configurations.

    show_tput_features: if True, include throughput_1/throughput_2 as
    features in the plot (relevant when EXCLUDE_OTHER_THROUGHPUTS=False).
    If False they are excluded to focus on non-target predictors.
    """
    targets = [t for t in ["throughput_1", "throughput_2", "throughput_3"]
               if t in ok["target"].unique()]
    n = len(targets)

    fig, axes = plt.subplots(1, n, figsize=(COL2, 3.4))
    if n == 1:
        axes = [axes]

    for ax, tgt in zip(axes, targets):
        sub    = ok[ok["target"] == tgt].copy()
        thresh = sub["accuracy"].quantile(0.9)
        top    = sub[sub["accuracy"] >= thresh]
        n_cfg  = len(top)

        ctr = Counter()
        for row in top["features"].dropna():
            for f in row.split(","):
                f = f.strip()
                if f == tgt:
                    continue
                # Optionally exclude throughput upstream features
                if not show_tput_features and f.startswith("throughput_"):
                    continue
                ctr[f] += 1

        if not ctr:
            ax.text(0.5, 0.5, "No features\nto display",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_title(TARGET_LABELS.get(tgt, tgt), fontweight="bold")
            continue

        items  = ctr.most_common(top_n)[::-1]
        feats  = [short(f) for f, _ in items]
        pcts   = [c / n_cfg for _, c in items]
        colors = [svc_color(f) for f, _ in items]

        ax.barh(feats, pcts, color=colors, edgecolor="white",
                linewidth=0.0, height=0.72)

        ax.set_xlim(0, 1.08)
        ax.set_xlabel("Frequency in top 10% configs")
        ax.set_title(TARGET_LABELS.get(tgt, tgt), fontweight="bold")
        ax.xaxis.set_major_formatter(
            matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        ax.tick_params(axis="y", labelsize=7.5)
        ax.axvline(0.5, color="#cccccc", lw=0.5, ls="--", zorder=0)

    handles = [
        mpatches.Patch(facecolor="#2196F3", label="Service 1",
                       edgecolor="none"),
        mpatches.Patch(facecolor="#43A047", label="Service 2",
                       edgecolor="none"),
        mpatches.Patch(facecolor="#E53935", label="Service 3",
                       edgecolor="none"),
        mpatches.Patch(facecolor="#9E9E9E", label="Other",
                       edgecolor="none"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3,
               frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=[0, 0.07, 1, 1])
    path = os.path.join(out_dir, "fig1_feature_importance.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"  saved {path}")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 2  Accuracy vs baselines
# ─────────────────────────────────────────────────────────────────────────────
def fig_accuracy_baselines(ok, out_dir):
    """
    Fix: persistence accuracy is now taken from the best configuration
    row rather than the first row, since it varies by n_bins.
    """
    targets = [t for t in ["throughput_1", "throughput_2", "throughput_3"]
               if t in ok["target"].unique()]

    rows = []
    for tgt in targets:
        sub      = ok[ok["target"] == tgt]
        best_row = sub.nlargest(1, "accuracy").iloc[0]
        rows.append({
            "label":       TARGET_LABELS.get(tgt, tgt),
            "best_dbn":    best_row["accuracy"],
            "persistence": best_row["persistence_accuracy"],
            "naive_best":  sub["naive_dbn_accuracy"].max(),
        })
    df = pd.DataFrame(rows)

    x = np.arange(len(df))
    w = 0.23
    fig, ax = plt.subplots(figsize=(COL1 * 1.4, 2.5))

    b1 = ax.bar(x - w, df["best_dbn"],    w, label="Learned DBN (best)",
                color="#1F77B4", edgecolor="white", lw=0.0)
    b2 = ax.bar(x,     df["persistence"], w, label="Persistence baseline",
                color="#7F7F7F", edgecolor="white", lw=0.0)
    b3 = ax.bar(x + w, df["naive_best"],  w, label="Naive DBN (best)",
                color="#D62728", edgecolor="white", lw=0.0)

    for bars in [b1, b2, b3]:
        for bar in bars:
            h = bar.get_height()
            if not np.isnan(h):
                ax.text(bar.get_x() + bar.get_width() / 2, h + 0.004,
                        f"{h:.2f}", ha="center", va="bottom", fontsize=6.5)

    ax.set_xticks(x)
    ax.set_xticklabels(df["label"])
    ax.set_ylabel("Accuracy")
    all_vals = df[["best_dbn", "persistence", "naive_best"]].values.flatten()
    valid    = all_vals[~np.isnan(all_vals)]
    ymin     = max(0.0, valid.min() - 0.12) if len(valid) > 0 else 0.0
    ax.set_ylim(ymin, 1.07)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))
    fig.legend(frameon=False, fontsize=7.5, loc="lower center",
               ncol=3, bbox_to_anchor=(0.5, -0.05))

    fig.tight_layout(rect=[0, 0.1, 1, 1])
    path = os.path.join(out_dir, "fig2_accuracy_baselines.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"  saved {path}")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 3  Hyperparameter sensitivity
# ─────────────────────────────────────────────────────────────────────────────
def fig_hyperparam_sensitivity(ok, out_dir):
    """
    Fix: n_bins cast to int before groupby.
    Fix: dbscan excluded from n_bins plot (its n_bins is None).
    """
    fig, axes = plt.subplots(1, 3, figsize=(COL2, 2.3))

    # (a) n_bins — exclude dbscan and cast to int
    ax = axes[0]
    nd = ok[ok["disc"] != "dbscan"].copy()
    nd = nd[nd["n_bins"].notna()].copy()
    nd["n_bins"] = nd["n_bins"].astype(float).astype(int)
    grp = nd.groupby("n_bins")["accuracy"].agg(["mean", "max"]).reset_index()

    if len(grp) > 0:
        ax.plot(grp["n_bins"], grp["mean"], "o-",  color=C["blue"], lw=1.2,
                ms=4, label="mean")
        ax.plot(grp["n_bins"], grp["max"],  "s--", color=C["gray"], lw=1.0,
                ms=3, label="max")
        ax.set_xticks(sorted(grp["n_bins"].unique()))
    ax.set_xlabel("$n_{\\mathrm{bins}}$")
    ax.set_ylabel("Accuracy")
    ax.set_title("(a) Discretisation bins", fontsize=8)
    ax.legend(frameon=False, fontsize=7)
    ax.set_ylim(max(0.0, ok["accuracy"].min() - 0.1), 1.02)

    # (b) K
    ax  = axes[1]
    grp = ok.groupby("K")["accuracy"].agg(["mean", "max"]).reset_index()
    if len(grp) > 0:
        ax.plot(grp["K"], grp["mean"], "o-",  color=C["blue"], lw=1.2,
                ms=4, label="mean")
        ax.plot(grp["K"], grp["max"],  "s--", color=C["gray"], lw=1.0,
                ms=3, label="max")
        ax.set_xticks(sorted(grp["K"].unique()))
    ax.set_xlabel("$K$ (features)")
    ax.set_title("(b) Feature count $K$", fontsize=8)
    ax.set_ylim(max(0.0, ok["accuracy"].min() - 0.1), 1.02)

    # (c) method comparison — disc and fs side by side
    ax        = axes[2]
    disc_mean = ok.groupby("disc")["accuracy"].mean().sort_values(ascending=False)
    fs_mean   = ok.groupby("fs")["accuracy"].mean().sort_values(ascending=False)
    all_lbls  = list(disc_mean.index) + list(fs_mean.index)
    all_vals  = list(disc_mean.values) + list(fs_mean.values)
    all_cols = [C["blue"]] * len(disc_mean) + [C["green"]] * len(fs_mean)

    ax.bar(range(len(all_vals)), all_vals, color=all_cols,
           edgecolor="white", lw=0.0, width=0.7)

    ax.set_xticks(range(len(all_lbls)))
    ax.set_xticklabels(all_lbls, rotation=38, ha="right", fontsize=7)
    ax.set_title("(c) Method comparison", fontsize=8)
    valid_vals = [v for v in all_vals if not np.isnan(v)]
    if valid_vals:
        ax.set_ylim(max(0, min(valid_vals) - 0.06), max(valid_vals) + 0.09)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))
    ax.axvline(len(disc_mean) - 0.5, color="#cccccc", lw=0.8, ls=":")

    handles = [
        mpatches.Patch(facecolor=C["blue"],  label="Disc. method",
                       edgecolor="none"),
        mpatches.Patch(facecolor=C["green"], label="Feature sel.",
                       edgecolor="none"),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=7)

    fig.tight_layout()
    path = os.path.join(out_dir, "fig3_hyperparam_sensitivity.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"  saved {path}")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 4  Consensus DBN graph
# ─────────────────────────────────────────────────────────────────────────────
def _load_edges(edges_dir, target):
    """
    Fix: pattern now matches actual saved filenames:
    rank1_dbn_throughput_3_K8_markov_kmeans_bic_bins4_edges.csv
    """
    pattern = os.path.join(edges_dir, f"*dbn_{target}_*_edges.csv")
    files   = glob.glob(pattern)
    if not files:
        print(f"  [warn] no edge files found matching: {pattern}")
        return Counter(), 0
    ctr = Counter()
    for f in files:
        try:
            df = pd.read_csv(f)
            if "parent" not in df.columns or "child" not in df.columns:
                continue
            for _, row in df.iterrows():
                ctr[(row["parent"], row["child"])] += 1
        except Exception as e:
            print(f"  [warn] could not read {f}: {e}")
    return ctr, len(files)


def _node_label(node):
    for suffix, ts in [("_t1", "(t+1)"), ("_t", "(t)")]:
        if node.endswith(suffix):
            base = (node[:-len(suffix)]
                    .replace("throughput_",                        "tput")
                    .replace("buffer_size_",                       "buf")
                    .replace("avg_p_latency_",                     "lat")
                    .replace("data_quality_",                      "dq")
                    .replace("cores_",                             "cores")
                    .replace("python_gc_objects_collected_total_", "gc_obj")
                    .replace("python_gc_collections_total_",       "gc_col")
                    .replace("process_cpu_seconds_total_",         "cpu")
                    .replace("process_virtual_memory_bytes_",      "vmem")
                    .replace("process_resident_memory_bytes_",     "rmem")
                    .replace("scrape_duration_seconds_",           "scrape"))
            return f"{base}\n{ts}"
    return node


def _node_color(node):
    if "throughput" in node: return "#C9DCF0"
    if "buffer"     in node: return "#C9EDD9"
    if "latency"    in node: return "#FDDCB0"
    if "data_qual"  in node: return "#E8DAEF"
    if "cores"      in node: return "#FAD7A0"
    return "#EBEBEB"


def fig_consensus_graph(edges_dir, target, out_dir, min_frac=0.5):
    """
    Builds a consensus graph from all saved edge files for a target.
    Only shows edges that appear in at least min_frac of saved models.
    This provides a structural stability view — edges that are robust
    across the top-N configurations.
    """
    ctr, n_files = _load_edges(edges_dir, target)
    if not ctr:
        print(f"  [skip] no edge files for {target}")
        return

    min_count = max(1, round(n_files * min_frac))
    edges_use = [(p, c, cnt) for (p, c), cnt in ctr.items()
                 if cnt >= min_count]

    if not edges_use:
        print(f"  [skip] no edges above threshold ({min_count}/{n_files}) "
              f"for {target}")
        return

    G = nx.DiGraph()
    for p, c, cnt in edges_use:
        G.add_edge(p, c, weight=cnt)

    priority = ["throughput_1", "throughput_2", "throughput_3",
                "buffer_size_3", "buffer_size_2", "buffer_size_1",
                "avg_p_latency", "data_quality", "cores"]

    def rank(n):
        base = n.replace("_t1", "").replace("_t", "")
        for i, p in enumerate(priority):
            if p in base:
                return i
        return 99

    t_nodes  = sorted([n for n in G.nodes() if n.endswith("_t")],  key=rank)
    t1_nodes = sorted([n for n in G.nodes() if n.endswith("_t1")], key=rank)

    y_step = 1.3
    pos = {}
    for i, n in enumerate(t_nodes):
        pos[n] = (0, -i * y_step)
    assigned_ys = set(y for _, y in pos.values())
    extra_y = -(len(t_nodes)) * y_step

    for n in t1_nodes:
        base = n.replace("_t1", "_t")
        if base in pos:
            pos[n] = (4, pos[base][1])
        else:
            while extra_y in assigned_ys:
                extra_y -= y_step
            pos[n] = (4, extra_y)
            assigned_ys.add(extra_y)
            extra_y -= y_step

    def edge_color(cnt):
        frac = cnt / n_files
        if frac >= 1.0: return "#1A5C35"
        if frac >= 0.7: return "#1A4E8A"
        return "#888888"

    def edge_width(cnt):
        return 0.8 + 2.8 * (cnt / n_files)

    edge_list   = [(p, c) for p, c, _ in edges_use]
    edge_colors = [edge_color(cnt) for _, _, cnt in edges_use]
    edge_widths = [edge_width(cnt) for _, _, cnt in edges_use]
    node_colors = [_node_color(n) for n in G.nodes()]
    labels      = {n: _node_label(n) for n in G.nodes()}

    h = max(3.5, y_step * max(len(t_nodes), len(t1_nodes)) * 0.85 + 2.0)
    fig, ax = plt.subplots(figsize=(COL2, h))

    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                           node_size=1700, node_shape="s",
                           edgecolors="#666", linewidths=0.7)
    nx.draw_networkx_labels(G, pos, labels=labels, ax=ax,
                            font_size=7, font_family="serif")
    nx.draw_networkx_edges(G, pos, ax=ax, edgelist=edge_list,
                           width=edge_widths, edge_color=edge_colors,
                           arrows=True, arrowsize=16,
                           connectionstyle="arc3,rad=0.12",
                           min_source_margin=28, min_target_margin=28)

    ys   = [y for _, y in pos.values()]
    y_lo = min(ys) - 0.9
    ax.axvline(2.0, color="#CCCCCC", lw=0.8, ls="--", zorder=0)
    ax.text(0, y_lo, "time $t$",   ha="center", fontsize=8, style="italic",
            color="#555")
    ax.text(4, y_lo, "time $t+1$", ha="center", fontsize=8, style="italic",
            color="#555")

    univ   = n_files
    strong = round(n_files * 0.7)
    edge_handles = [
        mlines.Line2D([], [], color="#1A5C35", lw=2.5,
                      label=f"Universal ({univ}/{n_files})"),
        mlines.Line2D([], [], color="#1A4E8A", lw=2.0,
                      label=f"Strong ($\\geq${strong}/{n_files})"),
        mlines.Line2D([], [], color="#888888", lw=1.2,
                      label=f"Majority ($\\geq${min_count}/{n_files})"),
    ]
    node_handles = [
        mpatches.Patch(fc="#C9DCF0", ec="#666", lw=0.6, label="throughput"),
        mpatches.Patch(fc="#C9EDD9", ec="#666", lw=0.6, label="buffer size"),
        mpatches.Patch(fc="#FDDCB0", ec="#666", lw=0.6, label="latency"),
        mpatches.Patch(fc="#E8DAEF", ec="#666", lw=0.6, label="data quality"),
        mpatches.Patch(fc="#FAD7A0", ec="#666", lw=0.6, label="cores"),
    ]
    ax.legend(handles=edge_handles + node_handles,
              loc="upper center", bbox_to_anchor=(0.5, 1.12),
              ncol=4, frameon=False, fontsize=7)
    ax.set_xlim(-1.2, 5.2)
    ax.axis("off")
    fig.tight_layout()

    path = os.path.join(out_dir, f"fig4_consensus_graph_{target}.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"  saved {path}")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 5  MAE distribution  (new — supports paper contribution on real-unit eval)
# ─────────────────────────────────────────────────────────────────────────────
def fig_mae_distribution(ok, out_dir):
    """
    Shows MAE in real units broken down by discretisation method and K.
    This directly supports the paper contribution of evaluating in
    original physical units rather than abstract bin indices.
    """
    mae_col = "mae"
    if mae_col not in ok.columns or ok[mae_col].isna().all():
        print("  [skip] no MAE data available")
        return

    mae_ok = ok[ok[mae_col].notna() & (ok[mae_col] < 1e6)].copy()

    fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.8))

    # (a) MAE by discretisation method
    ax   = axes[0]
    disc_order = (mae_ok.groupby("disc")[mae_col].median()
                  .sort_values().index.tolist())
    data = [mae_ok[mae_ok["disc"] == d][mae_col].dropna().values
            for d in disc_order]
    bp   = ax.boxplot(data, patch_artist=True, notch=False,
                      medianprops=dict(color="black", lw=1.5),
                      whiskerprops=dict(lw=0.8),
                      capprops=dict(lw=0.8),
                      flierprops=dict(marker="o", ms=2, alpha=0.4))
    colors_disc = [C["blue"], C["green"], C["orange"], C["red"]]
    for patch, color in zip(bp["boxes"], colors_disc[:len(disc_order)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_xticks(range(1, len(disc_order) + 1))
    ax.set_xticklabels(disc_order, rotation=30, ha="right", fontsize=7.5)
    ax.set_ylabel("MAE (original units)")
    ax.set_title("(a) MAE by discretisation method", fontsize=8)

    # (b) MAE by K
    ax       = axes[1]
    k_order  = sorted(mae_ok["K"].unique())
    data_k   = [mae_ok[mae_ok["K"] == k][mae_col].dropna().values
                for k in k_order]
    bp2      = ax.boxplot(data_k, patch_artist=True, notch=False,
                          medianprops=dict(color="black", lw=1.5),
                          whiskerprops=dict(lw=0.8),
                          capprops=dict(lw=0.8),
                          flierprops=dict(marker="o", ms=2, alpha=0.4))
    for patch in bp2["boxes"]:
        patch.set_facecolor(C["blue"])
        patch.set_alpha(0.7)
    ax.set_xticks(range(1, len(k_order) + 1))
    ax.set_xticklabels([f"K={k}" for k in k_order], fontsize=8)
    ax.set_ylabel("MAE (original units)")
    ax.set_title("(b) MAE by feature count $K$", fontsize=8)

    fig.tight_layout()
    path = os.path.join(out_dir, "fig5_mae_distribution.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"  saved {path}")


# ─────────────────────────────────────────────────────────────────────────────
# TABLE 1
# ─────────────────────────────────────────────────────────────────────────────
def table_results(ok, out_dir):
    """
    Fix: n_bins handled correctly for dbscan (stored as None/NaN).
    Persistence accuracy now taken from the best config row.
    """
    targets = [t for t in ["throughput_1", "throughput_2", "throughput_3"]
               if t in ok["target"].unique()]
    lines = [
        r"\begin{table}[t]", r"\centering",
        r"\caption{Best DBN configuration per target vs.\ baselines."
        r" MAE in original throughput units.}",
        r"\label{tab:results}",
        r"\begin{tabular}{lcccccc}", r"\toprule",
        r"Target & Best acc & Persistence & Naive DBN & MAE & $K$ & Config \\",
        r"\midrule",
    ]
    for tgt in targets:
        sub      = ok[ok["target"] == tgt]
        best     = sub.nlargest(1, "accuracy").iloc[0]
        nb_val   = best.get("n_bins", None)
        nb       = int(float(nb_val)) if pd.notna(nb_val) else "-"
        cfg      = f"{best['disc']}, {best['fs']}, bins={nb}"
        pers     = best["persistence_accuracy"]
        naive    = sub["naive_dbn_accuracy"].max()
        lines.append(
            f"{TARGET_LABELS.get(tgt, tgt)} & "
            f"{best['accuracy']:.3f} & "
            f"{pers:.3f} & "
            f"{naive:.3f} & "
            f"{best['mae']:.1f} & "
            f"{int(best['K'])} & "
            r"\texttt{" + cfg + r"} \\"
        )
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    path = os.path.join(out_dir, "table1_results.tex")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  saved {path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",           required=True,
                        help="Path to sweep results CSV")
    parser.add_argument("--edges_dir",     required=True,
                        help="Directory containing saved model edge CSVs")
    parser.add_argument("--out",           default="figures",
                        help="Output directory for figures")
    parser.add_argument("--min_edge_frac", type=float, default=0.5,
                        help="Min fraction of models an edge must appear in "
                             "for consensus graph (default 0.5)")
    parser.add_argument("--show_tput_features", action="store_true",
                        help="Include throughput_1/2 in feature importance plot")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print(f"\nLoading {args.csv} ...")
    df = pd.read_csv(args.csv)
    ok = df[df["error"].isna()].copy()
    print(f"  {len(ok)} successful runs")
    print(f"  targets: {ok['target'].unique().tolist()}")
    print(f"  total failed: {df['error'].notna().sum()}")

    print("\nFig 1: feature importance ...")
    fig_feature_importance(ok, args.out,
                           show_tput_features=args.show_tput_features)

    print("Fig 2: accuracy vs baselines ...")
    fig_accuracy_baselines(ok, args.out)

    print("Fig 3: hyperparameter sensitivity ...")
    fig_hyperparam_sensitivity(ok, args.out)

    print("Fig 4: consensus DBN graphs ...")
    for tgt in ok["target"].unique():
        fig_consensus_graph(args.edges_dir, tgt,
                            out_dir=args.out,
                            min_frac=args.min_edge_frac)

    print("Fig 5: MAE distribution ...")
    fig_mae_distribution(ok, args.out)

    print("Table 1: LaTeX results ...")
    table_results(ok, args.out)

    print(f"\nAll outputs saved to: {args.out}/")
    print("\nFiles produced:")
    for f in sorted(os.listdir(args.out)):
        print(f"  {f}")


if __name__ == "__main__":
    main()