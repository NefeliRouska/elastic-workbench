"""
DBN Sweep Results Visualizer v2
================================
Six figures:
  Fig 1 — Main results: accuracy, F1, log loss vs baselines
  Fig 2 — 3D surface: K x n_bins x metric (accuracy, F1, log loss)
  Fig 3 — Top 10% configuration profile (what best models look like)
  Fig 4 — Method comparison: accuracy and log loss separate panels
  Fig 5 — Feature importance filtered to accuracy > 90% configs
  Fig 6 — Consensus DBN graph x3 (one per throughput target highlighted)

Note: precision/recall columns added in new sweep run.
      Script gracefully handles their absence for backward compatibility.

Usage:
    python dbn_visualize_v2.py \
        --csv dbn_k_sweep_results_<stamp>.csv \
        --edges_dir saved_dbn_models_top5_<stamp> \
        --out figures/

    Optional:
        --min_edge_frac 0.8
        --top_pct 0.10        threshold for top-N figures (default 0.10 = top 10%)
        --acc_thresh 0.90     accuracy threshold for feature importance (default 0.90)
        --show_tput_features  include throughput cols in feature importance
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
import matplotlib.ticker
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata
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
    "purple": "#9467BD",
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
    "scrape_duration_seconds_3":           "scrape dur. (S3)",
}

EXCLUDE_FS   = ["pca"]
EXCLUDE_DISC = ["dbscan"]


def svc_color(feat):
    for s, color in {"1": "#2196F3", "2": "#43A047", "3": "#E53935"}.items():
        if feat.endswith(f"_{s}"):
            return color
    return "#9E9E9E"


def short(feat):
    return FEATURE_LABELS.get(feat, feat)


def _filter_methods(df):
    return df[~df["fs"].isin(EXCLUDE_FS) & ~df["disc"].isin(EXCLUDE_DISC)].copy()


def _has_col(df, col):
    return col in df.columns and df[col].notna().any()


# ─────────────────────────────────────────────────────────────────────────────
# FIG 1  Main results vs baselines
# ─────────────────────────────────────────────────────────────────────────────
def fig_main_results(ok, out_dir):
    """
    Panels: accuracy, F1, log loss (and precision/recall if available).
    Best configuration selected by accuracy per target.
    Persistence shown as N/A in log loss panel.
    """
    targets = [t for t in ["throughput_1", "throughput_2", "throughput_3"]
               if t in ok["target"].unique()]

    has_pr = _has_col(ok, "precision") and _has_col(ok, "recall")

    rows = []
    for tgt in targets:
        sub  = ok[ok["target"] == tgt]
        best = sub.nlargest(1, "accuracy").iloc[0]
        r = {
            "label":    TARGET_LABELS.get(tgt, tgt),
            "dbn_acc":  best["accuracy"],
            "pers_acc": best["persistence_accuracy"],
            "ar_acc":   best["ar_dbn_accuracy"],
            "dbn_f1":   best["f1"],
            "pers_f1":  best["persistence_f1"],
            "ar_f1":    best["ar_dbn_f1"],
            "dbn_ll":   best["log_loss"],
            "ar_ll":    best["ar_dbn_log_loss"],
        }
        if has_pr:
            r.update({
                "dbn_prec":  best.get("precision", np.nan),
                "pers_prec": best.get("persistence_precision", np.nan),
                "ar_prec":   best.get("ar_dbn_precision", np.nan),
                "dbn_rec":   best.get("recall", np.nan),
                "pers_rec":  best.get("persistence_recall", np.nan),
                "ar_rec":    best.get("ar_dbn_recall", np.nan),
            })
        rows.append(r)
    df = pd.DataFrame(rows)

    n_panels = 5 if has_pr else 3
    fig, axes = plt.subplots(1, n_panels, figsize=(COL2 * (n_panels / 3), 2.8))

    x       = np.arange(len(df))
    w       = 0.22
    offsets = [-w, 0, w]
    colors  = [C["blue"], C["gray"], C["red"]]
    labels  = ["Learned DBN", "Persistence", "AR-DBN"]

    def _panel(ax, vals_list, ylabel, title, na_idx=None):
        for i, (vals, lbl, col, off) in enumerate(
                zip(vals_list, labels, colors, offsets)):
            if na_idx is not None and i == na_idx:
                ax.bar(x + off, [0.01] * len(df), w,
                       color="#dddddd", edgecolor="#aaa", hatch="///", lw=0.5)
                for xi in x:
                    ax.text(xi + off, 0.02, "N/A", ha="center", va="bottom",
                            fontsize=5.5, color="#888", rotation=90)
            else:
                b = ax.bar(x + off, vals, w, label=lbl,
                           color=col, edgecolor="white", lw=0.0)
                for bar, v in zip(b, vals):
                    if not np.isnan(v):
                        ax.text(bar.get_x() + bar.get_width() / 2,
                                bar.get_height() + 0.005,
                                f"{v:.2f}", ha="center", va="bottom", fontsize=5.5)
        ax.set_xticks(x)
        ax.set_xticklabels(df["label"], fontsize=8)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=8)
        all_v = [v for i2, lst in enumerate(vals_list)
                 for v in lst
                 if not (na_idx is not None and i2 == na_idx) and not np.isnan(v)]
        if all_v:
            ax.set_ylim(max(0.0, min(all_v) - 0.10), max(all_v) + 0.12)
        ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))

    # Panel order: Accuracy -> Precision -> Recall -> F1 -> Log Loss
    _panel(axes[0],
           [df["dbn_acc"].values, df["pers_acc"].values, df["ar_acc"].values],
           "Accuracy", "(a) Accuracy")

    if has_pr:
        _panel(axes[1],
               [df["dbn_prec"].values, df["pers_prec"].values, df["ar_prec"].values],
               "Macro Precision", "(b) Precision")
        _panel(axes[2],
               [df["dbn_rec"].values, df["pers_rec"].values, df["ar_rec"].values],
               "Macro Recall", "(c) Recall")
        _panel(axes[3],
               [df["dbn_f1"].values, df["pers_f1"].values, df["ar_f1"].values],
               "Macro F1", "(d) Macro F1")
        _panel(axes[4],
               [df["dbn_ll"].values, [np.nan] * len(df), df["ar_ll"].values],
               "Log Loss (lower=better)", "(e) Log Loss", na_idx=1)
    else:
        _panel(axes[1],
               [df["dbn_f1"].values, df["pers_f1"].values, df["ar_f1"].values],
               "Macro F1", "(b) Macro F1")
        _panel(axes[2],
               [df["dbn_ll"].values, [np.nan] * len(df), df["ar_ll"].values],
               "Log Loss (lower=better)", "(c) Log Loss", na_idx=1)

    handles = [
        mpatches.Patch(facecolor=C["blue"], label="Learned DBN", edgecolor="none"),
        mpatches.Patch(facecolor=C["gray"], label="Persistence",  edgecolor="none"),
        mpatches.Patch(facecolor=C["red"],  label="AR-DBN",       edgecolor="none"),
        mpatches.Patch(facecolor="#dddddd", label="N/A (no probs)",
                       edgecolor="#aaa", hatch="///"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4,
               frameon=False, bbox_to_anchor=(0.5, -0.04), fontsize=7.5)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    path = os.path.join(out_dir, "fig1_main_results.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"  saved {path}")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 2  3D surface: K x n_bins x metric
# ─────────────────────────────────────────────────────────────────────────────
def fig_3d_surface(ok, out_dir):
    """
    Five 3D surface plots: accuracy, precision, recall, F1, log loss.
    Two rows: (a)(b)(c) top row, (d)(e) bottom row.
    X = K, Y = n_bins, Z = metric mean.
    n= counts shown on axis tick labels.
    """
    nd = ok[ok["disc"] != "dbscan"].copy()
    nd = nd[nd["n_bins"].notna()].copy()
    nd["n_bins"] = nd["n_bins"].astype(float).astype(int)

    agg_dict = {
        "acc_mean":  ("accuracy",  "mean"),
        "f1_mean":   ("f1",        "mean"),
        "ll_mean":   ("log_loss",  "mean"),
        "n":         ("accuracy",  "count"),
    }
    if _has_col(nd, "precision"):
        agg_dict["prec_mean"] = ("precision", "mean")
        agg_dict["rec_mean"]  = ("recall",    "mean")

    grp = nd.groupby(["K", "n_bins"]).agg(**agg_dict).reset_index()

    K_vals    = sorted(grp["K"].unique())
    bins_vals = sorted(grp["n_bins"].unique())

    # n counts for axis labels
    k_n    = {k: int(nd[nd["K"] == k]["accuracy"].count()) for k in K_vals}
    bins_n = {b: int(nd[nd["n_bins"] == b]["accuracy"].count()) for b in bins_vals}

    K_grid, bins_grid = np.meshgrid(K_vals, bins_vals)

    def _surface_z(metric_col):
        z = np.full(K_grid.shape, np.nan)
        for i, bv in enumerate(bins_vals):
            for j, kv in enumerate(K_vals):
                row = grp[(grp["K"] == kv) & (grp["n_bins"] == bv)]
                if len(row) > 0 and metric_col in grp.columns:
                    z[i, j] = row[metric_col].values[0]
        return z

    has_pr = "prec_mean" in grp.columns
    if has_pr:
        metrics = [
            ("acc_mean",  "Accuracy",        "Blues",   False, "(a)"),
            ("prec_mean", "Precision",        "Purples", False, "(b)"),
            ("rec_mean",  "Recall",           "Oranges", False, "(c)"),
            ("f1_mean",   "Macro F1",        "Greens",  False, "(d)"),
            ("ll_mean",   "Log Loss",        "Reds_r",  True,  "(e)"),
        ]
        fig = plt.figure(figsize=(COL2 * 1.1, 7.0))
        positions = [
            (2, 3, 1), (2, 3, 2), (2, 3, 3),
            (2, 3, 4), (2, 3, 5),
        ]
    else:
        metrics = [
            ("acc_mean",  "Accuracy",  "Blues",  False, "(a)"),
            ("f1_mean",   "Macro F1", "Greens", False, "(b)"),
            ("ll_mean",   "Log Loss", "Reds_r", True,  "(c)"),
        ]
        fig = plt.figure(figsize=(COL2, 3.8))
        positions = [(1, 3, 1), (1, 3, 2), (1, 3, 3)]

    for (row_n, col_n, pos_n), (col, label, cmap, lower_better, letter) in zip(positions, metrics):
        ax = fig.add_subplot(row_n, col_n, pos_n, projection="3d")
        z  = _surface_z(col)

        points = np.array([[K_vals[j], bins_vals[i]]
                           for i in range(len(bins_vals))
                           for j in range(len(K_vals))])
        values = z.flatten()
        valid  = ~np.isnan(values)

        K_fine    = np.linspace(min(K_vals), max(K_vals), 30)
        bins_fine = np.linspace(min(bins_vals), max(bins_vals), 30)
        Kf, Bf    = np.meshgrid(K_fine, bins_fine)

        if valid.sum() >= 3:
            Zf = griddata(points[valid], values[valid], (Kf, Bf), method="cubic")
        else:
            Zf = np.full(Kf.shape, np.nan)

        surf = ax.plot_surface(Kf, Bf, Zf, cmap=cmap,
                               alpha=0.85, linewidth=0, antialiased=True)

        for i in range(len(bins_vals)):
            for j in range(len(K_vals)):
                if not np.isnan(z[i, j]):
                    ax.scatter(K_vals[j], bins_vals[i], z[i, j],
                               color="black", s=12, zorder=5)

        ax.set_xlabel("K", fontsize=6, labelpad=3)
        ax.set_ylabel("$n_{{bins}}$", fontsize=6, labelpad=3)
        ax.set_zlabel(label, fontsize=6, labelpad=3)
        ax.set_title(f"{letter} {label}", fontsize=8)
        ax.set_xticks(K_vals)
        ax.set_xticklabels([f"K={k}\n(n={k_n[k]})" for k in K_vals], fontsize=5)
        ax.set_yticks(bins_vals)
        ax.set_yticklabels([f"{b}\n(n={bins_n[b]})" for b in bins_vals], fontsize=5)
        ax.tick_params(labelsize=5)

        if lower_better:
            ax.text2D(0.05, 0.95, "lower=better",
                      transform=ax.transAxes, fontsize=6, color="#666")

        fig.colorbar(surf, ax=ax, shrink=0.45, aspect=10, pad=0.12)

    fig.suptitle("Metric surface over hyperparameter grid",
                 fontsize=9, y=1.01)
    fig.tight_layout()
    path = os.path.join(out_dir, "fig2_3d_surface.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"  saved {path}")



# ─────────────────────────────────────────────────────────────────────────────
# FIG 3  Full sweep configuration profile
# ─────────────────────────────────────────────────────────────────────────────
def fig_full_profile(ok, out_dir):
    """
    Distribution of K, n_bins, disc method, fs method across ALL configurations.
    pca and dbscan excluded. Shows the complete picture before filtering.
    """
    ok_m = _filter_methods(ok)
    ok_m = ok_m[ok_m["n_bins"].notna()].copy()
    ok_m["n_bins"] = ok_m["n_bins"].astype(float).astype(int)
    n_all = len(ok_m)

    fig, axes = plt.subplots(1, 4, figsize=(COL2, 2.6))

    def _bar(ax, series, title, color):
        counts = series.value_counts().sort_index()
        pcts   = counts / n_all
        ax.bar(range(len(counts)), pcts.values, color=color,
               edgecolor="white", lw=0.0, width=0.6)
        ax.set_xticks(range(len(counts)))
        ax.set_xticklabels([str(x) for x in counts.index],
                           rotation=30, ha="right", fontsize=7)
        ax.set_ylabel("Fraction of all configs")
        ax.set_title(title, fontsize=8)
        ax.set_ylim(0, pcts.max() + 0.15)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        for xi, (val, pct) in enumerate(zip(counts.values, pcts.values)):
            ax.text(xi, pct + 0.01, f"{val}", ha="center", va="bottom", fontsize=6.5)

    _bar(axes[0], ok_m["n_bins"], "(a) $n_{{bins}}$", C["blue"])
    _bar(axes[1], ok_m["K"],      "(b) $K$",                  C["orange"])
    _bar(axes[2], ok_m["disc"],   "(c) Disc. method",         C["green"])
    _bar(axes[3], ok_m["fs"],     "(d) Feature selection",    C["purple"])

    fig.suptitle(f"Full sweep configuration profile (n={n_all})",
                 fontsize=8, y=1.02)
    fig.tight_layout()
    path = os.path.join(out_dir, "fig3_full_profile.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"  saved {path}")

# ─────────────────────────────────────────────────────────────────────────────
# FIG 3  Top 10% configuration profile
# ─────────────────────────────────────────────────────────────────────────────
def fig_top_profile(ok, out_dir, top_pct=0.10):
    """
    For configs in the top top_pct by accuracy, show distribution of
    K, n_bins, disc method, fs method.
    pca and dbscan excluded.
    """
    ok_m   = _filter_methods(ok)
    thresh = ok_m["accuracy"].quantile(1 - top_pct)
    top    = ok_m[ok_m["accuracy"] >= thresh].copy()
    n_top  = len(top)
    pct_label = f"top {int(top_pct*100)}% (n={n_top}, acc≥{thresh:.3f})"

    # Cast n_bins
    top = top[top["n_bins"].notna()].copy()
    top["n_bins"] = top["n_bins"].astype(float).astype(int)

    fig, axes = plt.subplots(1, 4, figsize=(COL2, 2.6))

    # Get all-config counts for context annotation
    ok_m_all = _filter_methods(ok)
    ok_m_all = ok_m_all[ok_m_all["n_bins"].notna()].copy()
    ok_m_all["n_bins"] = ok_m_all["n_bins"].astype(float).astype(int)
    n_all_configs = len(ok_m_all)

    def _bar(ax, series, title, color, all_series=None):
        counts    = series.value_counts().sort_index()
        pcts      = counts / n_top
        ax.bar(range(len(counts)), pcts.values, color=color,
               edgecolor="white", lw=0.0, width=0.6)
        ax.set_xticks(range(len(counts)))
        ax.set_xticklabels([str(x) for x in counts.index],
                           rotation=30, ha="right", fontsize=7)
        ax.set_ylabel("Fraction of top configs")
        ax.set_title(title, fontsize=8)
        ax.set_ylim(0, pcts.max() + 0.22)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        for xi, (val, pct) in enumerate(zip(counts.values, pcts.values)):
            # Count in top configs
            ax.text(xi, pct + 0.01, f"{val}", ha="center", va="bottom", fontsize=6.5)
            # Context: % of all configs
            if all_series is not None:
                all_counts = all_series.value_counts()
                key = counts.index[xi]
                all_n   = all_counts.get(key, 0)
                all_pct = all_n / n_all_configs
                ax.text(xi, pct + 0.07,
                        f"({all_pct:.0%} overall)",
                        ha="center", va="bottom", fontsize=5.5,
                        color="#666", style="italic")

    _bar(axes[0], top["n_bins"], "(a) $n_{{bins}}$", C["blue"],   ok_m_all["n_bins"])
    _bar(axes[1], top["K"],      "(b) $K$",                  C["orange"], ok_m_all["K"])
    _bar(axes[2], top["disc"],   "(c) Disc. method",         C["green"],  ok_m_all["disc"])
    _bar(axes[3], top["fs"],     "(d) Feature selection",    C["purple"], ok_m_all["fs"])

    fig.suptitle(f"Configuration profile — {pct_label}", fontsize=8, y=1.02)
    fig.tight_layout()
    path = os.path.join(out_dir, "fig4_top_profile.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"  saved {path}")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 4  Method comparison — accuracy and log loss in separate panels
# ─────────────────────────────────────────────────────────────────────────────
def fig_method_comparison(ok, out_dir, top_pct=0.10):
    """
    Two panels — accuracy and log loss — for disc and fs methods.
    Computed within top top_pct configs. pca and dbscan excluded.
    """
    ok_m   = _filter_methods(ok)
    thresh = ok_m["accuracy"].quantile(1 - top_pct)
    top    = ok_m[ok_m["accuracy"] >= thresh].copy()

    fig, axes = plt.subplots(1, 2, figsize=(COL2, 2.6))

    def _method_panel(ax, metric, ylabel, title, lower_better=False):
        disc_vals = top.groupby("disc")[metric].mean().sort_values(ascending=lower_better)
        fs_vals   = top.groupby("fs")[metric].mean().sort_values(ascending=lower_better)

        all_lbls = list(disc_vals.index) + list(fs_vals.index)
        all_vals = list(disc_vals.values) + list(fs_vals.values)
        n_disc   = len(disc_vals)
        col_disc = [C["blue"]]  * n_disc
        col_fs   = [C["green"]] * len(fs_vals)
        all_cols = col_disc + col_fs

        xb = np.arange(len(all_lbls))
        bars = ax.bar(xb, all_vals, color=all_cols,
                      edgecolor="white", lw=0.0, width=0.6)
        for bar, v in zip(bars, all_vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + (0.005 if not lower_better else 0.005),
                    f"{v:.3f}", ha="center", va="bottom", fontsize=6)

        ax.set_xticks(xb)
        ax.set_xticklabels(all_lbls, rotation=38, ha="right", fontsize=7)
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=8)
        ax.axvline(n_disc - 0.5, color="#cccccc", lw=0.8, ls=":")
        if lower_better:
            ax.text(0.97, 0.97, "lower=better", transform=ax.transAxes,
                    ha="right", va="top", fontsize=6.5, color="#666", style="italic")

        valid = [v for v in all_vals if not np.isnan(v)]
        if valid:
            margin = (max(valid) - min(valid)) * 0.2 + 0.05
            ax.set_ylim(max(0, min(valid) - margin), max(valid) + margin)
        ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter("%.2f"))

        handles = [
            mpatches.Patch(facecolor=C["blue"],  label="Disc. method", edgecolor="none"),
            mpatches.Patch(facecolor=C["green"], label="Feature sel.",  edgecolor="none"),
        ]
        ax.legend(handles=handles, frameon=False, fontsize=7)

    _method_panel(axes[0], "accuracy", "Mean Accuracy",  "(a) Accuracy by method")
    _method_panel(axes[1], "log_loss", "Mean Log Loss",
                  "(b) Log Loss by method", lower_better=True)

    pct_label = f"top {int(top_pct*100)}% configs (n={len(top)})"
    fig.suptitle(f"Method comparison — {pct_label}", fontsize=8, y=1.02)
    fig.tight_layout()
    path = os.path.join(out_dir, "fig5_method_comparison.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"  saved {path}")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 5  Feature importance — accuracy > acc_thresh configs
# ─────────────────────────────────────────────────────────────────────────────
def fig_feature_importance(ok, out_dir, acc_thresh=0.90,
                           top_n=10, show_tput_features=False):
    """
    Shows features most common in configurations with accuracy > acc_thresh.
    """
    targets = [t for t in ["throughput_1", "throughput_2", "throughput_3"]
               if t in ok["target"].unique()]
    n = len(targets)

    fig_w = COL2 if n > 1 else COL1 * 1.5
    fig, axes = plt.subplots(1, n, figsize=(fig_w, 3.4))
    if n == 1:
        axes = [axes]

    for ax, tgt in zip(axes, targets):
        sub   = ok[ok["target"] == tgt].copy()
        top   = sub[sub["accuracy"] >= acc_thresh]
        n_cfg = len(top)

        if n_cfg == 0:
            ax.text(0.5, 0.5, f"No configs\nacc≥{acc_thresh}",
                    ha="center", va="center", transform=ax.transAxes)
            ax.set_title(TARGET_LABELS.get(tgt, tgt), fontweight="bold")
            continue

        ctr = Counter()
        for row in top["features"].dropna():
            for f in row.split(","):
                f = f.strip()
                if f == tgt:
                    continue
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
        ax.set_xlabel("Frequency in configs")
        ax.set_title(f"{TARGET_LABELS.get(tgt, tgt)}\n(acc≥{acc_thresh}, n={n_cfg})",
                     fontweight="bold", fontsize=8)
        ax.xaxis.set_major_formatter(
            matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        ax.tick_params(axis="y", labelsize=7.5)
        ax.axvline(0.5, color="#cccccc", lw=0.5, ls="--", zorder=0)

    handles = [
        mpatches.Patch(facecolor="#2196F3", label="Service 1", edgecolor="none"),
        mpatches.Patch(facecolor="#43A047", label="Service 2", edgecolor="none"),
        mpatches.Patch(facecolor="#E53935", label="Service 3", edgecolor="none"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3,
               frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=[0, 0.07, 1, 1])
    path = os.path.join(out_dir, "fig6_feature_importance.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"  saved {path}")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 6  Consensus DBN graph x3 (one per throughput highlighted)
# ─────────────────────────────────────────────────────────────────────────────
def _load_edges(edges_dir, target):
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


def _node_color(node, highlight_target=None):
    """Color nodes, with special highlight for the target t+1 node."""
    if highlight_target and node == highlight_target:
        return "#FF4444"   # bright red — this is the highlighted target
    if "throughput" in node: return "#C9DCF0"
    if "buffer"     in node: return "#C9EDD9"
    if "latency"    in node: return "#FDDCB0"
    if "data_qual"  in node: return "#E8DAEF"
    if "cores"      in node: return "#FAD7A0"
    return "#EBEBEB"


def fig_consensus_graphs(edges_dir, model_target, out_dir, min_frac=0.5):
    """
    Draw three consensus graphs from the same model (trained on model_target),
    each highlighting a different throughput target node at t+1.
    """
    ctr, n_files = _load_edges(edges_dir, model_target)
    if not ctr:
        print(f"  [skip] no edge files for {model_target}")
        return

    min_count = max(1, round(n_files * min_frac))

    # Filter: above threshold, no t+1->t+1 edges
    edges_use = [
        (p, c, cnt) for (p, c), cnt in ctr.items()
        if cnt >= min_count
        and not (p.endswith("_t1") and c.endswith("_t1"))
    ]

    if not edges_use:
        print(f"  [skip] no edges above threshold for {model_target}")
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

    def edge_color(cnt, parent, child, highlight_target):
        """Two edge types: universal (5/5) dark green, strong (>=3/5) light green."""
        if child == highlight_target:
            return "#FF4444"
        frac = cnt / n_files
        if frac >= 1.0: return "#1A5C35"   # dark green — universal
        return "#52A96A"                    # light green — strong (>=3/5)

    def edge_width(cnt, child, highlight_target):
        base = 0.8 + 2.8 * (cnt / n_files)
        return base * 2.0 if child == highlight_target else base

    # Draw one graph per throughput target
    throughput_targets = [
        ("throughput_1", "throughput_1_t1", "Service 1"),
        ("throughput_2", "throughput_2_t1", "Service 2"),
        ("throughput_3", "throughput_3_t1", "Service 3"),
    ]

    for tgt_name, tgt_node, tgt_label in throughput_targets:
        # Check node exists in graph
        if tgt_node not in G.nodes():
            print(f"  [skip] {tgt_node} not in graph")
            continue

        edge_list   = [(p, c) for p, c, _ in edges_use]
        edge_colors = [edge_color(cnt, p, c, tgt_node)
                       for p, c, cnt in edges_use]
        edge_widths = [edge_width(cnt, c, tgt_node)
                       for _, c, cnt in edges_use]
        node_colors = [_node_color(n, highlight_target=tgt_node)
                       for n in G.nodes()]
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
        ax.text(0, y_lo, "time $t$",   ha="center", fontsize=8,
                style="italic", color="#555")
        ax.text(4, y_lo, "time $t+1$", ha="center", fontsize=8,
                style="italic", color="#555")

        ax.set_title(
            f"Consensus DBN — highlighting predictors of {tgt_label} (t+1)\n"
            f"Model trained on Service 3 throughput",
            fontsize=8, fontweight="bold", pad=40
        )

        univ   = n_files
        strong = round(n_files * 0.6)
        edge_handles = [
            mlines.Line2D([], [], color="#FF4444", lw=2.5,
                          label=f"→ {tgt_label} (highlighted target)"),
            mlines.Line2D([], [], color="#1A5C35", lw=2.5,
                          label=f"Universal ({univ}/{n_files})"),
            mlines.Line2D([], [], color="#52A96A", lw=2.0,
                          label=f"Strong (>={strong}/{n_files})"),
        ]
        node_handles = [
            mpatches.Patch(fc="#FF4444", ec="#666", lw=0.6,
                           label=f"Target: {tgt_label} (t+1)"),
            mpatches.Patch(fc="#C9DCF0", ec="#666", lw=0.6, label="throughput"),
            mpatches.Patch(fc="#C9EDD9", ec="#666", lw=0.6, label="buffer size"),
            mpatches.Patch(fc="#FDDCB0", ec="#666", lw=0.6, label="latency"),
            mpatches.Patch(fc="#E8DAEF", ec="#666", lw=0.6, label="data quality"),
            mpatches.Patch(fc="#FAD7A0", ec="#666", lw=0.6, label="cores"),
        ]
        ax.legend(handles=edge_handles + node_handles,
                  loc="upper center", bbox_to_anchor=(0.5, 1.14),
                  ncol=4, frameon=False, fontsize=6.5)
        ax.set_xlim(-1.2, 5.2)
        ax.axis("off")
        fig.tight_layout()

        path = os.path.join(out_dir, f"fig7_consensus_graph_{tgt_name}.pdf")
        fig.savefig(path)
        plt.close(fig)
        print(f"  saved {path}")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv",                required=True)
    parser.add_argument("--edges_dir",          required=True)
    parser.add_argument("--out",                default="figures")
    parser.add_argument("--min_edge_frac",      type=float, default=0.6)
    parser.add_argument("--top_pct",            type=float, default=0.10)
    parser.add_argument("--acc_thresh",         type=float, default=0.90)
    parser.add_argument("--show_tput_features", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print(f"\nLoading {args.csv} ...")
    df = pd.read_csv(args.csv)
    ok = df[df["error"].isna()].copy()
    print(f"  {len(ok)} successful runs")
    print(f"  targets: {ok['target'].unique().tolist()}")
    print(f"  total failed: {df['error'].notna().sum()}")
    if _has_col(ok, "precision"):
        print("  precision/recall columns: YES")
    else:
        print("  precision/recall columns: NO (run with new script to get them)")

    print("\nFig 1: main results vs baselines ...")
    fig_main_results(ok, args.out)

    print("Fig 2: 3D surface K x n_bins x metric ...")
    fig_3d_surface(ok, args.out)

    print("Fig 3: full sweep configuration profile ...")
    fig_full_profile(ok, args.out)

    print("Fig 4: top 10% configuration profile ...")
    fig_top_profile(ok, args.out, top_pct=args.top_pct)

    print("Fig 5: method comparison ...")
    fig_method_comparison(ok, args.out, top_pct=args.top_pct)

    print("Fig 6: feature importance (acc > thresh) ...")
    fig_feature_importance(ok, args.out,
                           acc_thresh=args.acc_thresh,
                           show_tput_features=args.show_tput_features)

    print("Fig 7: consensus DBN graphs (one per throughput target) ...")
    for tgt in ok["target"].unique():
        fig_consensus_graphs(args.edges_dir, tgt,
                             out_dir=args.out,
                             min_frac=args.min_edge_frac)

    print(f"\nAll outputs saved to: {args.out}/")
    print("\nFiles produced:")
    for f in sorted(os.listdir(args.out)):
        print(f"  {f}")


if __name__ == "__main__":
    main()