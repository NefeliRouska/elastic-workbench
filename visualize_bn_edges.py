import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import re
import textwrap

# ─────────────────────────────────────────────
# 1. Load your learned BN edges
# ─────────────────────────────────────────────
edges = pd.read_csv("BN_edges_prom_all_metrics_wide5.csv")
G = nx.DiGraph()
G.add_edges_from(edges.values)

# ─────────────────────────────────────────────
# 2. Make metric names human-readable
# ─────────────────────────────────────────────
def pretty_metric(s: str) -> str:
    """Clean Prometheus-style metric names for display."""
    s = re.split(r'__', s)[0]           # drop instance/job parts if present
    s = s.strip("_")

    mapping = {
        "container_cpu_usage_seconds_total": "CPU Usage (rate)",
        "container_cpu_system_seconds_total": "CPU System Time (rate)",
        "container_cpu_user_seconds_total": "CPU User Time (rate)",
        "container_memory_usage_bytes": "Memory Usage (bytes)",
        "container_memory_working_set_bytes": "Working Set Memory (bytes)",
        "container_fs_reads_bytes_total": "Disk Reads (bytes/s)",
        "container_fs_writes_bytes_total": "Disk Writes (bytes/s)",
        "container_network_receive_bytes_total": "Network RX (bytes/s)",
        "container_network_transmit_bytes_total": "Network TX (bytes/s)",
        "container_network_receive_packets_total": "Network RX (pkts/s)",
        "container_network_transmit_packets_total": "Network TX (pkts/s)",
        "buffer_size": "Buffer Size",
        "avg_p_latency": "Average Processing Latency",
        "throughput": "Throughput",
    }
    if s in mapping:
        return mapping[s]

    # generic cleanup
    s = s.replace("container_", "")
    s = s.replace("_total", "")
    s = s.replace("_bytes", " (bytes)")
    s = s.replace("_seconds", " (s)")
    s = s.replace("_", " ")
    s = re.sub(r"\s+", " ", s).strip()
    s = " ".join(w.capitalize() if "(" not in w else w for w in s.split())
    return s

# create readable labels for all nodes
labels = {n: "\n".join(textwrap.wrap(pretty_metric(n), width=18)) for n in G.nodes()}

# ─────────────────────────────────────────────
# 3. Draw the graph nicely
# ─────────────────────────────────────────────
plt.figure(figsize=(14, 11))
pos = nx.spring_layout(G, k=0.5, seed=7)

nx.draw_networkx_nodes(G, pos, node_size=1400, node_color="#e6f0ff", edgecolors="#444")
#nx.draw_networkx_edges(G, pos, arrows=True, arrowstyle='-|>', width=1.2, edge_color="#333")
nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, font_family="sans-serif")
nx.draw_networkx_edges(
    G, pos,
    arrows=True,
    arrowstyle='-|>',
    arrowsize=24,              # larger arrowheads so they're visible
    width=1.2,
    edge_color="#333",
    min_source_margin=10,      # move arrows away from node centers
    min_target_margin=12,      # prevents heads from being hidden under target node
)

plt.title("Bayesian Network of System Metrics", fontsize=14, pad=20)
plt.axis("off")
plt.tight_layout()
plt.show()
