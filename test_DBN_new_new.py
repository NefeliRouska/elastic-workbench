import ast
import numpy as np
import pandas as pd

from sklearn.preprocessing import KBinsDiscretizer
from pgmpy.inference import VariableElimination

from full_dynamic_bn_learn_final_new_new import build_dbn_model_2s


# ============================================================
# CONFIG
# ============================================================
CSV_PATH = "prom_all_metrics_wide.csv"
TRAIN_FRAC = 0.8
TARGET = "throughput"

N_BINS = 4
EVIDENCE_MODE = "full"
EVIDENCE_SUBSET = ["throughput", "avg_p_latency", "cores", "data_quality"]


# ============================================================
# LOAD + CLEAN
# ============================================================
def load_and_clean(path):

    df = pd.read_csv(path)

    if "s_config" in df.columns:
        def parse(val):
            try:
                d = ast.literal_eval(val)
                return d if isinstance(d, dict) else {}
            except:
                return {}

        cfg = df["s_config"].apply(parse).apply(pd.Series)
        cfg = cfg.rename(columns=lambda c: f"s_config_{c}")
        df = pd.concat([df.drop(columns=["s_config"]), cfg], axis=1)

    # drop timestamp-like columns
    for col in list(df.columns):
        if "time" in col.lower():
            df.drop(columns=[col], inplace=True)

    df = df.apply(pd.to_numeric, errors="ignore")

    # ⭐ critical fix
    df = df.select_dtypes(include=[np.number])
    df = df.dropna().reset_index(drop=True)

    return df


# ============================================================
# DISCRETIZER (train fitted)
# ============================================================
class Discretizer:

    def __init__(self, bins=4):
        self.bins = bins
        self.models = {}

    def fit(self, df):

        for c in df.columns:
            kbd = KBinsDiscretizer(
                n_bins=self.bins,
                encode="ordinal",
                strategy="kmeans"
            )
            kbd.fit(df[[c]])
            self.models[c] = kbd

    def transform(self, df):

        out = df.copy()

        for c, m in self.models.items():
            out[c] = m.transform(out[[c]]).astype(int).flatten()

        return out.astype(int)


# ============================================================
# EVALUATION
# ============================================================
def evaluate(model_2s, test_df):
    from pgmpy.inference import VariableElimination

    test_df = test_df.reset_index(drop=True)
    infer = VariableElimination(model_2s)

    # --- FIX: always define ev_cols ---
    ev_cols = list(test_df.columns)
    if EVIDENCE_MODE == "subset":
        ev_cols = EVIDENCE_SUBSET

    model_nodes = set(model_2s.nodes())
    valid_cols = [c for c in ev_cols if f"{c}_t" in model_nodes]

    n = len(test_df) - 1
    correct = 0
    probs_true = []

    for t in range(n):
        evidence = {
            f"{c}_t": int(test_df.iloc[t][c])
            for c in valid_cols
        }

        true_next = int(test_df.iloc[t + 1][TARGET])

        q = infer.query([f"{TARGET}_t1"], evidence=evidence, show_progress=False)
        probs = q.values / q.values.sum()

        pred = int(np.argmax(probs))
        probs_true.append(float(probs[true_next]))

        if pred == true_next:
            correct += 1

    return {
        "accuracy": correct / n,
        "mean_prob_true": float(np.mean(probs_true))
    }


# ============================================================
# MAIN PIPELINE
# ============================================================
def main():

    raw = load_and_clean(CSV_PATH)

    split = int(TRAIN_FRAC * len(raw))

    train_raw = raw.iloc[:split]
    test_raw = raw.iloc[split:]

    disc = Discretizer(N_BINS)
    disc.fit(train_raw)

    train_ready = disc.transform(train_raw)
    test_ready = disc.transform(test_raw)

    model_2s, *_ = build_dbn_model_2s(train_ready)

    res = evaluate(model_2s, test_ready)

    print("\nRESULTS")
    print(res)


if __name__ == "__main__":
    main()
