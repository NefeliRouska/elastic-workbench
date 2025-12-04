import pandas as pd, glob, os, re
from functools import reduce

folder = "prom_export_perturbation"  # your folder name

def safe_col(metric, labels):
    # keep only the most useful labels to avoid huge names
    keep = {}
    for kv in labels.split(';'):
        if '=' not in kv:
            continue
        k,v = kv.split('=',1)
        if k in {'container','instance','job'}:
            keep[k] = v
    label = "_".join(f"{k}={keep[k]}" for k in sorted(keep)) if keep else "nolabels"
    return re.sub(r'[^A-Za-z0-9_]+', '_', f"{metric}__{label}").strip('_')

dfs = []
for f in glob.glob(os.path.join(folder, "*.csv")):
    try:
        df = pd.read_csv(f)
        if not {'metric','labels','timestamp','value'}.issubset(df.columns):
            continue
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        df = df.dropna(subset=['timestamp','value'])
        colname = safe_col(df['metric'].iloc[0], df['labels'].iloc[0])
        dfs.append(df[['timestamp','value']].rename(columns={'value': colname}).sort_values('timestamp'))
    except Exception:
        pass

wide = reduce(lambda l,r: pd.merge_asof(l, r, on='timestamp', tolerance=pd.Timedelta('1s')), dfs)
wide = (wide.set_index('timestamp')
              .resample('15s').mean()
              .interpolate('time'))
wide.to_csv("baseline_merged_perturbation.csv")
print("Merged shape:", wide.shape)
