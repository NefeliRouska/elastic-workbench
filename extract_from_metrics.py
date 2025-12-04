import sys

# Usage:
#   python3 extract_slice.py <input_csv> <output_csv> <YYYY-MM-DD> <HH:MM> <HH:MM>
# Example:
#   python3 extract_slice.py share/metrics/metrics.csv share/metrics/metrics_filtered.csv 2025-11-10 12:24 12:39

if len(sys.argv) != 6:
    print("Usage: python3 extract_slice.py <input_csv> <output_csv> <YYYY-MM-DD> <HH:MM> <HH:MM>")
    sys.exit(1)

in_path, out_path, date_str, start_hhmm, end_hhmm = sys.argv[1:6]
start_key = f"{date_str} {start_hhmm}:00"
end_key   = f"{date_str} {end_hhmm}:59"

kept = 0
with open(in_path, "r", newline="") as fin, open(out_path, "w", newline="") as fout:
    header = fin.readline()
    if header:
        fout.write(header)

    for line in fin:
        if not line:
            continue
        # first 19 chars must be 'YYYY-MM-DD HH:MM:SS'
        if len(line) < 19:
            continue
        key = line[:19]
        # quick shape check
        if not (key[4] == "-" and key[7] == "-" and key[10] == " " and key[13] == ":" and key[16] == ":"):
            continue
        if start_key <= key <= end_key:
            fout.write(line)
            kept += 1

print(f"Kept {kept} rows -> {out_path}")
