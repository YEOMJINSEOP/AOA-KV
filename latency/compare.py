"""Print AOA-KV vs QUEST latency side by side from the CSV bench_latency.py writes.

Rows are matched on (model, context_len, token_budget, page_size, decode_len),
and only the newest row per (method, config) is used — re-running one arm
overwrites its side of the comparison instead of duplicating the table.
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

KEY = ("model", "context_len", "token_budget", "page_size", "decode_len")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True)
    args = p.parse_args()

    path = Path(args.csv)
    if not path.exists():
        raise SystemExit(f"no such CSV: {path}")

    latest = {}
    for row in csv.DictReader(path.open()):
        latest[(row["method"],) + tuple(row[k] for k in KEY)] = row

    configs = defaultdict(dict)
    for (method, *key), row in latest.items():
        configs[tuple(key)][method] = row

    hdr = (f"{'context':>8} {'budget':>7} {'page':>5} "
           f"{'QUEST prefill':>14} {'AOA prefill':>12} {'Δ%':>7}   "
           f"{'QUEST dec/tok':>14} {'AOA dec/tok':>12} {'Δ%':>7}")
    print(hdr)
    print("-" * len(hdr))

    def pct(base, new):
        if base is None or new is None:
            return "     -"
        return f"{(new - base) / base * 100:+6.1f}"

    for key in sorted(configs, key=lambda k: (k[0], int(k[1]), int(k[2]))):
        model, ctx, budget, page, _dec = key
        q, a = configs[key].get("QUEST"), configs[key].get("AOA_KV")
        qp = float(q["prefill_ms"]) if q else None
        ap = float(a["prefill_ms"]) if a else None
        qd = float(q["decode_ms_per_token"]) if q else None
        ad = float(a["decode_ms_per_token"]) if a else None
        fmt = lambda v, nd: ("-" if v is None else f"{v:.{nd}f}")
        print(f"{ctx:>8} {budget:>7} {page:>5} "
              f"{fmt(qp,2):>14} {fmt(ap,2):>12} {pct(qp,ap):>7}   "
              f"{fmt(qd,3):>14} {fmt(ad,3):>12} {pct(qd,ad):>7}")

    models = {k[0] for k in configs}
    gpus = {r["gpu"] for r in latest.values()}
    md5s = {(r["method"], r["kernel_md5"][:8]) for r in latest.values()}
    print()
    print(f"model(s): {', '.join(sorted(models))}")
    print(f"gpu(s)  : {', '.join(sorted(gpus))}")
    print(f"kernels : {', '.join(f'{m}={h}' for m, h in sorted(md5s))}")
    print("Δ% is AOA-KV relative to QUEST (negative = AOA-KV faster).")


if __name__ == "__main__":
    main()
