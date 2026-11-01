"""Summarize srvbench.py output: per-operation latency and burst makespan.

usage: python analyze.py RUN_DIR [RUN_DIR ...]

For each run directory prints, from the measured sweep pass (pass1, one conversation,
no contention) the median store / retrieve / lookup latency and the median latency of
the slowest rank, and from the burst phase the mean makespan per "CxP" burst (C
conversations arriving at once with prompt P). Writes RUN_DIR/summary.json.
"""

import json
import os
import statistics as st
import sys


def med(xs):
    return st.median(xs) * 1e3 if xs else float("nan")


def p90(xs):
    if not xs:
        return float("nan")
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(0.9 * (len(xs) - 1) + 0.5))] * 1e3


def summarize(d):
    out = {"run": os.path.basename(os.path.normpath(d))}
    cfg = os.path.join(d, "config.txt")
    if os.path.exists(cfg):
        out["config"] = open(cfg).read().strip()
    sw = os.path.join(d, "sweep.json")
    if os.path.exists(sw):
        recs = [r for r in json.load(open(sw))["records"] if r[3] == "pass1"]
        for kind in (
            "store",
            "retrieve",
            "lookup",
            "store_rank_max",
            "retrieve_rank_max",
            "store_submit",
            "retrieve_submit",
        ):
            xs = [r[2] for r in recs if r[0] == kind]
            out[f"sweep_{kind}_ms"] = {
                "n": len(xs),
                "p50": round(med(xs), 2),
                "p90": round(p90(xs), 2),
            }
        out["sweep_failed_ops"] = sum(1 for r in recs if r[0].endswith("_failed"))
    bu = os.path.join(d, "burst.json")
    if os.path.exists(bu):
        b = json.load(open(bu))
        spans = {}
        for m in b["marks"]:
            spans.setdefault(f"{m['C']}x{m['P']}", []).append(m["makespan_s"])
        out["burst_makespan_s"] = {k: round(st.mean(v), 2) for k, v in spans.items()}
        recs = [r for r in b["records"] if r[3] not in ("prefill",)]
        for kind in ("store", "retrieve"):
            xs = [r[2] for r in recs if r[0] == kind]
            out[f"burst_{kind}_ms"] = {
                "n": len(xs),
                "p50": round(med(xs), 2),
                "p90": round(p90(xs), 2),
            }
        out["burst_failed_ops"] = sum(1 for r in recs if r[0].endswith("_failed"))
    json.dump(out, open(os.path.join(d, "summary.json"), "w"), indent=1)
    return out


if __name__ == "__main__":
    for d in sys.argv[1:]:
        print(json.dumps(summarize(d)))
