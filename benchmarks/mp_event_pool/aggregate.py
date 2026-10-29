"""Aggregate repeated srvbench runs named ARM@rN into one table (mean and min..max).

usage: python aggregate.py RUNS_DIR [ARM ...]
Reads RUNS_DIR/ARM@r*/summary.json (written by analyze.py) and prints a markdown table.
"""

import glob
import json
import os
import statistics as st
import sys

METRICS = [
    ("sweep store p50 (ms)", lambda s: s["sweep_store_ms"]["p50"]),
    ("sweep retrieve p50 (ms)", lambda s: s["sweep_retrieve_ms"]["p50"]),
    ("burst store p50 (ms)", lambda s: s["burst_store_ms"]["p50"]),
    ("burst store p90 (ms)", lambda s: s["burst_store_ms"]["p90"]),
    ("burst retrieve p50 (ms)", lambda s: s["burst_retrieve_ms"]["p50"]),
    ("burst retrieve p90 (ms)", lambda s: s["burst_retrieve_ms"]["p90"]),
]


def cell(xs):
    if not xs:
        return "-"
    m = st.mean(xs)
    fmt = (lambda v: f"{v:.2f}") if m < 20 else (lambda v: f"{v:.0f}")
    return f"{fmt(m)} ({fmt(min(xs))}–{fmt(max(xs))})" if len(xs) > 1 else fmt(m)


def main(root, arms):
    if not arms:
        arms = sorted(
            {
                os.path.basename(d).split("@")[0]
                for d in glob.glob(os.path.join(root, "*@r*"))
            }
        )
    data = {}
    for a in arms:
        runs = [
            json.load(open(f))
            for f in sorted(glob.glob(os.path.join(root, f"{a}@r*", "summary.json")))
        ]
        data[a] = runs
    bursts = sorted(
        {k for a in arms for s in data[a] for k in s.get("burst_makespan_s", {})},
        key=lambda k: [int(x) for x in k.split("x")],
    )
    metrics = (
        METRICS[:2]
        + [
            (f"burst {k} makespan (s)", lambda s, k=k: s["burst_makespan_s"][k])
            for k in bursts
        ]
        + METRICS[2:]
    )
    print("| metric | " + " | ".join(f"{a} (n={len(data[a])})" for a in arms) + " |")
    print("|---|" + "---|" * len(arms))
    for name, get in metrics:
        row = []
        for a in arms:
            xs = []
            for s in data[a]:
                try:
                    xs.append(get(s))
                except (KeyError, TypeError):
                    pass
            row.append(cell(xs))
        print(f"| {name} | " + " | ".join(row) + " |")
    fails = {
        a: sum(
            s.get("sweep_failed_ops", 0) + s.get("burst_failed_ops", 0) for s in data[a]
        )
        for a in arms
    }
    print("\nfailed ops: " + ", ".join(f"{a}={n}" for a, n in fails.items()))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
