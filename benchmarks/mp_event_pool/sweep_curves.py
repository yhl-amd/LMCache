"""Turn srvbench sweep runs into per-prompt-length latency curves.

usage: python sweep_curves.py RUNS_DIR ARM [ARM ...] [--kinds retrieve,store,turn]

Reads RUNS_DIR/ARM@r*/sweep.json (written by srvbench.py), keeps the measured pass
(tag "pass1"), and for every prompt length averages each operation's latency over
the rounds of an arm. Writes RUNS_DIR/sweep_curves.json and RUNS_DIR/sweep_curves.csv
(columns: arm, kind, prompt_tokens, mean_ms, min_ms, max_ms, rounds) and prints the
value at a few prompt lengths.

sweep.json "records" are [kind, prompt_tokens, seconds, tag]. Kinds in the sweep:
  lookup     scheduler lookup until the prefetch answer
  retrieve   all 4 ranks read back [0, P - 16k)       (from the 2nd turn on)
  store      all 4 ranks store the new [P - 16k, P)
  turn       lookup + retrieve + store of one turn
  *_rank_max slowest rank of the operation; *_submit time spent in the submit call
"""

import argparse
import csv
import glob
import json
import os
import statistics as st


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs_dir")
    ap.add_argument("arms", nargs="+")
    ap.add_argument("--kinds", default="retrieve,store,turn,lookup")
    args = ap.parse_args()
    kinds = args.kinds.split(",")
    curves, rows = {}, []
    for arm in args.arms:
        runs = sorted(glob.glob(os.path.join(args.runs_dir, f"{arm}@r*", "sweep.json")))
        if not runs:
            print(f"{arm}: no runs")
            continue
        curves[arm] = {}
        for kind in kinds:
            per = {}
            for path in runs:
                for k, p, v, tag in json.load(open(path))["records"]:
                    if k == kind and tag == "pass1":
                        per.setdefault(p, []).append(v * 1e3)
            pts = []
            for p, vs in sorted(per.items()):
                pts.append([p, round(st.mean(vs), 1)])
                rows.append(
                    [
                        arm,
                        kind,
                        p,
                        round(st.mean(vs), 1),
                        round(min(vs), 1),
                        round(max(vs), 1),
                        len(vs),
                    ]
                )
            curves[arm][kind] = pts
    json.dump(curves, open(os.path.join(args.runs_dir, "sweep_curves.json"), "w"))
    with open(os.path.join(args.runs_dir, "sweep_curves.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            ["arm", "kind", "prompt_tokens", "mean_ms", "min_ms", "max_ms", "rounds"]
        )
        w.writerows(rows)
    show = [65536, 131072, 262144, 524288]
    for kind in kinds:
        print(f"\n{kind} (ms) at prompt " + ", ".join(f"{p // 1024}k" for p in show))
        for arm, c in curves.items():
            d = dict(c.get(kind, []))
            print(
                f"  {arm:12s} "
                + "  ".join(f"{d.get(p, float('nan')):8.1f}" for p in show)
            )


if __name__ == "__main__":
    main()
