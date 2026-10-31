"""LMCache MP server benchmark through ATOM's own adapter classes, no model.

Four worker processes (one per GPU, like ATOM TP4 ranks) register MiniMax-M3-shaped PAGE
views (301 uint8 tensors, 2,995,200 B per 128-token block per rank) with
AtomMPWorkerAdapter; the coordinator plays ATOM's scheduler with AtomMPSchedulerAdapter.
A turn of a conversation with prompt P: lookup(P) -> wait prefetch -> every rank
retrieves [0, P - STEP) -> every rank stores [P - STEP, P), keys carrying the full
prompt as ATOM's do -> end_session. Like ATOM, each rank records one interprocess event
per batch of submissions. Size L1 above the total data written (and allocate it up
front), otherwise eviction and allocation failures distort an arm: check server.log
for "triggering eviction" and the retrieve count in summary.json.

Phases:
  sweep  one conversation, turns 1..T_SWEEP (P = 16k .. T_SWEEP*16k), sequential with a
         short pause; run twice with fresh tokens, the second pass is the measurement.
         Per-op cost vs prompt length without contention.
  burst  for each "CxP" in SB_BURSTS: C conversations' prefixes [0, P - STEP) are stored
         first (not measured), then all C conversations run their turn at prompt P at
         the same instant (an arrival burst). Repeated SB_BURST_R times with fresh
         tokens.

Records (kind, prompt_tokens, value_s, tag) go to OUT/<phase>.json.

usage: python srvbench.py OUT PORT sweep|burst
env: SB_T_SWEEP (32), SB_BURSTS ("16x131072,32x32768"), SB_BURST_R (3), SB_STEP (16384)
"""

import json
import multiprocessing as mp
import os
import sys
import time

STEP = int(os.environ.get("SB_STEP", 16384))
BLOCK = 128
UNITS = [16384] * 179 + [512] * 122  # M3 per rank: K/V/index/draft K,V and fp8 scales
MODEL = "srvbench-m3"
WORLD = 4
T_SWEEP = int(os.environ.get("SB_T_SWEEP", 32))
BURSTS = [
    tuple(int(x) for x in b.split("x"))
    for b in os.environ.get("SB_BURSTS", "16x131072,32x32768").split(",")
]
BURST_R = int(os.environ.get("SB_BURST_R", 3))
MAXLEN = max([T_SWEEP * STEP] + [p for _, p in BURSTS])
_TOK_CACHE = {}


def tokens_for(seed):
    """Prompt tokens of conversation ``seed`` up to MAXLEN (callers slice them)."""
    toks = _TOK_CACHE.get(seed)
    if toks is None:
        import numpy as np

        toks = np.random.default_rng(seed).integers(0, 200_000, MAXLEN).tolist()
        _TOK_CACHE[seed] = toks
        while len(_TOK_CACHE) > 80:
            _TOK_CACHE.pop(next(iter(_TOK_CACHE)))
    return toks


def worker(rank, port, n_blocks, conn):
    import torch
    import zmq
    from lmcache.integration.atom import AtomMPTransferSpec, AtomMPWorkerAdapter
    from lmcache.integration.atom.multi_process_adapter import AtomMPParallelConfig
    from lmcache.v1.multiprocess.group_view import EngineGroupInfo

    torch.cuda.set_device(rank)
    tensors = {
        f"page.{i}.u{u}": torch.zeros(
            (n_blocks, 1, u), dtype=torch.uint8, device="cuda"
        )
        for i, u in enumerate(UNITS)
    }
    groups = [
        EngineGroupInfo(
            engine_group_id=0,
            layer_indices=tuple(i for i, u in enumerate(UNITS) if u == size),
            tokens_per_block=BLOCK,
        )
        for size in (16384, 512)
    ]
    ad = AtomMPWorkerAdapter(
        f"tcp://127.0.0.1:{port}",
        zmq.Context.instance(),
        MODEL,
        BLOCK,
        AtomMPParallelConfig(world_size=WORLD, worker_id=rank, tp_size=WORLD),
        transfer_mode="lmcache_driven",
    )
    ad.register_kv_caches(tensors, engine_group_infos=groups)
    conn.send(("ready", rank))
    prof = None
    if os.environ.get("SB_PROFILE") and rank == 0:
        import cProfile

        prof = cProfile.Profile()
    inflight = []
    while True:
        cmds = []
        while conn.poll(0):
            cmds.append(conn.recv())
        if any(c[0] == "stop" for c in cmds):
            break
        if cmds:
            t_ev = time.perf_counter()
            ev = torch.cuda.Event(
                interprocess=True
            )  # one per "step", like ATOM's start_load_kv
            ev.record(torch.cuda.current_stream())
            conn.send(("evt", rank, time.perf_counter() - t_ev))
            for kind, op_id, rid, seed, plen, start, end, blk0 in cmds:
                spec = AtomMPTransferSpec(
                    token_ids=tokens_for(seed)[:plen],
                    block_ids=[list(range(blk0 + start // BLOCK, blk0 + end // BLOCK))],
                    start=start,
                    end=end,
                )
                t = time.perf_counter()
                if prof:
                    prof.enable()
                fut = (
                    ad.submit_retrieve_request
                    if kind == "retrieve"
                    else ad.submit_store_request
                )(rid, spec, ev)
                if prof:
                    prof.disable()
                inflight.append((op_id, t, time.perf_counter() - t, fut))
        still = []
        for op_id, t, t_sub, fut in inflight:
            if fut is None or fut.query():
                ok = bool(fut.result()) if fut is not None else False
                conn.send(("done", rank, op_id, time.perf_counter() - t, t_sub, ok))
            else:
                still.append((op_id, t, t_sub, fut))
        inflight = still
        time.sleep(0.0005)
    ad.shutdown()
    if prof:
        prof.dump_stats(
            os.path.join(os.environ["SB_PROFILE"], f"submit_rank{rank}.prof")
        )


class Coordinator:
    def __init__(self, port, n_blocks):
        import zmq
        from lmcache.integration.atom import AtomMPSchedulerAdapter
        from lmcache.integration.atom.multi_process_adapter import AtomMPParallelConfig

        ctx = mp.get_context("spawn")
        self.pipes, self.procs = [], []
        for r in range(WORLD):
            a, b = ctx.Pipe()
            p = ctx.Process(target=worker, args=(r, port, n_blocks, b), daemon=True)
            p.start()
            self.pipes.append(a)
            self.procs.append(p)
        for a in self.pipes:
            assert a.recv()[0] == "ready"
        self.sch = AtomMPSchedulerAdapter(
            f"tcp://127.0.0.1:{port}",
            zmq.Context.instance(),
            MODEL,
            BLOCK,
            AtomMPParallelConfig(world_size=WORLD, worker_id=0, tp_size=WORLD),
        )
        self.op_seq = 0
        self.rec = []

    def close(self):
        for a in self.pipes:
            a.send(("stop",))
        for p in self.procs:
            p.join(60)
        self.sch.shutdown()

    def run_turns(self, tasks, tag):
        """Drive tasks to completion; a task is dict(seed, plen, blk0, store_only)."""
        pending = {}
        t_start = time.perf_counter()

        def send(task, kind, start, end):
            self.op_seq += 1
            for a in self.pipes:
                a.send(
                    (
                        kind,
                        self.op_seq,
                        task["rid"],
                        task["seed"],
                        task["plen"],
                        start,
                        end,
                        task["blk0"],
                    )
                )
            pending[self.op_seq] = {
                "task": task,
                "left": WORLD,
                "kind": kind,
                "t": time.perf_counter(),
                "ok": True,
                "lat": [],
                "sub": [],
            }

        for task in tasks:
            task["rid"] = f"sb-{task['seed']}-{task['plen']}-{tag}"
            task["t0"] = time.perf_counter()
            if task.get("store_only"):
                task["state"] = "storing"
                send(task, "store", 0, task["plen"])
            else:
                self.sch.maybe_submit_lookup_request(
                    task["rid"], tokens_for(task["seed"])[: task["plen"]]
                )
                task["state"] = "poll"
        done = 0
        while done < len(tasks):
            for task in tasks:
                if task["state"] != "poll":
                    continue
                hit = self.sch.check_lookup_result(task["rid"])
                if hit is None:
                    continue
                p = task["plen"]
                self.rec.append(("lookup", p, time.perf_counter() - task["t0"], tag))
                self.rec.append(("hit_tokens", p, hit, tag))
                self.sch.cleanup_lookup_result(task["rid"])
                prev = p - STEP
                if prev > 0 and hit >= prev:
                    task["state"] = "retrieving"
                    send(task, "retrieve", 0, prev)
                else:
                    if prev > 0:
                        self.rec.append(("short_hit", p, prev - hit, tag))
                    task["state"] = "storing"
                    send(task, "store", prev, p)
            for a in self.pipes:
                while a.poll(0):
                    msg = a.recv()
                    if msg[0] == "evt":
                        self.rec.append(("step_event", 0, msg[2], tag))
                        continue
                    _, rank, op_id, lat, t_sub, ok = msg
                    o = pending[op_id]
                    o["left"] -= 1
                    o["ok"] &= ok
                    o["lat"].append(lat)
                    o["sub"].append(t_sub)
                    if o["left"]:
                        continue
                    del pending[op_id]
                    task, kind, p = o["task"], o["kind"], o["task"]["plen"]
                    if task.get("store_only"):
                        self.sch.end_session(task["rid"])
                        task["state"] = "done"
                        done += 1
                        continue
                    self.rec.append((kind, p, time.perf_counter() - o["t"], tag))
                    self.rec.append((kind + "_rank_max", p, max(o["lat"]), tag))
                    self.rec.append((kind + "_submit", p, max(o["sub"]), tag))
                    if not o["ok"]:
                        self.rec.append((kind + "_failed", p, 1, tag))
                    if kind == "retrieve":
                        task["state"] = "storing"
                        send(task, "store", p - STEP, p)
                    else:
                        self.sch.end_session(task["rid"])
                        self.rec.append(
                            ("turn", p, time.perf_counter() - task["t0"], tag)
                        )
                        task["state"] = "done"
                        done += 1
            time.sleep(0.0005)
        return time.perf_counter() - t_start


def main(out, port, phase):
    seed0 = (os.getpid() << 20) ^ (time.time_ns() & 0xFFFFFFF)
    marks = []
    if phase == "sweep":
        co = Coordinator(port, T_SWEEP * STEP // BLOCK)
        for pas in range(2):
            seed = seed0 + pas
            t_wall = time.time()
            for t in range(1, T_SWEEP + 1):
                co.run_turns(
                    [{"seed": seed, "plen": t * STEP, "blk0": 0}], f"pass{pas}"
                )
                time.sleep(0.3)
            marks.append(
                {"tag": f"pass{pas}", "wall_start": t_wall, "wall_end": time.time()}
            )
    else:
        co = Coordinator(port, max(c * p // BLOCK for c, p in BURSTS))
        for c, p in BURSTS:
            for r in range(BURST_R):
                seeds = [seed0 + 10_000 * (len(marks) + 1) + i for i in range(c)]
                prefill = [
                    {
                        "seed": s,
                        "plen": p - STEP,
                        "blk0": i * (p // BLOCK),
                        "store_only": True,
                    }
                    for i, s in enumerate(seeds)
                ]
                co.run_turns(prefill, "prefill")
                time.sleep(3.0)
                tag = f"{c}x{p}#{r}"
                t_wall = time.time()
                makespan = co.run_turns(
                    [
                        {"seed": s, "plen": p, "blk0": i * (p // BLOCK)}
                        for i, s in enumerate(seeds)
                    ],
                    tag,
                )
                marks.append(
                    {
                        "tag": tag,
                        "C": c,
                        "P": p,
                        "makespan_s": makespan,
                        "wall_start": t_wall,
                        "wall_end": time.time(),
                    }
                )
                print(f"burst {tag}: makespan {makespan:.2f} s", flush=True)
                time.sleep(3.0)
    co.close()
    json.dump(
        {
            "phase": phase,
            "step": STEP,
            "bytes_per_token_rank": sum(UNITS) / BLOCK,
            "marks": marks,
            "records": co.rec,
        },
        open(os.path.join(out, f"{phase}.json"), "w"),
    )
    print(f"{phase} done: {len(co.rec)} records", flush=True)


if __name__ == "__main__":
    os.makedirs(sys.argv[1], exist_ok=True)
    main(sys.argv[1], int(sys.argv[2]), sys.argv[3])
