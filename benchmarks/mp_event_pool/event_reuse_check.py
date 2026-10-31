"""Reusing interprocess events across processes on ROCm 7.2.4.

ROCm 7.2.4's IPCEvent::streamWait indexes signal[read_index] without the "% 32" that
record/query use, so an imported event can be stream-waited only while the owner has
recorded it at most 32 times. Two schemes are checked here, two processes on GPU 0:

  wait_rot: owner re-records one event per round (after a ~10 ms kernel), switching to a
            fresh event every ROT records; the peer imports each handle once (cache) and
            makes its stream wait on it. (client producer events, server waits)
  query:    owner re-records ONE event forever; the peer imported it once and polls
            query(). (server completion events, client polls)

Unsafe if the peer ever sees completion before the owner's kernel finished (owner's
synchronize() time, CLOCK_MONOTONIC across processes), or if query() right after a
re-record reports the previous, completed record. Then times the per-op event calls.

usage: python event_reuse_check.py [rounds] [rot]
  (rot > 32 reproduces the ROCm 7.2.x streamWait bug)
"""

import multiprocessing as mp
import statistics as st
import sys
import time

import torch


def calib(ms):
    torch.cuda._sleep(1000)
    torch.cuda.synchronize()
    a = time.perf_counter()
    torch.cuda._sleep(10_000_000)
    torch.cuda.synchronize()
    return int(10_000_000 * ms / ((time.perf_counter() - a) * 1e3))


def owner(conn, rounds, mode, rot):
    torch.cuda.set_device(0)
    cyc = calib(10.0)
    s = torch.cuda.Stream()
    ev, uses = None, rot
    conn.recv()
    for i in range(rounds):
        if ev is None or (mode == "wait_rot" and uses >= rot):
            ev, uses = torch.cuda.Event(interprocess=True), 0
        with torch.cuda.stream(s):
            torch.cuda._sleep(cyc)
            ev.record(s)
        uses += 1
        conn.send(("rec", i, ev.ipc_handle()))
        s.synchronize()
        conn.send(("done", i, time.perf_counter()))
        if not conn.poll(60):
            print(f"owner: peer silent in round {i}", flush=True)
            return
        conn.recv()
    conn.send(("end",))


def peer(conn, mode, q):
    torch.cuda.set_device(0)
    dev = torch.cuda.current_device()
    s = torch.cuda.Stream()
    cache = {}
    conn.send("ready")
    early, stale, lag, n = 0, 0, [], 0
    while True:
        msg = conn.recv()
        if msg[0] == "end":
            break
        _, i, h = msg
        if h not in cache:
            cache[h] = torch.cuda.Event.from_ipc_handle(dev, h)
        imp = cache[h]
        t0 = time.perf_counter()
        if mode == "wait_rot":
            imp.wait(s)
            c = torch.cuda.Event()
            c.record(s)
            while not c.query():
                if time.perf_counter() - t0 > 20:
                    q.put((mode, -1, -1, 0.0, 0.0, i, len(cache)))
                    return
        else:
            if imp.query():
                stale += 1
            while not imp.query():
                if time.perf_counter() - t0 > 20:
                    q.put((mode, -1, -1, 0.0, 0.0, i, len(cache)))
                    return
        t_seen = time.perf_counter()
        _, _, t_done = conn.recv()
        early += t_seen < t_done - 1e-3
        lag.append((t_seen - t_done) * 1e3)
        n += 1
        conn.send("ok")
    q.put((mode, early, stale, st.median(lag), max(lag), n, len(cache)))


def cost_owner(conn, busy_ms):
    torch.cuda.set_device(0)
    cyc = calib(busy_ms) if busy_ms else 0
    t = {
        "create_ipc": [],
        "first_record_ipc": [],
        "re_record_ipc": [],
        "ipc_handle": [],
        "destroy_ipc": [],
        "record_plain": [],
    }
    pooled = torch.cuda.Event(interprocess=True)
    pooled.record()
    for _ in range(30):
        if cyc:
            torch.cuda._sleep(cyc)
        a = time.perf_counter()
        e = torch.cuda.Event(interprocess=True)
        t["create_ipc"].append(time.perf_counter() - a)
        a = time.perf_counter()
        e.record()
        t["first_record_ipc"].append(time.perf_counter() - a)
        a = time.perf_counter()
        h = e.ipc_handle()
        t["ipc_handle"].append(time.perf_counter() - a)
        a = time.perf_counter()
        pooled.record()
        t["re_record_ipc"].append(time.perf_counter() - a)
        p = torch.cuda.Event()
        a = time.perf_counter()
        p.record()
        t["record_plain"].append(time.perf_counter() - a)
        conn.send(h)
        conn.recv()  # peer imported it
        a = time.perf_counter()
        del e
        t["destroy_ipc"].append(time.perf_counter() - a)
        torch.cuda.synchronize()
    conn.send(None)
    conn.send({k: 1e3 * st.median(v) for k, v in t.items()})


def cost_peer(conn, busy_ms, q):
    torch.cuda.set_device(0)
    dev = torch.cuda.current_device()
    cyc = calib(busy_ms) if busy_ms else 0
    s = torch.cuda.Stream()
    t = {"import": [], "wait": [], "destroy_imported": []}
    while True:
        h = conn.recv()
        if h is None:
            break
        if cyc:
            with torch.cuda.stream(s):
                torch.cuda._sleep(
                    cyc
                )  # the peer's own GPU is busy, like the server mid-copy
        a = time.perf_counter()
        i = torch.cuda.Event.from_ipc_handle(dev, h)
        t["import"].append(time.perf_counter() - a)
        a = time.perf_counter()
        i.wait(s)
        t["wait"].append(time.perf_counter() - a)
        a = time.perf_counter()
        del i
        t["destroy_imported"].append(time.perf_counter() - a)
        conn.send("ok")
        torch.cuda.synchronize()
    own = conn.recv()
    q.put((own, {k: 1e3 * st.median(v) for k, v in t.items()}))


if __name__ == "__main__":
    rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    rot = int(sys.argv[2]) if len(sys.argv) > 2 else 31
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    for mode in ("wait_rot", "query"):
        a, b = ctx.Pipe()
        p1 = ctx.Process(target=owner, args=(a, rounds, mode, rot))
        p2 = ctx.Process(target=peer, args=(b, mode, q))
        p1.start()
        p2.start()
        p1.join(300)
        p2.join(60)
        for pr in (p1, p2):
            if pr.is_alive():
                pr.kill()
        if q.empty():
            print(f"{mode}: peer crashed (see traceback)", flush=True)
            continue
        m, early, stale, lag50, lagmax, n, nimp = q.get()
        print(
            f"{m:8s}: {n}/{rounds} rounds ({nimp} imports, "
            f"rot={rot if m == 'wait_rot' else 'none'}): "
            f"seen-before-done {early}, stale-complete after re-record {stale}, "
            f"lag median {lag50:.2f} ms max {lagmax:.2f} ms",
            flush=True,
        )
    for busy in (0, 50):
        a, b = ctx.Pipe()
        p1 = ctx.Process(target=cost_owner, args=(a, busy))
        p2 = ctx.Process(target=cost_peer, args=(b, busy, q))
        p1.start()
        p2.start()
        p1.join(300)
        p2.join(60)
        own, peer_t = q.get()
        tag = f"own GPU busy ({busy} ms kernel)" if busy else "GPU idle"
        print(
            f"op costs, {tag}: owner "
            + ", ".join(f"{k} {v:.3f}" for k, v in own.items())
            + " | peer "
            + ", ".join(f"{k} {v:.3f}" for k, v in peer_t.items())
            + " ms",
            flush=True,
        )
