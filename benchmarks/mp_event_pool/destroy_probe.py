"""Does destroying a HIP IPC event stall other threads? (ROCm 7.2.x)

Background: on ROCm 7.2.x, ~IPCEvent() calls ihipHostUnregister, which runs
SyncAllStreams() (rocm-systems issue #7520; fixed in develop by #7710, 7.2
backport #11529 not merged). So a destroy waits for all work queued on the
device. This probe measures that wait and whether it also stalls HIP calls made
by other threads, on one GPU, in one process (like one engine rank or one
LMCache server device context):

  * a "forward" thread keeps launching tiny kernels and recording events on its
    own stream, timing every call;
  * per trial, a busy kernel (~BUSY_MS) goes onto another stream, then an event
    that completed long ago is destroyed, and the destroy is timed.

Arms (each NTRIALS trials):
  bg_ipc     destroy an IPC event on a background thread while the GPU is busy
  bg_plain   same, a plain event (control: is it IPC-specific?)
  bg_none    no destroy (baseline for the forward thread's latencies)
  fwd_busy   destroy an IPC event on the forward thread while the GPU is busy
  fwd_idle   destroy an IPC event on the forward thread after the busy work is done

An optional consumer process imports each IPC event handle and waits on it,
as the LMCache server does, before the event is destroyed.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import statistics
import threading
import time

import torch


def consumer_main(q_in, q_out, device):
    torch.cuda.set_device(device)
    stream = torch.cuda.Stream()
    held = []
    while True:
        item = q_in.get()
        if item is None:
            break
        ev = torch.cuda.Event.from_ipc_handle(device, item)
        stream.wait_event(ev)
        stream.synchronize()
        held.append(ev)  # keep imported events alive; the producer destroys its side
        q_out.put(True)


def pct(xs, p):
    if not xs:
        return float("nan")
    xs = sorted(xs)
    k = min(len(xs) - 1, max(0, int(round(p / 100.0 * (len(xs) - 1)))))
    return xs[k]


class ForwardThread(threading.Thread):
    """Launch tiny kernels and record events; time every call."""

    def __init__(self, device):
        super().__init__(daemon=True)
        self.device = device
        self.stop = threading.Event()
        self.samples = []  # (t_start_ns, dur_ns, kind)
        self.do_destroy = None  # callable run on this thread when set
        self.destroy_lock = threading.Lock()
        self.keep = []

    def run(self):
        torch.cuda.set_device(self.device)
        s = torch.cuda.Stream()
        x = torch.zeros(1024, device=self.device)
        ev_plain = torch.cuda.Event()
        ev_ipc = torch.cuda.Event(interprocess=True)
        n = 0
        with torch.cuda.stream(s):
            while not self.stop.is_set():
                with self.destroy_lock:
                    job, self.do_destroy = self.do_destroy, None
                if job is not None:
                    job()
                t0 = time.perf_counter_ns()
                x.add_(1.0)
                t1 = time.perf_counter_ns()
                self.samples.append((t0, t1 - t0, "launch"))
                n += 1
                if n % 8 == 0:
                    t0 = time.perf_counter_ns()
                    ev_plain.record(s)
                    t1 = time.perf_counter_ns()
                    self.samples.append((t0, t1 - t0, "record_plain"))
                if n % 8 == 4:
                    t0 = time.perf_counter_ns()
                    ev_ipc.record(s)
                    t1 = time.perf_counter_ns()
                    self.samples.append((t0, t1 - t0, "record_ipc"))
                if n % 8 == 2:
                    t0 = time.perf_counter_ns()
                    ev_ipc.query()
                    t1 = time.perf_counter_ns()
                    self.samples.append((t0, t1 - t0, "query_ipc"))
                if n % 32 == 6:
                    # what an engine step (ATOM start_load_kv) or a stock server
                    # handler (completion event) does per operation: a fresh IPC
                    # event, recorded and exported
                    t0 = time.perf_counter_ns()
                    fresh = torch.cuda.Event(interprocess=True)
                    fresh.record(
                        s
                    )  # torch creates the HIP event lazily, on first record
                    t1 = time.perf_counter_ns()
                    fresh.ipc_handle()
                    t2 = time.perf_counter_ns()
                    self.samples.append((t0, t1 - t0, "create_record_ipc"))
                    self.samples.append((t1, t2 - t1, "export_ipc"))
                    self.samples.append((t0, t2 - t0, "create_export_ipc"))
                    self.keep.append(fresh)  # destroyed later, outside the windows
                # keep the queue shallow, like a launch-bound step thread
                if n % 64 == 0:
                    s.synchronize()
                time.sleep(0.00002)


def make_done_event(kind, device, consumer):
    """An event recorded and completed long before it is destroyed."""
    ev = torch.cuda.Event(interprocess=(kind == "ipc"))
    side = torch.cuda.Stream()
    ev.record(side)
    if kind == "ipc":
        handle = ev.ipc_handle()
        if consumer is not None:
            q_in, q_out = consumer
            q_in.put(handle)
            q_out.get(timeout=30)
    ev.synchronize()
    return ev


def calibrate_sleep_cycles(busy_ms):
    s = torch.cuda.Stream()
    cycles = 10_000_000
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for _ in range(3):
        with torch.cuda.stream(s):
            a.record(s)
            torch.cuda._sleep(cycles)
            b.record(s)
        b.synchronize()
        ms = a.elapsed_time(b)
        cycles = int(cycles * busy_ms / max(ms, 1e-3))
    return cycles


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=0)
    ap.add_argument("--trials", type=int, default=30)
    ap.add_argument("--busy-ms", type=float, default=40.0)
    ap.add_argument("--consumer", type=int, default=1)
    ap.add_argument("--arms", default="bg_none,bg_plain,bg_ipc,fwd_busy,fwd_idle")
    ap.add_argument("--out", default="result.json")
    args = ap.parse_args()

    torch.cuda.set_device(args.device)
    torch.zeros(1, device=args.device)
    consumer = None
    proc = None
    if args.consumer:
        ctx = mp.get_context("spawn")
        q_in, q_out = ctx.Queue(), ctx.Queue()
        proc = ctx.Process(
            target=consumer_main, args=(q_in, q_out, args.device), daemon=True
        )
        proc.start()
        consumer = (q_in, q_out)

    cycles = calibrate_sleep_cycles(args.busy_ms)
    busy_stream = torch.cuda.Stream()

    fwd = ForwardThread(args.device)
    fwd.start()
    time.sleep(1.0)  # warm up

    results = {
        "env": {
            "torch": torch.__version__,
            "hip": getattr(torch.version, "hip", None),
            "device": torch.cuda.get_device_name(args.device),
            "busy_ms": args.busy_ms,
            "sleep_cycles": cycles,
            "trials": args.trials,
            "consumer": bool(args.consumer),
        },
        "arms": {},
    }

    for arm in args.arms.split(","):
        windows = []  # (t0_ns, t1_ns) of each destroy / baseline window
        destroy_ms = []
        for _ in range(args.trials):
            kind = "plain" if arm == "bg_plain" else "ipc"
            ev = (
                None
                if arm == "bg_none"
                else make_done_event(kind, args.device, consumer)
            )
            busy_done = torch.cuda.Event()
            with torch.cuda.stream(busy_stream):
                torch.cuda._sleep(cycles)
                busy_done.record(busy_stream)

            if arm in ("bg_ipc", "bg_plain"):
                t0 = time.perf_counter_ns()
                del ev
                t1 = time.perf_counter_ns()
            elif arm == "bg_none":
                t0 = time.perf_counter_ns()
                time.sleep(args.busy_ms / 1000.0)
                t1 = time.perf_counter_ns()
            else:
                if arm == "fwd_idle":
                    busy_done.synchronize()
                box = {}
                done = threading.Event()
                holder = [ev]
                del ev

                def job(holder=holder, box=box, done=done):
                    a = time.perf_counter_ns()
                    holder.clear()  # drops the last reference on the forward thread
                    box["t"] = (a, time.perf_counter_ns())
                    done.set()

                with fwd.destroy_lock:
                    fwd.do_destroy = job
                done.wait(10)
                t0, t1 = box["t"]
            windows.append((t0, t1))
            destroy_ms.append((t1 - t0) / 1e6)
            busy_done.synchronize()
            time.sleep(0.03)

        # forward-thread call latencies inside vs outside the windows
        inside = {
            "launch": [],
            "record_plain": [],
            "record_ipc": [],
            "query_ipc": [],
            "create_record_ipc": [],
            "export_ipc": [],
            "create_export_ipc": [],
        }
        lo, hi = windows[0][0], windows[-1][1]
        for t, d, k in fwd.samples:
            if t < lo or t > hi:
                continue
            if any(a <= t <= b for a, b in windows):
                inside[k].append(d / 1e3)
        summary = {
            "destroy_ms": {
                "p50": statistics.median(destroy_ms),
                "max": max(destroy_ms),
                "min": min(destroy_ms),
            },
            "fwd_calls_in_window_us": {
                k: {
                    "n": len(v),
                    "p50": pct(v, 50),
                    "p99": pct(v, 99),
                    "max": max(v) if v else None,
                }
                for k, v in inside.items()
            },
        }
        results["arms"][arm] = summary
        print(arm, json.dumps(summary), flush=True)

    fwd.stop.set()
    fwd.join(5)
    if proc is not None:
        consumer[0].put(None)
        proc.join(10)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
