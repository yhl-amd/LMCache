"""Can a stream wait on an imported IPC event hang an all-CU persistent kernel?

ATOM's mono decode runs persistent kernels whose CTAs fill every CU and wait for
each other, so all of them must be resident at once. On ROCm 7.2 a stream wait on
an imported IPC event is a one-work-item kernel that polls the event's signal.
If that polling kernel is resident when the persistent kernel is dispatched and
the persistent kernel then cannot place all its CTAs, the persistent kernel waits
for the CU and the polling kernel waits for the signal written after the
persistent kernel: neither finishes.

Two processes on one GPU, like an engine rank and the LMCache server:

  engine: per iteration, keep its stream busy (sleep kernel), queue the persistent
          kernel, record an interprocess event, and send the handle at once
          (--mode gate: send it only after event.query() is True)
  server: import the handle (cached per handle) and order a small copy after it
          (--mode stream: stream.wait_event, the device-side wait;
           --mode host / gate: poll event.query() on the host first)

The persistent kernel's grid is (blocks per CU from the occupancy API) x CUs, so
it is exactly co-resident; --block sets how full each CU is. An iteration that
does not finish within --timeout seconds is reported as a hang and the run stops.

usage: python mono_hang_repro.py --mode stream|host|gate
       [--block 1024] [--blocks-per-cu N]
"""

import argparse
import multiprocessing as mp
import os
import sys
import threading
import time

SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

__global__ void persist(unsigned int* bar, unsigned int nblk, int rounds) {
  for (int r = 0; r < rounds; ++r) {
    __syncthreads();
    if (threadIdx.x == 0) {
      __threadfence();
      atomicAdd(bar, 1u);
      unsigned int target = nblk * (unsigned int)(r + 1);
      while (atomicAdd(bar, 0u) < target) { __builtin_amdgcn_s_sleep(2); }
    }
    __syncthreads();
  }
}

int64_t blocks_per_cu(int64_t block) {
  int n = 0;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, persist, (int)block, 0);
  return n;
}

void launch(torch::Tensor bar, int64_t grid, int64_t block, int64_t rounds) {
  auto stream = at::cuda::getCurrentCUDAStream();
  cudaMemsetAsync(bar.data_ptr(), 0, sizeof(unsigned int), stream);
  persist<<<(unsigned int)grid, (unsigned int)block, 0, stream>>>(
      (unsigned int*)bar.data_ptr(), (unsigned int)grid, (int)rounds);
}
"""


def build():
    from torch.utils.cpp_extension import load_inline

    return load_inline(
        name="mono_hang_persist",
        cpp_sources="int64_t blocks_per_cu(int64_t block);\n"
        "void launch(torch::Tensor bar, int64_t grid, int64_t block, int64_t rounds);",
        cuda_sources=SRC,
        functions=["blocks_per_cu", "launch"],
        verbose=False,
    )


def server(conn, mode):
    import torch

    torch.cuda.set_device(0)
    stream = torch.cuda.Stream()
    src = torch.ones(1 << 20, device="cuda")
    dst = torch.empty_like(src)
    imported = {}
    conn.send("ready")
    while True:
        msg = conn.recv()
        if msg is None:
            break
        ev = imported.get(msg)
        if ev is None:
            ev = torch.cuda.Event.from_ipc_handle(0, msg)
            imported[msg] = ev
        if mode == "stream":
            stream.wait_event(ev)
        else:
            while not ev.query():
                time.sleep(50e-6)
        with torch.cuda.stream(stream):
            dst.copy_(src)
        stream.synchronize()
        conn.send("ok")


def engine(args, conn):
    import torch

    torch.cuda.set_device(0)
    ext = build()
    props = torch.cuda.get_device_properties(0)
    per_cu = ext.blocks_per_cu(args.block)
    if args.blocks_per_cu:  # fewer than fit: leave room on every CU
        per_cu = min(per_cu, args.blocks_per_cu)
    grid = per_cu * props.multi_processor_count
    print(
        f"{props.name}: {props.multi_processor_count} CUs, block {args.block} -> "
        f"{per_cu} blocks/CU, grid {grid}, mode {args.mode}",
        flush=True,
    )
    bar = torch.zeros(1, dtype=torch.int32, device="cuda")
    # warm up: the persistent kernel alone must finish (all CTAs co-resident)
    ext.launch(bar, grid, args.block, args.rounds)
    torch.cuda.synchronize()
    assert conn.recv() == "ready"

    pool = [None] * 16
    records = [0] * 16
    hang = threading.Event()
    for it in range(args.iters):
        t0 = time.time()
        slot = it % 16
        if pool[slot] is None or records[slot] >= 31:  # ROCm 7.2: <= 32 records
            pool[slot] = torch.cuda.Event(interprocess=True)
            records[slot] = 0
        ev = pool[slot]
        torch.cuda._sleep(args.sleep_cycles)  # stream busy: dispatch later
        ext.launch(bar, grid, args.block, args.rounds)  # the "mono" kernel
        ev.record()
        records[slot] += 1
        if args.mode == "gate":
            while not ev.query():
                time.sleep(50e-6)
        conn.send(ev.ipc_handle())

        done = threading.Event()

        def wait_all(done=done):
            torch.cuda.synchronize()
            conn.recv()
            done.set()

        th = threading.Thread(target=wait_all, daemon=True)
        th.start()
        if not done.wait(args.timeout):
            print(
                f"HANG at iteration {it}: not finished after {args.timeout} s",
                flush=True,
            )
            hang.set()
            break
        if it % 50 == 0:
            print(f"iter {it}: {1e3 * (time.time() - t0):.1f} ms", flush=True)
    if not hang.is_set():
        print(f"OK: {args.iters} iterations, no hang", flush=True)
        conn.send(None)
    os._exit(2 if hang.is_set() else 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["stream", "host", "gate"], required=True)
    ap.add_argument("--block", type=int, default=1024)
    ap.add_argument("--rounds", type=int, default=200)
    ap.add_argument("--blocks-per-cu", type=int, default=0, help="0 = as many as fit")
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--sleep-cycles", type=int, default=20_000_000)
    ap.add_argument("--timeout", type=float, default=20.0)
    args = ap.parse_args()
    ctx = mp.get_context("spawn")
    a, b = ctx.Pipe()
    s = ctx.Process(
        target=server,
        args=(b, "stream" if args.mode == "stream" else "host"),
        daemon=True,
    )
    s.start()
    e = ctx.Process(target=engine, args=(args, a))
    e.start()
    e.join()
    if s.is_alive():
        s.kill()
    sys.exit(e.exitcode)


if __name__ == "__main__":
    main()
