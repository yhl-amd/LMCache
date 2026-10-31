# Pooled IPC events for LMCache MP (`lmcache_driven`): design and reproduction

中文版：[README.zh.md](README.zh.md)

**TL;DR.** In `lmcache_driven` mode every store/retrieve creates, imports, and destroys
interprocess (IPC) events. On ROCm 7.2.x, destroying an IPC event waits until all of
the process's work on that GPU drains (`~IPCEvent` → `ihipHostUnregister` →
`SyncAllStreams`). While it waits, other threads that create an IPC event also stall.
So per-operation event churn turns a ~20 ms KV copy into ~20 ms of blocked CPU time in
the server, once per operation. The opt-in pooled-events path
(`LMCACHE_MP_POOLED_EVENTS=1`) records a fixed set of events over and over and imports
each handle only once, so no imported event is destroyed per operation. Everything here
is reproducible on one MI300/MI350-class node with the scripts in this directory.

- [1. Background: two IPC events per operation](#1-background-two-ipc-events-per-operation)
- [2. The problem: destroying an IPC event on ROCm 7.2.x](#2-the-problem-destroying-an-ipc-event-on-rocm-72x)
- [3. Where it hits LMCache](#3-where-it-hits-lmcache)
- [4. Design: pooled events](#4-design-pooled-events)
- [5. Why reuse is safe](#5-why-reuse-is-safe)
- [6. Reproduce](#6-reproduce)
- [7. Results](#7-results)
- [8. Limitations and open questions](#8-limitations-and-open-questions)

## 1. Background: two IPC events per operation

In `lmcache_driven` mode the engine registers its KV cache with the server as IPC
memory handles, and the server copies KV directly from or into engine memory on its own
copy stream. The two processes cannot see each other's GPU queues, so they order work
with two interprocess events. Neither CPU waits.

| | ① producer event (engine → server) | ② completion event (server → engine) |
|---|---|---|
| created by | engine, per step (ATOM: `start_load_kv`) | server, per store/retrieve (`create_event()`) |
| recorded on | engine's forward stream, after the work that wrote the KV | server's copy stream, after the copy |
| used for | server copy stream `wait`s on it before copying | engine `query()`s it every step (non-blocking); when done, store blocks can be freed / retrieved requests can run |
| stock lifetime | server imports it per request, destroys it after use | server exports it and keeps it in a 2048-entry ring; engine imports it per request, destroys it when the future is released |

One store, on a timeline:

```
engine CPU     [create+record ①]──request(① handle, blocks, key)──┐            ┌─reply(② handle)─> [import ②, query each step] ... ② done → free blocks
engine GPU     [ forward writing KV ]①                            │            │
                                    ╎ (GPU-side wait)             ▼            │
server thread                       ╎                [import ①, create ②, enqueue wait+copy, record ②] → reply
server GPU                          ╎ [wait ①]·······[ copy KV (~20 ms for 383 MB) ]②
```

## 2. The problem: destroying an IPC event on ROCm 7.2.x

`hipEventDestroy` on a recorded IPC event goes through `~IPCEvent()` →
`ihipHostUnregister` → `SyncAllStreams()`. So it waits for **all** work this process has
queued on the device, not just the event's own work:
[rocm-systems#7520](https://github.com/ROCm/rocm-systems/issues/7520) (open). develop
fixed it in [#7710](https://github.com/ROCm/rocm-systems/pull/7710) (deferred
per-device cleanup). The 7.2 backport,
[#11529](https://github.com/ROCm/rocm-systems/pull/11529), is **not merged**, so ROCm
7.2.4 still has the behaviour.

Measured with `destroy_probe.py` (MI355X, ROCm 7.2.4, torch 2.10.0+rocm7.2.4, one GPU,
30 trials, a 40 ms kernel queued on another stream before each destroy):

| destroy | p50 |
|---|---|
| plain event, background thread, GPU busy | 0.0008 ms |
| **IPC event, background thread, GPU busy** | **40.8 ms** |
| IPC event, same thread, GPU busy | 40.9 ms |
| IPC event, same thread, after the busy work finished | 1.25 ms |

HIP calls made by **another** thread while the destroy waits:

| call on the other thread | normal p50 | during the destroy p50 |
|---|---|---|
| kernel launch | 3.8 µs | 3.5 µs |
| `record` (plain / IPC event) | 8.4 / 8.6 µs | 7.1 / 5.7 µs |
| `query` | 0.3 µs | 0.3 µs |
| **create a new IPC event (on first record)** | **43 µs** | **38.6 ms** |

So a destroy blocks the destroying thread for the device's queued work, and it also
blocks **any thread that creates an IPC event** for the same time. Launches, records,
and queries keep going.

`event_reuse_check.py` (two processes, one GPU) gives the per-call costs on the
importing side: `import` 0.026 ms, `wait` 0.003 ms, destroying the imported event
1.3 ms with the GPU idle and **50.9 ms** when the importer's own GPU work is a 50 ms
kernel. The importer's own work is what counts: in the server, that is its copy.

## 3. Where it hits LMCache

Per store/retrieve, the stock path creates one IPC event (②) and imports one (①) in
the server, and imports one (②) in the engine. Each of these is later destroyed.

- **f22dec28 and earlier (still what the ATOM image ships: `rocm/atom-dev:latest` built
  2026-10-28):** `store()` / `retrieve()` drop the imported ①
  when they return. The destroy blocks the handler thread until this copy (and every
  other copy the server queued on that GPU) finishes, and the reply carrying ② is only
  sent after that. The handler is the per-client affinity thread, so the next request
  waits behind it. With `--max-gpu-workers 1` (the default) all ranks share one
  thread, so the copies of four GPUs run one after another.
- **dev after [#5439](https://github.com/LMCache/LMCache/pull/5439) ("Hold imported
  IPC events"):** the import is held and released by a stream-ordered host callback on
  the device host-func dispatcher thread, so the handler no longer destroys it inline.
  The import and the destroy still happen once per operation. They now block the
  dispatcher thread, which also delivers L1 write/read completions. And while that
  destroy waits, the handler's next `create_event()` for ② stalls (section 2, last
  table). Section 7 measures how much of the stall is left.
- **Completion ring:** once 2048 completion events have been exported, each new one
  pushes out and destroys the oldest, which is again an IPC event destroy on the server.
- **Engine side:** the engine imports ② per operation and destroys it when the future
  is released. Engines that create and destroy their own ① per step (ATOM does) pay the
  same price on their side; that part is outside LMCache.

## 4. Design: pooled events

Opt-in: the client sets `LMCACHE_MP_POOLED_EVENTS=1`. The key carries a transport hint
`pooled_events` (`compare=False`, so it is not part of cache identity). The server
picks the pooled path per request; old clients and old servers are unaffected.

| | stock | pooled |
|---|---|---|
| ① producer event | engine's per-step event is sent | client adapter records one of **16 rotating events** at the same stream position, right after the engine's event; each event is replaced after **31 records** |
| server import of ① | import per request, destroy afterwards | **import once per handle**, cached per instance (LRU 4096); `wait` on the cached import |
| ② completion event | new event per request; 2048 ring, oldest destroyed | **512 events per instance**, re-recorded round-robin, never destroyed while the instance is registered |
| engine import of ② | import per request, destroy on release | handles carry the prefix `LMCPOOL1`; **imported once**, cached (LRU 16384) |

Per operation, stock does 1 create + 2 imports + 3 destroys of IPC events across the two
processes. Pooled does none in steady state: one new ① every 31 operations per slot, and
cache evictions after hours.

Code (dev port, 4 files, +196 lines):
`lmcache/integration/atom/multi_process_adapter.py` (client pool),
`lmcache/v1/multiprocess/modules/lmcache_driven_transfer.py` (server import cache,
completion pool), `lmcache/v1/platform/base/event_ipc.py` (client import cache for
pooled handles), `lmcache/v1/multiprocess/custom_types.py` (`pooled_events`,
`POOLED_EVENT_MAGIC`, `IPC_EVENT_MAX_RECORDS`).

## 5. Why reuse is safe

- **Re-recording is conservative.** `record` moves an event to a new stream position,
  and `wait`/`query` always observe the latest record. If an event is re-recorded
  before the other side waits on or queries the older use, the other side sees a
  *later* position on the *same* stream. It waits longer, but it never proceeds early.
  Pool sizes (16 producer slots per rank, 512 completion slots per instance) make this
  rare.
- **The 32-record limit.** On ROCm 7.2.x, `IPCEventEmulated::streamWait` indexes
  `signal[read_index]` without `% 32`. An imported event whose owner has recorded it
  more than 32 times fails `wait` with `hipErrorInvalidValue`
  (develop fix [#12230](https://github.com/ROCm/rocm-systems/pull/12230);
  [#11749](https://github.com/ROCm/rocm-systems/pull/11749) and
  [#11511](https://github.com/ROCm/rocm-systems/pull/11511) are open). Producer events
  are therefore retired after 31 records. Completion events are only `query`'d, and
  query does wrap, so they are never retired.
- **Lifetime.** Cached imports and pooled completion events live as long as the
  instance's context entry. Imports leave the cache only by LRU eviction: 4096 producer
  handles, at one new handle per 31 operations per slot, cover hours. So an eviction
  never hits an event with a pending wait in practice. #5439's hold/release token is
  kept on the pooled path as well.
- **Checked:** `event_reuse_check.py` runs both schemes across two processes for 400
  rounds: zero completions seen before the owner's kernel finished, and zero stale
  "complete" reports right after a re-record. With `rot=40` (more than 32 records) the
  peer's `wait` fails at the 33rd record, which reproduces the ROCm bug.

## 6. Reproduce

Hardware: one node with 4 AMD Instinct GPUs (MI300X/MI325X/MI350X/MI355X), ROCm 7.2.x.
Our runs: 8×MI355X, image `rocm/atom-dev:latest` (2026-10-28, ROCm 7.2.4,
torch 2.10.0+rocm7.2.4, LMCache 0.5.6.dev139+gf22dec28), GPUs 4–7.

**Level 1: one GPU, torch only (~2 min).**

```bash
python destroy_probe.py --trials 30 --busy-ms 40 --consumer 1 --out probe.json
```

**Level 2: two processes, one GPU, torch only (~3 min).**

```bash
python event_reuse_check.py 400 31   # pooled schemes: expect 0 unsafe, 0 stale
python event_reuse_check.py 60 40    # >32 records: peer wait fails at record 33 (ROCm 7.2.x bug)
```

**Level 3: LMCache MP server + 4 clients, no model (~4 min per arm).**
`srvbench.py` starts 4 rank processes that register MiniMax-M3-shaped KV (301 tensors,
2,995,200 B per 128-token block per rank) through `AtomMPWorkerAdapter`, plus a
scheduler that issues lookup → retrieve → store per turn, the way ATOM does. `sweep`
runs one conversation with prompts of 16k…256k tokens (per-op cost without
contention). `burst` makes 16×64k and 32×32k conversations arrive at once, 3
repetitions each (638 GB written to L1 in total).

```bash
# build each LMCache version into its own directory, e.g.
pip install --no-deps --no-build-isolation --target /w/site-dev /path/to/LMCache   # needs grpcio-tools
export HIP_VISIBLE_DEVICES=4,5,6,7 SB_T_SWEEP=16 SB_BURSTS=16x65536,32x32768 SB_BURST_R=3 SB_L1=800 SB_LAZY=0

PYTHONPATH=/w/site-dev      bash run_srvbench.sh runs/dev-stock@r1 0
PYTHONPATH=/w/site-pool-dev bash run_srvbench.sh runs/dev-pool@r1  1
PYTHONPATH=/w/site-dev      bash run_srvbench.sh runs/dev-stock-w4@r1 0 --max-gpu-workers 4
python aggregate.py runs dev-stock dev-pool dev-stock-w4
python sweep_curves.py runs dev-stock dev-pool dev-stock-w4   # per-prompt-length curves (pass1, mean of rounds)
```

`run_srvbench.sh` starts the server (`--chunk-size 256 --supported-transfer-mode
lmcache_driven --eviction-ratio 0.01`, L1 size from `SB_L1`, plus any extra args), runs
both phases, and writes `summary.json`. Latencies are measured on the client: from
submitting an operation on all 4 ranks until every rank's completion event reports
done. **Give L1 enough headroom that nothing is evicted, and allocate it up front
(`SB_LAZY=0`).** With a 400 GB L1, or with `--l1-use-lazy` while L1 grows, the burst
phase evicts and even fails allocations, and memory pressure, not events, decides the
latency. Check `server.log` for `triggering eviction` and compare the number of
retrieves in `summary.json` (144 per run when nothing was evicted).

## 7. Results

All numbers: MI355X ×4 (GPUs 4–7 of an 8-GPU node), ROCm 7.2.4, `SB_T_SWEEP=16
SB_BURSTS=16x65536,32x32768 SB_BURST_R=3`, L1 800 GB allocated up front (`SB_LAZY=0`),
**3 runs per arm**, mean (min–max). Only the event pool is switched on in the `pool`
arms (no client hashing, no bulk keys). `f22` = LMCache f22dec28 (what the ATOM image
ships). `dev` = LMCache dev 4995e604 (after #5439). `-w4` = `--max-gpu-workers 4`.

**Single conversation (sweep, no contention):** all arms are clean (no eviction).

| client-observed latency, p50 | f22 stock | f22 pool | dev stock | dev pool | f22 stock -w4 | dev stock -w4 | dev pool -w4 |
|---|---|---|---|---|---|---|---|
| store, 16k new tokens × 4 ranks (ms) | 164 (157–169) | **97** (95–100) | 180 (148–242) | **93** (84–103) | 124 (115–129) | 106 (102–112) | **52** (46–58) |
| retrieve, up to 240k tokens × 4 ranks (ms) | 386 (382–391) | **167** (161–174) | 394 (369–444) | **165** (157–174) | 191 (187–198) | 169 (157–177) | **96** (90–100) |

**Many conversations at once (burst):**

| | f22 stock | f22 pool | dev stock | dev pool ¹ | f22 stock -w4 | dev stock -w4 |
|---|---|---|---|---|---|---|
| 16 × 64k arrive together: makespan (s) | 6.53 (5.28–7.41) | **2.45** (2.40–2.48) | 5.85 (5.49–6.45) | **2.44** (2.22–2.57) | 3.32 (3.29–3.34) | 3.19 (3.02–3.34) |
| 32 × 32k arrive together: makespan (s) | 8.92 (6.38–10.52) | **3.48** (3.37–3.55) | 7.35 (6.27–9.49) | **3.86** (3.28–4.47) | 4.91 (4.62–5.48) | 5.05 (4.78–5.47) |
| store latency p50 / p90 (ms) | 3909 / 4903 | **1447 / 1715** | 2924 / 4280 | **1523 / 2136** | 2214 / 2504 | 2098 / 2694 |
| retrieve latency p50 / p90 (ms) | 2055 / 3839 | **804 / 1418** | 1856 / 3278 | **751 / 1325** | 1015 / 1868 | 1001 / 1825 |

¹ In one of the three `dev pool` runs, L1 hit the eviction watermark during the last
32×32k bursts, and 116 of 144 turns found their prefix and retrieved it (all other
listed runs: 142–144 of 144). The `dev pool -w4` burst numbers are left out: L1 evicted
in all three runs (77–101 of 144 retrieves), so those runs did less work than the
others and are not comparable. Its sweep numbers above are clean.

**What this says.**

- **#5439 alone does not remove the stall.** dev stock is within noise of f22 stock in
  both phases. The destroy moved off the handler thread, but every operation still
  creates, imports, and destroys IPC events.
- **The pool halves single-request latency and cuts burst makespan by 1.9–2.7×** on
  both versions: store −41…−48 %, retrieve −57…−58 %, burst store/retrieve p90 −50…−65 %.
- **More GPU workers help, but less than the pool.** `--max-gpu-workers 4` lets each
  rank's destroy block only its own thread, so the four GPUs stop serializing. dev stock
  -w4 bursts take 3.19 / 5.05 s, against 2.44 / 3.86 s for the pool on a single worker.
  The two combine: dev pool -w4 has the lowest sweep latency (52 / 96 ms).
- **Faster arms reach the L1 watermark first.** At equal bytes written, stock and pool
  use the same L1 memory (485 vs 484 GB after 80k chunks: no leak). The pool keeps more
  writes in flight, though, so its usage ratio peaks higher. Size L1 with headroom
  (we used 800 GB for 638 GB of data; 1200 GB would be safer for the -w4 pool arm).

## 7b. Persistent kernels (e.g. ATOM mono) and the device-side wait

On ROCm 7.2 the server's `stream.wait_event` on an imported IPC event is a
one-work-item kernel (`streamOpsWait`, rocclr default `GPU_STREAMOPS_CP_WAIT=false`)
that polls the event's signal until the client's GPU work writes it. ATOM's mono
decode kernels need all their CTAs co-resident (one per CU) and spin on each
other. If the polling kernel is resident when such a kernel is dispatched and
takes the slot one of its CTAs needs, the persistent kernel waits for the CU and
the polling kernel waits for a signal written after the persistent kernel: both
hang. Reproduced with `mono_hang_repro.py` on one MI355X (ROCm 7.2.4):

| persistent kernel fills | `--mode stream` (stock wait) | `--mode host` | `--mode gate` |
|---|---|---|---|
| every CU (2 × 1024-thread CTAs per CU) | **hangs at iteration 2** (2 of 2 runs) | 100 iterations OK | 100 iterations OK |
| half of each CU (1 CTA per CU) | 100 iterations OK | – | – |

Two opt-in switches remove the device-side wait on unfinished work:

- `LMCACHE_MP_SUBMIT_AFTER_PRODUCER=1` (client): the ATOM adapter holds each
  store/retrieve on a host thread until its producer event has fired, then sends
  it. `submit_*` returns a future that reports not-done until then; the engine's
  calling thread spends 0.75 ms instead of 4.4 ms per store submit.
- `LMCACHE_MP_HOST_WAIT_PRODUCER=1` (server): check the producer event with
  `query()` (host memory only); skip the stream wait when it has fired, otherwise
  poll on the host instead of enqueueing a device-side wait.

With both on, `srvbench` (pooled events, `--max-gpu-workers 4`, 2 runs each, no
L1 expansion) is unchanged: single conversation store/retrieve p50 33/84 ms vs
33/85 ms, bursts 0.90/1.20 s vs 0.85/1.32 s, 0 failed ops. ROCm develop
([#5575](https://github.com/ROCm/rocm-systems/pull/5575)) replaces emulated IPC
events with ROCr IPC signals waited on by the CP, which removes the polling kernel.

**Benchmark note.** Lazy L1 growth pins 10 GB chunks with `hipHostRegister`, and
while a chunk is pinned the server's IPC event records and IPC handle opens stall
for seconds (seen as ~10 s burst stalls, registration timeouts and reaped
instances). Set `SB_L1` equal to `SB_L1_INIT` so L1 never grows during a run.

## 8. Limitations and open questions

- **ROCm-specific.** On CUDA, `cudaEventDestroy` does not synchronize, so there the pool
  saves only the import/create calls. We have not measured CUDA. Once the ROCm 7.2
  backport (#11529) or a release with #7710 ships, the destroy cost goes away, and the
  pool becomes a smaller optimization (fewer imports).
- **Synthetic load.** `srvbench` has no model forward, so the GPU is busy only with
  copies. In a real engine the forward keeps the GPU busier, which makes each destroy
  longer: we measured p50 8–10 ms and max 2.4 s for IPC event destroys in a live
  MiniMax-M3 TP4 c48 service.
- **Engine-side events.** The pool covers the events LMCache creates or imports. An
  engine that creates and destroys its own interprocess event every step still pays the
  destroy itself.
- **Relation to `--max-gpu-workers`.** More GPU workers let different ranks' requests
  run on different server threads. They do not remove the destroy, which still blocks
  whichever thread performs it. The `-w4` arms in section 7 measure the combination.
