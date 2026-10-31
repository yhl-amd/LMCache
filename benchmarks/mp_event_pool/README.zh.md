# LMCache MP（`lmcache_driven`）事件池：设计与复现

English: [README.md](README.md)

**一句话**：在 `lmcache_driven` 模式下，每次 store/retrieve 都要新建、导入、销毁跨进程（IPC）event。ROCm 7.2.x 上，销毁一个 IPC event 要等本进程在这张卡上排着的 GPU 工作全部做完（`~IPCEvent` → `ihipHostUnregister` → `SyncAllStreams`）；等的这段时间里，别的线程新建 IPC event 也会被卡住。所以每次操作都会把一次约 20 ms 的 KV 拷贝，变成 server 上约 20 ms 被卡住的 CPU 时间。事件池（`LMCACHE_MP_POOLED_EVENTS=1`，默认关）让一组固定的 event 反复 record，每个 handle 只导入一次，于是每次操作不再销毁任何导入的 event。本文所有结论都可以用本目录下的脚本，在一台 MI300/MI350 级别的机器上复现。

- [1. 背景：每次操作里的两个 IPC event](#1-背景每次操作里的两个-ipc-event)
- [2. 问题：ROCm 7.2.x 上销毁 IPC event 会阻塞](#2-问题rocm-72x-上销毁-ipc-event-会阻塞)
- [3. 在 LMCache 里卡在哪](#3-在-lmcache-里卡在哪)
- [4. 设计：事件池](#4-设计事件池)
- [5. 复用为什么安全](#5-复用为什么安全)
- [6. 复现步骤](#6-复现步骤)
- [7. 结果](#7-结果)
- [8. 局限与待确认的问题](#8-局限与待确认的问题)

## 1. 背景：每次操作里的两个 IPC event

`lmcache_driven` 模式下，引擎把 KV cache 显存以 IPC handle 注册给 server，之后由 server 在自己的拷贝 stream 上直接读写引擎的显存。两个进程的 GPU 队列互相看不见，所以用两个跨进程 event 在 GPU 上排先后，两边的 CPU 都不用等。

| | ① 生产者 event（引擎 → server） | ② 完成 event（server → 引擎） |
|---|---|---|
| 谁创建 | 引擎，每步一个（ATOM：`start_load_kv`） | server，每次 store/retrieve 一个（`create_event()`） |
| record 在哪 | 引擎的 forward stream，排在写这些 KV 的工作之后 | server 的拷贝 stream，排在这次拷贝之后 |
| 用途 | server 的拷贝 stream 先 `wait` 它再拷 | 引擎每步 `query()` 它（不阻塞）；到了之后，store 借的 block 可以释放，retrieve 的请求可以开跑 |
| 原版的生命周期 | server 每个请求导入一次，用完销毁 | server 导出后放进 2048 的环形队列保活；引擎每个请求导入一次，future 释放时销毁 |

一次 store 的时间线：

```
引擎 CPU      [新建并 record ①]──请求(① 的 handle、block、key)──┐            ┌─回复(② 的 handle)─> [导入 ②，每步 query] ... ② 到了 → 释放 block
引擎 GPU      [ forward 写 KV ]①                                │            │
                               ╎（在 GPU 上等）                  ▼            │
server 线程                    ╎                  [导入 ①、新建 ②、排 wait+拷贝、record ②] → 回复
server GPU                     ╎ [等 ①]···········[ 拷贝 KV（383 MB 约 20 ms）]②
```

## 2. 问题：ROCm 7.2.x 上销毁 IPC event 会阻塞

对一个 record 过的 IPC event 调 `hipEventDestroy`，会走 `~IPCEvent()` → `ihipHostUnregister` → `SyncAllStreams()`。所以它等的是本进程在这张卡上排着的**全部**工作，不只是这个 event 自己：[rocm-systems#7520](https://github.com/ROCm/rocm-systems/issues/7520)（open）。develop 已由 [#7710](https://github.com/ROCm/rocm-systems/pull/7710) 修复（按设备延迟清理）；给 7.2 的 backport [#11529](https://github.com/ROCm/rocm-systems/pull/11529) **还没合**，所以 ROCm 7.2.4 仍然是这个行为。

用 `destroy_probe.py` 实测（MI355X，ROCm 7.2.4，torch 2.10.0+rocm7.2.4，单卡，30 次；每次销毁前在另一条 stream 上排一个 40 ms 的 kernel）：

| 销毁 | p50 |
|---|---|
| 普通 event，后台线程，GPU 忙 | 0.0008 ms |
| **IPC event，后台线程，GPU 忙** | **40.8 ms** |
| IPC event，同一线程，GPU 忙 | 40.9 ms |
| IPC event，同一线程，忙的工作做完以后 | 1.25 ms |

销毁等待期间，**另一个线程**上的 HIP 调用：

| 另一个线程上的调用 | 平时 p50 | 销毁期间 p50 |
|---|---|---|
| kernel launch | 3.8 µs | 3.5 µs |
| `record`（普通 / IPC event） | 8.4 / 8.6 µs | 7.1 / 5.7 µs |
| `query` | 0.3 µs | 0.3 µs |
| **新建 IPC event（第一次 record 时创建）** | **43 µs** | **38.6 ms** |

也就是说，销毁会让执行销毁的线程等这张卡上排着的工作做完，同时让**任何新建 IPC event 的线程**也等同样长的时间；launch、record、query 不受影响。

`event_reuse_check.py`（两个进程、一张卡）测了导入方的单次开销：导入 0.026 ms，wait 0.003 ms；销毁导入的 event 在 GPU 空闲时 1.3 ms，导入方自己的 GPU 上有一个 50 ms 的 kernel 时要 **50.9 ms**。起作用的是导入方自己的工作，在 server 里就是它的拷贝。

## 3. 在 LMCache 里卡在哪

原版每次 store/retrieve：server 新建一个 IPC event（②）、导入一个（①），引擎导入一个（②），之后这些都会被销毁。

- **f22dec28 及以前（ATOM 镜像现在带的仍是这个版本：`rocm/atom-dev:latest`，10-28 构建）**：`store()` / `retrieve()` 返回时丢掉导入的 ①。销毁会把处理线程卡到这次拷贝（以及 server 在这张卡上排的所有拷贝）做完，带 ② 的回复要等销毁结束才发出。处理线程是按 client 绑定的 affinity 线程，下一个请求只能排在后面；默认 `--max-gpu-workers 1` 时所有 rank 共用一个线程，4 张卡的拷贝被串成一条。
- **dev 合入 [#5439](https://github.com/LMCache/LMCache/pull/5439)（“Hold imported IPC events”）之后**：导入的 ① 被先留着，由一个按 stream 顺序执行的 host callback 在 device host-func dispatcher 线程上释放，处理线程不再当场销毁。但每次操作仍然导入一次、销毁一次，现在卡住的是 dispatcher 线程（它还负责投递 L1 读写完成通知）；而且这次销毁等待期间，处理线程为下一个请求 `create_event()` 新建 ② 时也会被卡住（见第 2 节最后一张表）。第 7 节实测了还剩多少。
- **完成 event 的环形队列**：导出满 2048 个之后，每新建一个就挤掉并销毁最老的，这又是一次 server 上的 IPC event 销毁。
- **引擎侧**：引擎每次导入 ②，future 释放时销毁。如果引擎自己每步新建、销毁 ①（ATOM 就是这样），这部分开销由引擎自己承担，不在 LMCache 里。

## 4. 设计：事件池

默认关闭，client 设置 `LMCACHE_MP_POOLED_EVENTS=1` 后开启。key 上带一个传输层提示 `pooled_events`（`compare=False`，不参与缓存身份）；server 按请求选择走池子还是走原路径，老 client 和老 server 都不受影响。

| | 原版 | 事件池 |
|---|---|---|
| ① 生产者 event | 发送引擎每步的那个 event | client adapter 在同一 stream 位置（紧跟引擎的 event）record **16 个轮转 event** 中的一个；每个 event record **31 次**后换新 |
| server 导入 ① | 每个请求导入一次，用完销毁 | **每个 handle 只导入一次**，按实例缓存（LRU 4096），之后直接 `wait` 缓存的导入 |
| ② 完成 event | 每个请求新建一个；2048 环形队列，满了销毁最老的 | **每个实例 512 个**，轮流重新 record，实例注册期间从不销毁 |
| 引擎导入 ② | 每个请求导入一次，释放时销毁 | handle 带前缀 `LMCPOOL1`，**只导入一次**，缓存（LRU 16384） |

每次操作，原版在两个进程里一共要新建 1 个、导入 2 个、销毁 3 个 IPC event；事件池在稳定状态下一个都不用（每个槽位每 31 次操作新建一个 ①，缓存淘汰要几个小时才发生一次）。

代码（dev 移植版，4 个文件，+196 行）：`lmcache/integration/atom/multi_process_adapter.py`（client 事件池）、`lmcache/v1/multiprocess/modules/lmcache_driven_transfer.py`（server 导入缓存、完成 event 池）、`lmcache/v1/platform/base/event_ipc.py`（client 对带前缀 handle 的导入缓存）、`lmcache/v1/multiprocess/custom_types.py`（`pooled_events`、`POOLED_EVENT_MAGIC`、`IPC_EVENT_MAX_RECORDS`）。

## 5. 复用为什么安全

- **重新 record 是保守的**：`record` 把 event 挪到新的 stream 位置，`wait`/`query` 看的总是最近一次 record。如果对方还没 wait/query 上一次的使用，这个 event 就被重新 record 了，对方看到的是**同一条** stream 上**更靠后**的位置：只会多等，不会提前放行。池子的大小（每个 rank 16 个生产者槽位、每个实例 512 个完成槽位）就是为了让这种情况很少发生。
- **32 次 record 的限制**：ROCm 7.2.x 的 `IPCEventEmulated::streamWait` 用 `signal[read_index]` 取下标时漏了 `% 32`，owner record 超过 32 次后，导入方的 `wait` 会报 `hipErrorInvalidValue`（develop 修复 [#12230](https://github.com/ROCm/rocm-systems/pull/12230)；[#11749](https://github.com/ROCm/rocm-systems/pull/11749)、[#11511](https://github.com/ROCm/rocm-systems/pull/11511) 仍 open）。所以生产者 event record 31 次就换新。完成 event 只被 `query`，query 的下标是对的，所以不用换。
- **生命周期**：缓存的导入和池里的完成 event 跟实例的 context entry 同生命周期，只有 LRU 淘汰会让导入离开缓存。4096 个生产者 handle、每个槽位每 31 次操作才出一个新 handle，够用几个小时，所以实际上不会淘汰到一个还有 wait 在排队的 event。事件池路径上也保留了 #5439 的 hold/release。
- **已验证**：`event_reuse_check.py` 在两个进程间把两种复用方式各跑 400 轮：owner 的 kernel 结束之前被判为完成的次数为 0，刚重新 record 后误报“已完成”的次数也为 0。用 `rot=40`（超过 32 次）时，对方在第 33 次 record 时 `wait` 失败，复现了 ROCm 的这个 bug。

## 6. 复现步骤

硬件：一台有 4 张 AMD Instinct GPU（MI300X/MI325X/MI350X/MI355X）的机器，ROCm 7.2.x。我们用的是 8×MI355X，镜像 `rocm/atom-dev:latest`（2026-10-28 构建，ROCm 7.2.4，torch 2.10.0+rocm7.2.4，LMCache 0.5.6.dev139+gf22dec28），GPU 4–7。

**第 1 层：单卡，只用 torch（约 2 分钟）**

```bash
python destroy_probe.py --trials 30 --busy-ms 40 --consumer 1 --out probe.json
```

**第 2 层：两个进程、一张卡，只用 torch（约 3 分钟）**

```bash
python event_reuse_check.py 400 31   # 事件池的两种复用方式：预期 0 次不安全、0 次过期
python event_reuse_check.py 60 40    # 超过 32 次 record：对方在第 33 次 wait 失败（ROCm 7.2.x bug）
```

**第 3 层：LMCache MP server + 4 个 client，不需要模型（每组约 4 分钟）**

`srvbench.py` 起 4 个 rank 进程，通过 `AtomMPWorkerAdapter` 注册 MiniMax-M3 形状的 KV（每个 rank 301 个 tensor，每 128 token 一个 block 2,995,200 B），再起一个 scheduler，按 ATOM 的方式每轮执行 lookup → retrieve → store。`sweep`：单条会话，prompt 从 16k 到 256k token，测没有竞争时每次操作的开销。`burst`：16 条 64k、32 条 32k 的会话同时到达，各重复 3 次（总共往 L1 写 638 GB）。

```bash
# 每个 LMCache 版本编译到单独的目录，例如
pip install --no-deps --no-build-isolation --target /w/site-dev /path/to/LMCache   # 需要 grpcio-tools
export HIP_VISIBLE_DEVICES=4,5,6,7 SB_T_SWEEP=16 SB_BURSTS=16x65536,32x32768 SB_BURST_R=3 SB_L1=800 SB_LAZY=0

PYTHONPATH=/w/site-dev      bash run_srvbench.sh runs/dev-stock@r1 0
PYTHONPATH=/w/site-pool-dev bash run_srvbench.sh runs/dev-pool@r1  1
PYTHONPATH=/w/site-dev      bash run_srvbench.sh runs/dev-stock-w4@r1 0 --max-gpu-workers 4
python aggregate.py runs dev-stock dev-pool dev-stock-w4
python sweep_curves.py runs dev-stock dev-pool dev-stock-w4   # 按 prompt 长度出曲线（pass1，各轮均值）
```

`run_srvbench.sh` 启动 server（`--chunk-size 256 --supported-transfer-mode lmcache_driven --eviction-ratio 0.01`，L1 由 `SB_L1` 指定，额外参数原样传给 server），跑完两个阶段后写出 `summary.json`。延迟在 client 端测：从 4 个 rank 都提交了这次操作开始，到每个 rank 的完成 event 都报告完成为止。**L1 要留足余量、保证全程不淘汰，并且启动时一次分配好（`SB_LAZY=0`）**：L1 只有 400 GB，或者用 `--l1-use-lazy` 边用边扩时，burst 阶段会淘汰甚至分配失败，延迟主要由内存压力决定，看不出 event 的差别。可以在 `server.log` 里搜 `triggering eviction`，并看 `summary.json` 里 retrieve 的次数（没有淘汰时每次是 144）。

## 7. 结果

所有数据：MI355X ×4（8 卡机器的 GPU 4–7），ROCm 7.2.4，`SB_T_SWEEP=16 SB_BURSTS=16x65536,32x32768 SB_BURST_R=3`，L1 800 GB 启动时一次分配好（`SB_LAZY=0`），**每组跑 3 次**，给出均值（最小–最大）。`pool` 组只开事件池（不开 client 哈希、不开批量构造 key）。`f22` = LMCache f22dec28（ATOM 镜像带的版本），`dev` = LMCache dev 4995e604（含 #5439），`-w4` = `--max-gpu-workers 4`。

**单条会话（sweep，无竞争）**：所有组都没有淘汰，数据都可比。

| client 端延迟 p50 | f22 原版 | f22 事件池 | dev 原版 | dev 事件池 | f22 原版 -w4 | dev 原版 -w4 | dev 事件池 -w4 |
|---|---|---|---|---|---|---|---|
| store，每 rank 16k 新 token ×4 rank（ms） | 164 (157–169) | **97** (95–100) | 180 (148–242) | **93** (84–103) | 124 (115–129) | 106 (102–112) | **52** (46–58) |
| retrieve，每 rank 最多 240k token ×4 rank（ms） | 386 (382–391) | **167** (161–174) | 394 (369–444) | **165** (157–174) | 191 (187–198) | 169 (157–177) | **96** (90–100) |

**多条会话同时到达（burst）**：

| | f22 原版 | f22 事件池 | dev 原版 | dev 事件池 ¹ | f22 原版 -w4 | dev 原版 -w4 |
|---|---|---|---|---|---|---|
| 16 条 64k 同时到达：总耗时（s） | 6.53 (5.28–7.41) | **2.45** (2.40–2.48) | 5.85 (5.49–6.45) | **2.44** (2.22–2.57) | 3.32 (3.29–3.34) | 3.19 (3.02–3.34) |
| 32 条 32k 同时到达：总耗时（s） | 8.92 (6.38–10.52) | **3.48** (3.37–3.55) | 7.35 (6.27–9.49) | **3.86** (3.28–4.47) | 4.91 (4.62–5.48) | 5.05 (4.78–5.47) |
| store 延迟 p50 / p90（ms） | 3909 / 4903 | **1447 / 1715** | 2924 / 4280 | **1523 / 2136** | 2214 / 2504 | 2098 / 2694 |
| retrieve 延迟 p50 / p90（ms） | 2055 / 3839 | **804 / 1418** | 1856 / 3278 | **751 / 1325** | 1015 / 1868 | 1001 / 1825 |

¹ `dev 事件池` 的 3 次里有 1 次在最后几轮 32×32k 时 L1 碰到了淘汰线，144 轮里只有 116 轮命中前缀并做了 retrieve（表中其余各次都是 142–144 轮）。`dev 事件池 -w4` 的 burst 数据没有列出：它 3 次都发生了淘汰（144 轮里只有 77–101 轮做了 retrieve），实际做的工作比其他组少，不可比；上面 sweep 的数据是干净的。

**结论**

- **只靠 #5439 去不掉卡顿**：两个阶段里 dev 原版和 f22 原版都在噪声范围内。销毁挪出了处理线程，但每次操作仍然要新建、导入、销毁 IPC event。
- **事件池让单次请求延迟减半，多会话同时到达时的总耗时缩短到 1/1.9–1/2.7**，两个版本上都是如此：store −41…−48 %，retrieve −57…−58 %，burst 中 store/retrieve 的 p90 −50…−65 %。
- **多开 GPU worker 有帮助，但不如事件池**：`--max-gpu-workers 4` 让每个 rank 的销毁只卡住自己的线程，4 张卡不再串行。dev 原版 -w4 的 burst 是 3.19 / 5.05 s，而事件池只用 1 个 worker 就是 2.44 / 3.86 s。两者可以叠加：dev 事件池 -w4 的 sweep 延迟最低（52 / 96 ms）。
- **越快的组越先碰到 L1 淘汰线**：写入相同字节数时，原版和事件池占用的 L1 内存一样（写入 8 万个 chunk 后 485 GB 对 484 GB，没有泄漏）；但事件池同时在途的写入更多，占用比例的峰值更高。L1 要留余量（我们 638 GB 数据用了 800 GB；事件池 -w4 组用 1200 GB 更稳妥）。

## 7b. 持久 kernel（例如 ATOM mono）与 GPU 侧等待

ROCm 7.2 上，server 对导入的 IPC event 做 `stream.wait_event`，实际是启动一个只有 1 个 work-item 的轮询 kernel（`streamOpsWait`，rocclr 默认 `GPU_STREAMOPS_CP_WAIT=false`），一直读 event 的信号，直到 client 的 GPU 工作把它写成 0。ATOM 的 mono decode kernel 要求所有 CTA 同时驻留（每个 CU 一个）并互相自旋等待。如果持久 kernel 派发时轮询 kernel 已经驻留、占住了它某个 CTA 需要的槽位，持久 kernel 等这个 CU，轮询 kernel 等持久 kernel 跑完后才会写的信号，两者互相卡死。用 `mono_hang_repro.py` 在单张 MI355X（ROCm 7.2.4）上复现：

| 持久 kernel 占满 | `--mode stream`（原版 GPU 等待） | `--mode host` | `--mode gate` |
|---|---|---|---|
| 每个 CU（每 CU 2 个 1024 线程 CTA） | **第 2 轮卡死**（2 次都是） | 100 轮正常 | 100 轮正常 |
| 每个 CU 一半（每 CU 1 个 CTA） | 100 轮正常 | – | – |

两个默认关闭的开关，保证不在 GPU 上等一个还没完成的 event：

- `LMCACHE_MP_SUBMIT_AFTER_PRODUCER=1`（client）：ATOM adapter 把每个 store/retrieve 交给后台线程，等 producer event 完成后再发送；`submit_*` 立即返回一个 future，发送前它报告未完成。引擎调用线程上每次 store 提交从 4.4 ms 降到 0.75 ms。
- `LMCACHE_MP_HOST_WAIT_PRODUCER=1`（server）：用 `query()`（只读主机内存）检查 producer event，已完成就跳过 stream wait，未完成就在 CPU 上轮询，不再往 GPU 上排等待。

两个都开时，`srvbench`（事件池、`--max-gpu-workers 4`、各 2 轮、L1 不扩容）性能不变：单条会话 store/retrieve p50 33/84 ms 对 33/85 ms，多会话 0.90/1.20 s 对 0.85/1.32 s，0 次失败。ROCm develop（[#5575](https://github.com/ROCm/rocm-systems/pull/5575)）把模拟的 IPC event 换成由 CP 等待的 ROCr IPC 信号，可以从根本上去掉这个轮询 kernel。

**压测注意**：L1 的 lazy 扩容每次用 `hipHostRegister` 锁 10 GB，锁的过程中 server 的 IPC event record 和注册时打开 IPC 句柄会被挡住几秒（表现为约 10 s 的 burst 卡顿、注册超时、实例被回收）。压测时让 `SB_L1` 等于 `SB_L1_INIT`，跑的过程中不扩容。

## 8. 局限与待确认的问题

- **只针对 ROCm**：CUDA 上 `cudaEventDestroy` 不会同步，事件池在那里只省掉导入和新建的调用。我们没有测 CUDA。等 ROCm 7.2 的 backport（#11529）合入，或者带 #7710 的版本发布后，销毁的开销会消失，事件池就变成一个较小的优化（少几次导入）。
- **合成负载**：`srvbench` 没有模型 forward，GPU 上只有拷贝。真实引擎里 forward 让 GPU 更忙，每次销毁会等得更久：在一个真实的 MiniMax-M3 TP4 c48 服务里，我们测到 IPC event 销毁 p50 8–10 ms、最大 2.4 s。
- **引擎侧的 event**：事件池只覆盖 LMCache 自己新建或导入的 event。引擎如果每步自己新建、销毁跨进程 event，这部分销毁仍由引擎承担。
- **和 `--max-gpu-workers` 的关系**：多开 GPU worker，可以让不同 rank 的请求落在 server 的不同线程上；但销毁本身没有去掉，仍会卡住执行它的那个线程。第 7 节的 `-w4` 组测的就是两者叠加的效果。
