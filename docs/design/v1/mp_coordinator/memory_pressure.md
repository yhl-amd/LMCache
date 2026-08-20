# Fleet Memory Pressure

A coordinator-level read-only view of how full each MP server's memory
compartments are. It joins two things the coordinator holds separately: byte
usage derived from the admitted cache-event stream, and capacity declared by
each server at registration. Neither is a pressure reading on its own.

Code: `lmcache/v1/mp_coordinator/controllers/usage_manager.py` (usage view),
`lmcache/v1/mp_coordinator/server_config.py` (capacity registry),
`lmcache/v1/mp_coordinator/http_apis/memory_api.py` (REST endpoints),
`lmcache/v1/distributed/storage_manager.py` (MP-server capacity source),
`lmcache/v1/mp_coordinator/registrar.py` (MP-server declaration).

## Why

The coordinator can already say how many bytes a tier holds. It cannot say
whether that is a lot, because nothing tells it how many bytes the tier *can*
hold. Pressure is `used / capacity`, and the cache-event stream carries only
the numerator.

The usage half needs no new tracking. `CacheUsageManager` already rolls the
admitted cache-event stream up per `(instance_id, backend)` for both tiers —
`get_bytes_by_instance(tier)` — which is exactly the axis a pressure reading
needs. This capability adds only the denominator.

## Why capacity does not ride the event stream

Capacity is configuration. It changes when a server boots and when it is
reconfigured — not once per cache event. Putting it on a per-event channel
would republish an unchanging number thousands of times a second; putting it
on registration republishes it exactly when it can change, because a
reconfigured server re-registers.

## Architecture

```
MP server                                   Coordinator
─────────                                   ───────────
StorageManager.get_memory_capacities()
  L1: get_configured_capacity_bytes()
      → one entry per backing medium
  L2: per adapter, total_capacity_bytes
      + shared flag from its config
        │
        │  registrar: ModuleMemoryCapacity → ModuleCapacityModel
        ▼
  POST /instances ──────────────────▶  ServerConfigRegistry.declare()
   (memory_modules)                      replaces the prior set wholesale
                                                    │
L2 adapter / L1 manager publish                     │  capacity
  l1.* / l2.* on the event bus                      │
        │                                           │
        ▼                                           │
  CacheEventSubscriber (cache_events.md)            │
        │                                           │
        ▼                                           │
  POST /events ─────────────────────▶  EventGate (ingest.md)                │
                                          └─ CacheEventBroadcaster          │
                                             ├─ KeyDirectory                │
                                             ├─ FleetEvictionController     │
                                             └─ CacheUsageManager  ─────────┤
                                                  (instance, tier, backend) │
                                                                    usage   │
                                                                            ▼
                                                       GET /memory  joins both
                                                       GET /memory/{instance_id}
```

## The compartment axis

A "module" is a compartment that owns bytes: the L1 pool of one backing
medium, or one L2 adapter. Identity is `(tier, backend)` — the same axis
cache events tag placements with, so a declaration and a usage total join
without translation.

L1 capacity is reported **per medium** because one tier can span several: a
hybrid Device-DAX tier backs objects with both `devdax` and `dram`, and
`L1ObjectMeta.backend` tags each placement accordingly. Flattening it to one
total would leave two compartments of usage sharing one denominator.

## Capacity is the configured size, not the live heap

`get_configured_capacity_bytes()` exists rather than reusing
`get_memory_usage()[1]`. On the default lazy allocator that total is the
**currently grown heap**: it starts small and grows on demand, so a freshly
booted server would report itself nearly full and then appear to drain as the
pool warms. The configured size is stable from boot and is the only sound
denominator.

| Manager | `get_memory_usage()[1]` | `get_configured_capacity_bytes()` |
| --- | --- | --- |
| `L1MemoryManager` (default, lazy) | grown heap | `{dram: size_in_bytes}` |
| `GDSL1MemoryManager` | configured slab | `{gds: size_in_bytes}` |
| `DevDaxL1MemoryManager` | live active arenas | `{devdax: …}` + `{dram: …}` when hybrid |

`L1Manager.report_status()` exposes the same value as
`memory_configured_bytes` so the status dict and the capacity API cannot
drift. `memory_total_bytes` keeps its existing meaning for existing consumers.

## Shared pools are counted once

An adapter with `shared=True` is storage several instances mount — one S3
bucket, one CXL region. Its bytes and its capacity are fleet-scoped. Summing
them across the N servers that report them would overstate both by N, and the
result looks plausible, which is worse than an obvious error.

The usage tracker follows the same convention the key directory and the
per-salt view use: shared placements are keyed under an empty owner
(`SHARED_OWNER`), so they are counted once and attributed to no instance.
`GET /memory` reports them under `shared_modules`, never inside an instance.

Capacity for a shared pool is resolved across every server that declares it.
Declarations should agree; when they do not, the pool is reported as
undeclared rather than picking one, since preferring a value would make the
reading depend on registration order.

## Unknown is a value

`capacity_bytes == 0` means the server declares no limit. This is the
**common** case, not an edge case: `fs`, `mooncake`, `p2p`, and `sagemaker`
return `0` unconditionally with no configuration knob at all.

So `usage_ratio` is `null` — not `0.0`, not `-1.0` — whenever there is no
capacity to divide by. A number there would be read as a real occupancy, and
a fleet view that treats capacity-less backends as empty reports "healthy"
regardless of what is happening.

Ratios above `1.0` are **not clamped**. A tier holding more than its declared
cap means the declaration disagrees with what the tier admitted, and hiding
that would hide a misconfiguration.

## Lifecycle

- **Registration** declares capacity, before membership is recorded — so a
  memory read cannot find a server registered but its modules undeclared.
  Re-registration replaces the set wholesale: a server that dropped an
  adapter must not keep the old compartment's capacity.
- **Fencing** (`fence_instance`, on restart or departure) discards the
  instance's **L1** bytes. L1 lives in the reporting process and dies with
  it; L2 bytes outlive the reporter and leave only through `DELETE`.
- **Deregistration** drops the capacity declaration. A departed server's caps
  describe a process that no longer exists, and keeping them would grow
  without bound across a churning fleet. Its surviving L2 bytes are still
  reported, without a ratio.

An instance appears in `GET /memory` when it is registered, when it holds
bytes, or when it has declared capacity — so a deregistered server whose L2
placements survive is not silently dropped.

## Scope

Read-only. This never evicts, throttles, or pushes. There is no derived
pressure level, no smoothing, no trend, and no ranking: while most L2
backends have no declared capacity, a normalized `LOW`/`HIGH`/`CRITICAL`
score would be confidently wrong on the majority of deployments. Bytes,
capacity-where-known, and an explicit unknown are what the data supports
today.

Named follow-ups: `lmcache describe` should prefer `memory_configured_bytes`
over `memory_total_bytes` (it currently prints the grown heap as "L1
capacity"); forwarding `L1_ALLOCATION_FAILED` to the coordinator would give a
directly-measured pressure signal rather than an inferred ratio; a derived
level becomes reasonable once capacity declaration is widespread.
