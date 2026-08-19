# SPDX-License-Identifier: Apache-2.0
"""Fleet memory-pressure endpoints for the MP coordinator.

Joins the two halves the coordinator already holds separately: how many
bytes each compartment holds (derived from the admitted cache-event stream
by :class:`~lmcache.v1.mp_coordinator.controllers.memory_usage.MemoryUsageTracker`)
and how many it can hold (declared at registration and kept in
:class:`~lmcache.v1.mp_coordinator.server_config.ServerConfigRegistry`).
Neither half is a pressure reading on its own.

Read-only: these endpoints never evict, throttle, or push. A compartment
whose server declared no capacity is reported with its byte count and a
``null`` ratio rather than an inferred one.
"""

# Third Party
from fastapi import APIRouter, HTTPException, Request, status

# First Party
from lmcache.v1.distributed.api import Tier
from lmcache.v1.mp_coordinator.controllers.memory_usage import (
    ModuleUsage,
)
from lmcache.v1.mp_coordinator.http_apis.dependencies import get_context
from lmcache.v1.mp_coordinator.schemas import (
    FleetMemoryResponse,
    InstanceMemoryStatus,
    ModuleMemoryStatus,
)
from lmcache.v1.mp_coordinator.server_config import (
    UNDECLARED_CAPACITY,
    ModuleCapacity,
)

router = APIRouter()


def _capacity_index(
    modules: tuple[ModuleCapacity, ...],
) -> dict[tuple[Tier, str], int]:
    """Index declared capacities by compartment.

    Args:
        modules: One server's declarations.

    Returns:
        Mapping from ``(tier, backend)`` to declared bytes.
    """
    return {(m.tier, m.backend): m.capacity_bytes for m in modules}


def _to_status(usage: ModuleUsage, capacity_bytes: int) -> ModuleMemoryStatus:
    """Join one compartment's usage to its declared capacity.

    Args:
        usage: The compartment's current byte total.
        capacity_bytes: Its declared capacity, or
            :data:`~lmcache.v1.mp_coordinator.server_config.UNDECLARED_CAPACITY`.

    Returns:
        The joined status, with ``usage_ratio`` left ``None`` when there is
        no capacity to divide by.
    """
    ratio = (
        usage.used_bytes / capacity_bytes
        if capacity_bytes > UNDECLARED_CAPACITY
        else None
    )
    return ModuleMemoryStatus(
        tier=usage.tier,
        backend=usage.backend,
        shared=usage.shared,
        used_bytes=usage.used_bytes,
        capacity_bytes=capacity_bytes,
        usage_ratio=ratio,
    )


def _instance_status(
    instance_id: str,
    usage_modules: tuple[ModuleUsage, ...],
    declared: tuple[ModuleCapacity, ...],
    registered: bool,
) -> InstanceMemoryStatus:
    """Build one server's status from its usage and its declaration.

    Compartments the server declared but has not yet filled are included
    with ``used_bytes=0``: an empty pool is a real, useful answer, and
    omitting it would make a freshly started server look unmonitored.

    Args:
        instance_id: The server being described.
        usage_modules: Its privately-owned compartments holding bytes.
        declared: Its declared capacities.
        registered: Whether it is currently in the instance registry.

    Returns:
        The assembled status.
    """
    capacities = _capacity_index(declared)
    seen: set[tuple[Tier, str]] = set()
    statuses: list[ModuleMemoryStatus] = []
    for usage in usage_modules:
        identity = (usage.tier, usage.backend)
        seen.add(identity)
        statuses.append(
            _to_status(usage, capacities.get(identity, UNDECLARED_CAPACITY))
        )
    for module in declared:
        identity = (module.tier, module.backend)
        if identity in seen or module.shared:
            continue
        statuses.append(
            _to_status(
                ModuleUsage(
                    tier=module.tier,
                    backend=module.backend,
                    used_bytes=0,
                    shared=False,
                ),
                module.capacity_bytes,
            )
        )
    statuses.sort(key=lambda m: (m.tier.value, m.backend))
    return InstanceMemoryStatus(
        instance_id=instance_id,
        registered=registered,
        declared_capacity=bool(declared),
        modules=statuses,
    )


def _shared_capacities(
    declarations: dict[str, tuple[ModuleCapacity, ...]],
) -> dict[tuple[Tier, str], int]:
    """Resolve each shared pool's capacity across every server that declares it.

    A shared pool is one physical store, so its declarations should agree.
    When they do not, the pool is reported as undeclared: there is no basis
    for preferring one server's number, and silently taking the first would
    make the reading depend on registration order.

    Args:
        declarations: Every server's declared compartments.

    Returns:
        Mapping from ``(tier, backend)`` to the agreed capacity, or to
        :data:`~lmcache.v1.mp_coordinator.server_config.UNDECLARED_CAPACITY`
        where servers disagree.
    """
    claims: dict[tuple[Tier, str], set[int]] = {}
    for modules in declarations.values():
        for module in modules:
            if not module.shared:
                continue
            claims.setdefault((module.tier, module.backend), set()).add(
                module.capacity_bytes
            )
    resolved: dict[tuple[Tier, str], int] = {}
    for identity, values in claims.items():
        resolved[identity] = values.pop() if len(values) == 1 else UNDECLARED_CAPACITY
    return resolved


@router.get("/memory")
async def fleet_memory(request: Request) -> FleetMemoryResponse:
    """Return the memory status of every MP server and shared pool.

    Args:
        request: The incoming request, carrying the coordinator context.

    Returns:
        A :class:`FleetMemoryResponse`. Servers appear when they are
        registered or when they still hold bytes, so a deregistered server
        whose L2 placements survive is not silently dropped.
    """
    ctx = get_context(request)
    usage = ctx.memory_usage
    declarations = ctx.server_config.get_all()
    registered = {instance.instance_id for instance in ctx.registry.all_instances()}

    instance_ids = sorted(registered | set(usage.get_instances()) | set(declarations))
    instances = [
        _instance_status(
            instance_id=instance_id,
            usage_modules=usage.get_for_instance(instance_id),
            declared=declarations.get(instance_id, ()),
            registered=instance_id in registered,
        )
        for instance_id in instance_ids
    ]

    shared_caps = _shared_capacities(declarations)
    shared = [
        _to_status(
            module, shared_caps.get((module.tier, module.backend), UNDECLARED_CAPACITY)
        )
        for module in usage.get_shared()
    ]
    return FleetMemoryResponse(instances=instances, shared_modules=shared)


@router.get("/memory/{instance_id}")
async def instance_memory(instance_id: str, request: Request) -> InstanceMemoryStatus:
    """Return one MP server's memory status.

    Args:
        instance_id: The server to report on.
        request: The incoming request, carrying the coordinator context.

    Returns:
        That server's :class:`InstanceMemoryStatus`.

    Raises:
        HTTPException: 404 if the coordinator knows nothing about the id --
            it is neither registered, nor holding bytes, nor has declared
            any capacity.
    """
    ctx = get_context(request)
    declared = ctx.server_config.get(instance_id)
    usage_modules = ctx.memory_usage.get_for_instance(instance_id)
    registered = ctx.registry.contains(instance_id)
    if not registered and not declared and not usage_modules:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"unknown instance {instance_id!r}",
        )
    return _instance_status(
        instance_id=instance_id,
        usage_modules=usage_modules,
        declared=declared,
        registered=registered,
    )
