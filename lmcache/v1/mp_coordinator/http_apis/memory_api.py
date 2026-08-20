# SPDX-License-Identifier: Apache-2.0
"""Fleet memory-pressure endpoints for the MP coordinator.

Joins per-compartment byte totals (``MemoryUsageTracker``) with the
capacities declared at registration (``ServerConfigRegistry``). Read-only:
never evicts, throttles, or pushes. A ``null`` usage ratio means capacity is
undeclared, not that the compartment is empty.
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


def _to_status(usage: ModuleUsage, capacity_bytes: int) -> ModuleMemoryStatus:
    """Join one compartment's usage to its declared capacity.

    Args:
        usage: The compartment's current byte total.
        capacity_bytes: Its declared capacity, or
            :data:`~lmcache.v1.mp_coordinator.server_config.UNDECLARED_CAPACITY`
            (0) if undeclared.

    Returns:
        The joined status; ``usage_ratio`` is ``None`` when capacity is
        undeclared.
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

    Declared-but-unfilled compartments are reported with ``used_bytes=0`` so
    a freshly started server does not look unmonitored.

    Args:
        instance_id: The server being described.
        usage_modules: Its privately-owned compartments holding bytes.
        declared: Its declared capacities.
        registered: Whether it is currently in the instance registry.

    Returns:
        The assembled status.
    """
    capacities = {(m.tier, m.backend): m.capacity_bytes for m in declared}
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

    A shared pool is one physical store, so declarations are agreed on, never
    summed; disagreement resolves to undeclared rather than to whichever
    server registered first.

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
        A :class:`FleetMemoryResponse`. A server appears if it is registered,
        still holds bytes, or has declared capacity.
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
        HTTPException: 404 if the id is unknown -- not registered, holding no
            bytes, and declaring no capacity.
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
