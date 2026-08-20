# SPDX-License-Identifier: Apache-2.0
"""Fleet memory-pressure endpoints for the MP coordinator.

Joins per-compartment byte totals (``CacheUsageManager``'s per-instance
rollup) with the capacities declared at registration
(``ServerConfigRegistry``). Read-only: never evicts, throttles, or pushes. A
``null`` usage ratio means capacity is undeclared, not that the compartment
is empty.
"""

# Third Party
from fastapi import APIRouter, HTTPException, Request, status

# First Party
from lmcache.v1.distributed.api import Tier
from lmcache.v1.mp_coordinator.controllers.usage_manager import CacheUsageManager
from lmcache.v1.mp_coordinator.http_apis.dependencies import get_context
from lmcache.v1.mp_coordinator.schemas import (
    FleetMemoryResponse,
    InstanceMemoryStatus,
    ModuleMemoryStatus,
)
from lmcache.v1.mp_coordinator.server_config import UNDECLARED_CAPACITY, ModuleCapacity

router = APIRouter()

# ``CacheUsageManager`` keys fleet-shared pools under this instance id: their
# bytes belong to the fleet, so they are counted once, never per mount.
_SHARED_OWNER = ""

_TIERS = (Tier.L1, Tier.L2)


def _usage_by_owner(
    usage_manager: CacheUsageManager,
) -> dict[str, dict[tuple[Tier, str], int]]:
    """Collect used bytes for every owner across both tiers.

    Args:
        usage_manager: The fleet usage view.

    Returns:
        ``instance_id`` -> ``(tier, backend)`` -> bytes. Shared pools appear
        under :data:`_SHARED_OWNER`.
    """
    by_owner: dict[str, dict[tuple[Tier, str], int]] = {}
    for tier in _TIERS:
        for owner, backends in usage_manager.get_bytes_by_instance(tier).items():
            for backend, used in backends.items():
                by_owner.setdefault(owner, {})[(tier, backend)] = used
    return by_owner


def _to_status(
    tier: Tier, backend: str, used_bytes: int, capacity_bytes: int, shared: bool
) -> ModuleMemoryStatus:
    """Join one compartment's usage to its declared capacity.

    Args:
        tier: The compartment's cache tier.
        backend: Storage backend within the tier.
        used_bytes: Bytes the compartment holds.
        capacity_bytes: Declared capacity, or :data:`UNDECLARED_CAPACITY`.
        shared: Whether this is a fleet-shared pool.

    Returns:
        The joined status; ``usage_ratio`` is ``None`` with no capacity to
        divide by.
    """
    return ModuleMemoryStatus(
        tier=tier,
        backend=backend,
        shared=shared,
        used_bytes=used_bytes,
        capacity_bytes=capacity_bytes,
        usage_ratio=(
            used_bytes / capacity_bytes
            if capacity_bytes > UNDECLARED_CAPACITY
            else None
        ),
    )


def _instance_status(
    instance_id: str,
    used: dict[tuple[Tier, str], int],
    declared: tuple[ModuleCapacity, ...],
    registered: bool,
) -> InstanceMemoryStatus:
    """Build one server's status from its usage and its declaration.

    Declared-but-empty compartments report ``used_bytes=0`` so a freshly
    started server does not look unmonitored.

    Args:
        instance_id: The server being described.
        used: Its ``(tier, backend)`` byte totals.
        declared: Its declared capacities.
        registered: Whether it is currently in the instance registry.

    Returns:
        The assembled status.
    """
    capacities = {(m.tier, m.backend): m.capacity_bytes for m in declared}
    statuses = [
        _to_status(
            tier,
            backend,
            used_bytes,
            capacities.get((tier, backend), UNDECLARED_CAPACITY),
            shared=False,
        )
        for (tier, backend), used_bytes in used.items()
    ]
    for module in declared:
        if module.shared or (module.tier, module.backend) in used:
            continue
        statuses.append(
            _to_status(
                module.tier, module.backend, 0, module.capacity_bytes, shared=False
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
    """Resolve each shared pool's capacity across its declaring servers.

    One pool is one store, so declarations should agree. A disagreement reads
    as undeclared rather than picking one, which would make the answer depend
    on registration order.

    Args:
        declarations: Every server's declared compartments.

    Returns:
        ``(tier, backend)`` -> agreed capacity, or :data:`UNDECLARED_CAPACITY`.
    """
    claims: dict[tuple[Tier, str], set[int]] = {}
    for modules in declarations.values():
        for module in modules:
            if module.shared:
                claims.setdefault((module.tier, module.backend), set()).add(
                    module.capacity_bytes
                )
    return {
        identity: values.pop() if len(values) == 1 else UNDECLARED_CAPACITY
        for identity, values in claims.items()
    }


@router.get("/memory")
async def fleet_memory(request: Request) -> FleetMemoryResponse:
    """Return the memory status of every MP server and shared pool.

    Args:
        request: The incoming request, carrying the coordinator context.

    Returns:
        A :class:`FleetMemoryResponse`. A server appears when it is
        registered, still holds bytes, or declared capacity, so a
        deregistered server whose L2 placements survive is not dropped.
    """
    ctx = get_context(request)
    declarations = ctx.server_config.get_all()
    registered = {instance.instance_id for instance in ctx.registry.all_instances()}
    by_owner = _usage_by_owner(ctx.usage_manager)
    owned = {owner for owner in by_owner if owner != _SHARED_OWNER}

    instances = [
        _instance_status(
            instance_id=instance_id,
            used=by_owner.get(instance_id, {}),
            declared=declarations.get(instance_id, ()),
            registered=instance_id in registered,
        )
        for instance_id in sorted(registered | owned | set(declarations))
    ]

    shared_caps = _shared_capacities(declarations)
    shared = sorted(
        (
            _to_status(
                tier,
                backend,
                used_bytes,
                shared_caps.get((tier, backend), UNDECLARED_CAPACITY),
                shared=True,
            )
            for (tier, backend), used_bytes in by_owner.get(_SHARED_OWNER, {}).items()
        ),
        key=lambda m: (m.tier.value, m.backend),
    )
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
        HTTPException: 404 when the coordinator knows nothing about the id --
            not registered, holding no bytes, and having declared nothing.
    """
    ctx = get_context(request)
    declared = ctx.server_config.get(instance_id)
    used = _usage_by_owner(ctx.usage_manager).get(instance_id, {})
    registered = ctx.registry.contains(instance_id)
    if not registered and not declared and not used:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"unknown instance {instance_id!r}",
        )
    return _instance_status(
        instance_id=instance_id,
        used=used,
        declared=declared,
        registered=registered,
    )
