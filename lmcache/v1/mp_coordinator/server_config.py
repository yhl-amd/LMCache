# SPDX-License-Identifier: Apache-2.0
"""Per-MP-server memory configuration declared at registration.

The cache-event stream tells the coordinator how many bytes a module
*holds*; it never tells it how many that module *can* hold. Capacity is
configuration -- it changes when a server boots and when it is
reconfigured, not once per cache event -- so servers declare it on
``POST /instances`` and this registry is where the fleet's declarations
live. Joining a declaration to a usage total is what turns a byte count
into a pressure reading.

This module stores declarations only. It does not read cache events, does
not compute pressure, and does not track liveness -- membership stays
:class:`~lmcache.v1.mp_coordinator.registry.InstanceRegistry`'s job, which
is why capacity lives here rather than on ``MPInstance``.
"""

# Future
from __future__ import annotations

# Standard
from collections.abc import Sequence
from dataclasses import dataclass
import threading

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import Tier

logger = init_logger(__name__)

# ``capacity_bytes`` value meaning "this server declared no cap for this
# module". Distinct from a real cap, which is always positive: an adapter
# with no configured limit (the default for fs / mooncake / p2p /
# sagemaker) genuinely has no denominator, and reporting one as ``0``
# capacity would read as permanently full.
UNDECLARED_CAPACITY = 0


@dataclass(frozen=True)
class ModuleCapacity:
    """One memory compartment's declared capacity on one MP server.

    A "module" here is a compartment that owns bytes: the L1 pool, or one
    L2 adapter. This is the same ``(tier, backend)`` axis the cache-event
    stream reports placements on, so a declaration and a usage total join
    without translation.

    Attributes:
        tier: The cache tier this compartment belongs to (``L1`` or
            ``L2``; never ``ALL``).
        backend: The storage backend within the tier (``"dram"``,
            ``"cxl"``, ``"fs"``, ``"s3"``, ...). Non-empty.
        capacity_bytes: Declared capacity in bytes, or
            :data:`UNDECLARED_CAPACITY` when the server has no configured
            limit for this compartment.
        shared: ``True`` when this backend is a storage domain several
            instances mount (one S3 bucket, one CXL pool). Fleet-scoped
            capacity, so it must not be summed across the instances that
            declare it.
    """

    tier: Tier
    backend: str
    capacity_bytes: int
    shared: bool = False

    def __post_init__(self) -> None:
        """Enforce intrinsic invariants.

        Raises:
            ValueError: If ``backend`` is empty, ``capacity_bytes`` is
                negative, or ``tier`` is not a concrete tier.
        """
        if not self.backend:
            raise ValueError("backend must be non-empty")
        if self.capacity_bytes < 0:
            raise ValueError(f"capacity_bytes must be >= 0 (got {self.capacity_bytes})")
        if self.tier not in (Tier.L1, Tier.L2):
            raise ValueError(
                f"capacity must target a concrete tier (got {self.tier.value!r})"
            )


class ServerConfigRegistry:
    """Thread-safe store of each MP server's declared module capacities.

    Declarations arrive through :meth:`declare` (from the registration
    handler) and are dropped through :meth:`forget` (on deregistration).
    Reads go through :meth:`get` and :meth:`get_all`.

    A re-registration replaces a server's declaration wholesale rather
    than merging: a server that dropped an L2 adapter must not keep the
    old compartment's capacity, and replacement is the only way to say so
    with a full declaration.
    """

    def __init__(self) -> None:
        """Initialize an empty registry."""
        self._lock = threading.Lock()
        self._by_instance: dict[str, tuple[ModuleCapacity, ...]] = {}

    def declare(self, instance_id: str, modules: Sequence[ModuleCapacity]) -> None:
        """Record ``instance_id``'s module capacities, replacing any prior set.

        Args:
            instance_id: The declaring server's id. Non-empty.
            modules: Its compartments. An empty sequence records that the
                server declared nothing, which reads back as no known
                capacity rather than as zero capacity.

        Raises:
            ValueError: If ``instance_id`` is empty, or two entries share
                a ``(tier, backend)`` identity.
        """
        if not instance_id:
            raise ValueError("instance_id must be non-empty")
        seen: set[tuple[Tier, str]] = set()
        for module in modules:
            identity = (module.tier, module.backend)
            if identity in seen:
                raise ValueError(
                    f"duplicate capacity declaration for "
                    f"{module.tier.value}/{module.backend}"
                )
            seen.add(identity)
        with self._lock:
            self._by_instance[instance_id] = tuple(modules)

    def get(self, instance_id: str) -> tuple[ModuleCapacity, ...]:
        """Return ``instance_id``'s declared compartments.

        Args:
            instance_id: The server to look up.

        Returns:
            Its declarations, or an empty tuple when the server is unknown
            or declared nothing.
        """
        with self._lock:
            return self._by_instance.get(instance_id, ())

    def get_all(self) -> dict[str, tuple[ModuleCapacity, ...]]:
        """Return a snapshot of every server's declarations.

        Returns:
            A copy mapping ``instance_id`` to its compartments; mutating it
            does not affect the registry.
        """
        with self._lock:
            return dict(self._by_instance)

    def forget(self, instance_id: str) -> None:
        """Drop ``instance_id``'s declaration. Idempotent.

        Args:
            instance_id: The departed server.
        """
        with self._lock:
            self._by_instance.pop(instance_id, None)
