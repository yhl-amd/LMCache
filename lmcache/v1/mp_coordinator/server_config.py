# SPDX-License-Identifier: Apache-2.0
"""Per-MP-server memory configuration declared at registration.

Cache events report bytes held, never bytes holdable, so servers declare
capacity on ``POST /instances`` and this registry stores it. Declarations
only: no event handling, no pressure math, no liveness -- membership stays
:class:`~lmcache.v1.mp_coordinator.registry.InstanceRegistry`'s job.
"""

# Future
from __future__ import annotations

# Standard
from collections.abc import Sequence
from dataclasses import dataclass
import threading

# First Party
from lmcache.v1.distributed.api import Tier

# ``capacity_bytes`` sentinel meaning "no cap declared". Real caps are always
# positive; an unlimited adapter reported as 0 would read as permanently full.
UNDECLARED_CAPACITY = 0


@dataclass(frozen=True)
class ModuleCapacity:
    """One memory compartment's declared capacity on one MP server.

    A module is the L1 pool or one L2 adapter, keyed on the same
    ``(tier, backend)`` axis cache events report placements on.

    Attributes:
        tier: Cache tier (``L1`` or ``L2``; never ``ALL``).
        backend: Storage backend within the tier (``"dram"``, ``"cxl"``,
            ``"fs"``, ``"s3"``, ...). Non-empty.
        capacity_bytes: Declared capacity in bytes, or
            :data:`UNDECLARED_CAPACITY` when no limit is configured.
        shared: ``True`` when several instances mount one storage domain
            (one S3 bucket, one CXL pool). Fleet-scoped: count once, never
            sum across declaring instances.
    """

    tier: Tier
    backend: str
    capacity_bytes: int
    shared: bool = False

    def __post_init__(self) -> None:
        """Validate the declaration.

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

    :meth:`declare` replaces a server's declaration wholesale, so a dropped
    L2 adapter's capacity does not linger.
    """

    def __init__(self) -> None:
        """Initialize an empty registry."""
        self._lock = threading.Lock()
        self._by_instance: dict[str, tuple[ModuleCapacity, ...]] = {}

    def declare(self, instance_id: str, modules: Sequence[ModuleCapacity]) -> None:
        """Record ``instance_id``'s module capacities, replacing any prior set.

        Args:
            instance_id: The declaring server's id. Non-empty.
            modules: Its compartments. Empty records that the server declared
                nothing, which reads back as unknown capacity, not zero.

        Raises:
            ValueError: If ``instance_id`` is empty, or two entries share a
                ``(tier, backend)`` identity.
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
            A copy mapping ``instance_id`` to its compartments.
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
