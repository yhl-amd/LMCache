# SPDX-License-Identifier: Apache-2.0
"""Per-server, per-module byte usage view for the MP coordinator.

Tracks the ``(instance_id, tier, backend)`` axis that the per-salt
:class:`~lmcache.v1.mp_coordinator.controllers.usage_manager.L2UsageManager`
folds away, and covers L1 as well as L2.
"""

# Future
from __future__ import annotations

# Standard
from dataclasses import dataclass
import threading

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import ObjectKey, Tier
from lmcache.v1.mp_coordinator.api import CacheEventBatch, CacheEventType

logger = init_logger(__name__)

# Owner value for a fleet-shared pool (one S3 bucket, one CXL region), which
# is counted once for the fleet instead of once per reporting instance.
SHARED_OWNER = ""


@dataclass(frozen=True)
class ModuleUsage:
    """Bytes held by one memory compartment.

    Attributes:
        tier: The compartment's cache tier (``L1`` or ``L2``).
        backend: The storage backend within the tier.
        used_bytes: Bytes currently placed in this compartment.
        shared: ``True`` for a fleet-shared pool, whose ``used_bytes`` is
            the pool total and belongs to no single instance.
    """

    tier: Tier
    backend: str
    used_bytes: int
    shared: bool


# One tracked placement: ``(key, owner, tier, backend)``. ``owner`` is the
# reporting instance for private storage and :data:`SHARED_OWNER` for a
# shared pool.
_PlacementId = tuple[ObjectKey, str, Tier, str]

# One compartment: ``(owner, tier, backend)``.
_CompartmentId = tuple[str, Tier, str]


class MemoryUsageTracker:
    """Thread-safe per-``(instance, tier, backend)`` byte usage view.

    A :class:`~lmcache.v1.mp_coordinator.ingest.event_broadcaster.CacheEventConsumer`:
    :meth:`consume` and :meth:`fence_instance` mutate; :meth:`get_for_instance`,
    :meth:`get_shared`, and :meth:`get_instances` read.

    Per-placement sizes are remembered because ``DELETE`` entries carry no
    ``size_bytes``.
    """

    def __init__(self) -> None:
        """Initialize an empty view."""
        self._lock = threading.Lock()
        self._placement_sizes: dict[_PlacementId, int] = {}
        self._bytes_by_compartment: dict[_CompartmentId, int] = {}
        # L1 placements per instance, so a fence drops exactly what that
        # instance reported without scanning every placement.
        self._l1_placements: dict[str, set[_PlacementId]] = {}

    def consume(self, batch: CacheEventBatch) -> None:
        """Account one gate-admitted batch.

        ``STORE`` upserts a placement's bytes, ``DELETE`` removes them, and
        ``ACCESS`` is ignored -- it carries no placement identity or size.

        Args:
            batch: The admitted batch.
        """
        if batch.event_type == CacheEventType.ACCESS:
            return
        owner = SHARED_OWNER if batch.shared else batch.instance_id
        compartment: _CompartmentId = (owner, batch.tier, batch.backend)
        track_l1 = batch.tier == Tier.L1 and not batch.shared
        with self._lock:
            for entry in batch.entries:
                key = entry.key.to_object_key()
                placement_id: _PlacementId = (key, owner, batch.tier, batch.backend)
                if batch.event_type == CacheEventType.STORE:
                    previous = self._placement_sizes.get(placement_id, 0)
                    self._placement_sizes[placement_id] = entry.size_bytes
                    self._adjust_locked(compartment, entry.size_bytes - previous)
                    if track_l1:
                        self._l1_placements.setdefault(owner, set()).add(placement_id)
                else:
                    size = self._placement_sizes.pop(placement_id, 0)
                    self._adjust_locked(compartment, -size)
                    if track_l1:
                        tracked = self._l1_placements.get(owner)
                        if tracked is not None:
                            tracked.discard(placement_id)

    def fence_instance(self, instance_id: str) -> None:
        """Discard the L1 bytes ``instance_id`` reported.

        L1 dies with the reporting process; L2 outlives it and leaves only
        through ``DELETE``, so L2 bytes are kept.

        Args:
            instance_id: The instance whose reported L1 state is void.
        """
        with self._lock:
            for placement_id in self._l1_placements.pop(instance_id, set()):
                size = self._placement_sizes.pop(placement_id, 0)
                _key, owner, tier, backend = placement_id
                self._adjust_locked((owner, tier, backend), -size)

    def get_for_instance(self, instance_id: str) -> tuple[ModuleUsage, ...]:
        """Return the compartments ``instance_id`` privately owns.

        Args:
            instance_id: The server to report on.

        Returns:
            One :class:`ModuleUsage` per compartment holding bytes, sorted
            by ``(tier, backend)``. Shared pools are excluded; read them
            through :meth:`get_shared`.
        """
        with self._lock:
            found = [
                ModuleUsage(tier=tier, backend=backend, used_bytes=used, shared=False)
                for (owner, tier, backend), used in self._bytes_by_compartment.items()
                if owner == instance_id
            ]
        return tuple(sorted(found, key=lambda m: (m.tier.value, m.backend)))

    def get_shared(self) -> tuple[ModuleUsage, ...]:
        """Return the fleet-shared compartments.

        Returns:
            One :class:`ModuleUsage` per shared pool holding bytes, sorted
            by ``(tier, backend)``. Totals are fleet-wide and must not be
            summed across instances.
        """
        with self._lock:
            found = [
                ModuleUsage(tier=tier, backend=backend, used_bytes=used, shared=True)
                for (owner, tier, backend), used in self._bytes_by_compartment.items()
                if owner == SHARED_OWNER
            ]
        return tuple(sorted(found, key=lambda m: (m.tier.value, m.backend)))

    def get_instances(self) -> tuple[str, ...]:
        """Return the ids of instances currently holding bytes.

        Returns:
            Sorted instance ids. An instance holding no bytes is absent, so
            callers needing the full fleet should iterate the registry.
        """
        with self._lock:
            owners = {
                owner
                for owner, _tier, _backend in self._bytes_by_compartment
                if owner != SHARED_OWNER
            }
        return tuple(sorted(owners))

    def _adjust_locked(self, compartment: _CompartmentId, delta: int) -> None:
        """Apply ``delta`` bytes to one compartment's total.

        Args:
            compartment: The ``(owner, tier, backend)`` being adjusted.
            delta: Signed byte change; ``0`` is a no-op.
        """
        if delta == 0:
            return
        new_total = self._bytes_by_compartment.get(compartment, 0) + delta
        if new_total <= 0:
            if new_total < 0:
                owner, tier, backend = compartment
                logger.warning(
                    "Memory usage underflow for instance=%r %s/%s (delta %d); "
                    "clamping to 0",
                    owner,
                    tier.value,
                    backend,
                    delta,
                )
            self._bytes_by_compartment.pop(compartment, None)
        else:
            self._bytes_by_compartment[compartment] = new_total
