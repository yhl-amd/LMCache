# SPDX-License-Identifier: Apache-2.0
"""Structural interface shared by the L1 memory manager tiers."""

# Standard
from typing import Optional, Protocol, runtime_checkable

# First Party
from lmcache.v1.distributed.api import L1BackendType, MemoryLayoutDesc
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.internal_api import L1MemoryDesc
from lmcache.v1.memory_management import MemoryObj


@runtime_checkable
class L1ManagerProtocol(Protocol):
    """Structural interface for an L1 memory manager.

    Both :class:`L1MemoryManager` (CPU pinned-DRAM tier) and
    :class:`GDSL1MemoryManager` (GDS slab-file tier) satisfy this protocol, so
    ``L1Manager`` can hold either behind one type.
    """

    def allocate(
        self, layout_desc: MemoryLayoutDesc, count: int
    ) -> tuple[L1Error, list[MemoryObj]]:
        """Allocate ``count`` memory objects for the given layout."""
        ...

    def free(self, mem_objs: list[MemoryObj]) -> L1Error:
        """Free the given memory objects."""
        ...

    def get_memory_usage(self) -> tuple[int, int]:
        """Return ``(used_bytes, total_bytes)``."""
        ...

    def get_configured_capacity_bytes(self) -> dict[L1BackendType, int]:
        """Return the configured capacity of each backing medium.

        Distinct from ``get_memory_usage()[1]``, which some allocators
        report as the *currently grown* heap rather than the configured
        cap -- a freshly booted lazy tier would otherwise look full. This
        is the operator-declared size, so it is stable from boot and is
        the correct denominator for an occupancy figure.

        Keyed per medium because one L1 tier can span several: a hybrid
        Device-DAX tier backs objects with both ``devdax`` and ``dram``,
        and cache events tag L1 placements the same way, so capacity and
        usage line up on the same compartments.

        Returns:
            Configured bytes per medium; mediums with no capacity are
            omitted rather than reported as zero.
        """
        ...

    def get_backend_type(self, memory_obj: MemoryObj) -> L1BackendType:
        """Return the storage medium backing ``memory_obj``."""
        ...

    def get_l1_memory_desc(self) -> Optional[L1MemoryDesc]:
        """Describe the underlying L1 buffer for L2-adapter registration.

        Returns ``None`` for tiers with no registerable L1 buffer (e.g. GDS).
        """
        ...

    def close(self) -> None:
        """Release all resources."""
        ...

    def memcheck(self) -> bool:
        """Verify allocator bookkeeping consistency."""
        ...
