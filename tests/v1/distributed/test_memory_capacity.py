# SPDX-License-Identifier: Apache-2.0
"""Tests for the MP server's memory-capacity declaration.

Capacity is what the coordinator cannot derive from cache events, so these
cover the two things that make the declaration trustworthy: that it reports
the *configured* size rather than a lazily grown heap, and that a tier
spanning several mediums reports one compartment per medium.
"""

# Standard
import threading

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.api import L1BackendType, Tier
from lmcache.v1.distributed.memory_manager.l1_memory_manager import L1MemoryManager

GIB = 1 << 30


class _FakeAdapterConfig:
    """Stands in for an ``L2AdapterConfigBase``."""

    def __init__(self, shared: bool) -> None:
        self.shared = shared


class _FakeDescriptor:
    """Stands in for an ``AdapterDescriptor``."""

    def __init__(self, type_name: str, shared: bool) -> None:
        self.type_name = type_name
        self.config = _FakeAdapterConfig(shared)


class _FakeUsage:
    """Stands in for an ``AdapterUsage``."""

    def __init__(self, capacity_bytes: int) -> None:
        self.total_capacity_bytes = capacity_bytes


class _FakeAdapter:
    """An L2 adapter that reports a fixed capacity, or raises."""

    def __init__(self, capacity_bytes: int, fail: bool = False) -> None:
        self._capacity_bytes = capacity_bytes
        self._fail = fail

    def get_usage(self) -> _FakeUsage:
        if self._fail:
            raise RuntimeError("adapter unavailable")
        return _FakeUsage(self._capacity_bytes)


class _FakeL1Manager:
    """An L1 manager reporting a fixed per-medium capacity."""

    def __init__(self, capacities: dict[L1BackendType, int]) -> None:
        self._capacities = capacities

    def get_configured_capacity_bytes(self) -> dict[L1BackendType, int]:
        return self._capacities


def _capacities(l1: dict, adapters: list) -> list:
    """Run ``StorageManager.get_memory_capacities`` against fakes.

    Bound as an unbound function so the real ``StorageManager.__init__`` --
    which allocates pinned memory and spawns threads -- is not needed.
    """
    # First Party
    from lmcache.v1.distributed.storage_manager import StorageManager

    class _Stub:
        pass

    stub = _Stub()
    stub._l1_manager = _FakeL1Manager(l1)
    stub._snapshot_adapters = lambda: [
        (index, desc, adapter) for index, (desc, adapter) in enumerate(adapters)
    ]
    return StorageManager.get_memory_capacities(stub)


class TestL1ConfiguredCapacity:
    def test_cpu_tier_reports_configured_size_not_grown_heap(self) -> None:
        # The lazy allocator starts small and grows; reporting its current
        # heap as the denominator would make a fresh server read ~100% full.
        manager = L1MemoryManager.__new__(L1MemoryManager)
        manager._size_in_bytes = 40 * GIB
        assert manager.get_configured_capacity_bytes() == {L1BackendType.DRAM: 40 * GIB}

    def test_unconfigured_tier_reports_nothing_rather_than_zero(self) -> None:
        manager = L1MemoryManager.__new__(L1MemoryManager)
        manager._size_in_bytes = 0
        assert manager.get_configured_capacity_bytes() == {}


class TestStorageManagerCapacities:
    def test_reports_l1_per_medium(self) -> None:
        # A hybrid Device-DAX tier spans two mediums, and cache events tag
        # L1 placements per medium, so capacity must match that shape.
        found = _capacities(
            {L1BackendType.DEVDAX: 100 * GIB, L1BackendType.DRAM: 10 * GIB}, []
        )
        assert {(c.tier, c.backend, c.capacity_bytes) for c in found} == {
            (Tier.L1, "devdax", 100 * GIB),
            (Tier.L1, "dram", 10 * GIB),
        }
        assert all(c.shared is False for c in found)

    def test_reports_each_l2_adapter_with_its_shared_flag(self) -> None:
        found = _capacities(
            {L1BackendType.DRAM: 40 * GIB},
            [
                (_FakeDescriptor("fs", shared=False), _FakeAdapter(200 * GIB)),
                (_FakeDescriptor("s3", shared=True), _FakeAdapter(4000 * GIB)),
            ],
        )
        by_backend = {c.backend: c for c in found}
        assert by_backend["fs"].tier == Tier.L2
        assert by_backend["fs"].shared is False
        assert by_backend["s3"].shared is True
        assert by_backend["s3"].capacity_bytes == 4000 * GIB

    def test_adapter_without_a_configured_cap_reports_zero(self) -> None:
        # fs / mooncake / p2p / sagemaker return 0 unconditionally. Zero
        # means unknown downstream, never "full".
        found = _capacities(
            {}, [(_FakeDescriptor("fs", shared=False), _FakeAdapter(0))]
        )
        assert [c.capacity_bytes for c in found] == [0]

    def test_failing_adapter_is_omitted_not_reported_wrong(self) -> None:
        found = _capacities(
            {},
            [
                (_FakeDescriptor("fs", shared=False), _FakeAdapter(0, fail=True)),
                (_FakeDescriptor("s3", shared=False), _FakeAdapter(9 * GIB)),
            ],
        )
        assert [c.backend for c in found] == ["s3"]

    def test_server_with_nothing_configured_declares_nothing(self) -> None:
        assert _capacities({}, []) == []


@pytest.mark.parametrize(
    "capacities",
    [
        {L1BackendType.DRAM: 40 * GIB},
        {L1BackendType.GDS: 8 * GIB},
        {L1BackendType.DEVDAX: 100 * GIB, L1BackendType.DRAM: 10 * GIB},
    ],
)
def test_backend_names_match_the_cache_event_vocabulary(capacities: dict) -> None:
    # Capacity joins usage on (tier, backend), so these strings must be the
    # same ones L1 cache events carry.
    found = _capacities(capacities, [])
    assert {c.backend for c in found} == {b.value for b in capacities}


class TestReportStatusSharesTheSource:
    """``report_status`` and the capacity API must not drift apart.

    ``lmcache describe`` reads the status dict; the coordinator reads the
    capacity API. Both must resolve to the same declared size, or the CLI
    and the fleet view will disagree about the same server.
    """

    def _l1_manager(self, configured: dict[L1BackendType, int]) -> object:
        # First Party
        from lmcache.v1.distributed.l1_manager import L1Manager

        class _MemoryManager:
            def get_memory_usage(self) -> tuple[int, int]:
                # Deliberately unequal to the configured total: this is the
                # lazily grown heap, which is what the old field reported.
                return (1 * GIB, 3 * GIB)

            def get_configured_capacity_bytes(self) -> dict[L1BackendType, int]:
                return configured

            def memcheck(self) -> bool:
                return True

        manager = L1Manager.__new__(L1Manager)
        manager._memory_manager = _MemoryManager()
        manager._objects = {}
        manager._write_ttl_seconds = 600.0
        manager._read_ttl_seconds = 600.0
        # report_status is lock-guarded; the real __init__ is skipped here
        # because it allocates pinned memory.
        manager._lock = threading.Lock()
        return manager

    def test_status_reports_configured_separately_from_grown_heap(self) -> None:
        manager = self._l1_manager({L1BackendType.DRAM: 40 * GIB})
        status = manager.report_status()
        assert status["memory_configured_bytes"] == 40 * GIB
        # The pre-existing field keeps its old meaning for old consumers.
        assert status["memory_total_bytes"] == 3 * GIB

    def test_status_sums_a_hybrid_tier(self) -> None:
        manager = self._l1_manager(
            {L1BackendType.DEVDAX: 100 * GIB, L1BackendType.DRAM: 10 * GIB}
        )
        assert manager.report_status()["memory_configured_bytes"] == 110 * GIB

    def test_status_and_capacity_api_agree(self) -> None:
        configured = {L1BackendType.DEVDAX: 100 * GIB, L1BackendType.DRAM: 10 * GIB}
        manager = self._l1_manager(configured)
        assert manager.report_status()["memory_configured_bytes"] == sum(
            manager.get_configured_capacity_bytes().values()
        )

    def test_unconfigured_tier_reports_zero_not_the_heap(self) -> None:
        manager = self._l1_manager({})
        assert manager.report_status()["memory_configured_bytes"] == 0
