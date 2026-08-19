# SPDX-License-Identifier: Apache-2.0
"""Tests for the per-server, per-module memory usage view."""

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.api import ObjectKey, Tier
from lmcache.v1.mp_coordinator.api import (
    CacheEventBatch,
    CacheEventEntry,
    CacheEventType,
)
from lmcache.v1.mp_coordinator.controllers.memory_usage import MemoryUsageTracker


def _key(index: int) -> ObjectKey:
    """Build a distinct object key."""
    return ObjectKey(
        chunk_hash=bytes([index]) * 32,
        model_name="model",
        kv_rank=0,
        cache_salt="tenant",
    )


def _batch(
    seq: int,
    tier: Tier,
    backend: str,
    event_type: CacheEventType,
    entries: list[CacheEventEntry],
    shared: bool = False,
    instance_id: str = "mp-1",
    incarnation: int = 1,
) -> CacheEventBatch:
    """Build one cache-event batch."""
    return CacheEventBatch(
        instance_id=instance_id,
        incarnation=incarnation,
        seq=seq,
        event_type=event_type,
        tier=tier,
        backend=backend,
        shared=shared,
        ts=1.0,
        entries=entries,
    )


def _store(index: int, size_bytes: int) -> list[CacheEventEntry]:
    """Build a single-entry STORE payload."""
    return [
        CacheEventEntry(key=_key(index).to_encoded_object_key(), size_bytes=size_bytes)
    ]


def _sizeless(index: int) -> list[CacheEventEntry]:
    """Build a single-entry DELETE/ACCESS payload, which carries no size."""
    return [CacheEventEntry(key=_key(index).to_encoded_object_key())]


def _used(tracker: MemoryUsageTracker, instance_id: str) -> dict[tuple[str, str], int]:
    """Return ``{(tier, backend): used_bytes}`` for one instance."""
    return {
        (m.tier.value, m.backend): m.used_bytes
        for m in tracker.get_for_instance(instance_id)
    }


class TestConsume:
    def test_tracks_both_tiers_separately(self) -> None:
        tracker = MemoryUsageTracker()
        tracker.consume(
            _batch(1, Tier.L1, "dram", CacheEventType.STORE, _store(1, 100))
        )
        tracker.consume(_batch(2, Tier.L2, "fs", CacheEventType.STORE, _store(2, 500)))
        assert _used(tracker, "mp-1") == {("l1", "dram"): 100, ("l2", "fs"): 500}

    def test_same_backend_different_instances_do_not_mix(self) -> None:
        tracker = MemoryUsageTracker()
        tracker.consume(
            _batch(1, Tier.L1, "dram", CacheEventType.STORE, _store(1, 100))
        )
        tracker.consume(
            _batch(
                1,
                Tier.L1,
                "dram",
                CacheEventType.STORE,
                _store(2, 700),
                instance_id="mp-2",
            )
        )
        assert _used(tracker, "mp-1") == {("l1", "dram"): 100}
        assert _used(tracker, "mp-2") == {("l1", "dram"): 700}
        assert tracker.get_instances() == ("mp-1", "mp-2")

    def test_restore_applies_a_delta_not_a_second_charge(self) -> None:
        tracker = MemoryUsageTracker()
        tracker.consume(
            _batch(1, Tier.L1, "dram", CacheEventType.STORE, _store(1, 100))
        )
        tracker.consume(
            _batch(2, Tier.L1, "dram", CacheEventType.STORE, _store(1, 250))
        )
        assert _used(tracker, "mp-1") == {("l1", "dram"): 250}

    def test_delete_releases_the_remembered_size(self) -> None:
        # DELETE entries carry no size_bytes, so the tracker must have
        # remembered what the STORE admitted.
        tracker = MemoryUsageTracker()
        tracker.consume(
            _batch(1, Tier.L1, "dram", CacheEventType.STORE, _store(1, 100))
        )
        tracker.consume(_batch(2, Tier.L1, "dram", CacheEventType.DELETE, _sizeless(1)))
        assert _used(tracker, "mp-1") == {}

    def test_delete_of_untracked_key_is_a_noop(self) -> None:
        tracker = MemoryUsageTracker()
        tracker.consume(_batch(1, Tier.L1, "dram", CacheEventType.DELETE, _sizeless(9)))
        assert _used(tracker, "mp-1") == {}

    def test_access_does_not_change_bytes(self) -> None:
        tracker = MemoryUsageTracker()
        tracker.consume(_batch(1, Tier.L2, "fs", CacheEventType.STORE, _store(1, 400)))
        tracker.consume(_batch(2, Tier.L2, "fs", CacheEventType.ACCESS, _sizeless(1)))
        assert _used(tracker, "mp-1") == {("l2", "fs"): 400}


class TestSharedPools:
    def test_shared_bytes_are_not_attributed_to_the_reporter(self) -> None:
        tracker = MemoryUsageTracker()
        tracker.consume(
            _batch(1, Tier.L2, "s3", CacheEventType.STORE, _store(1, 900), shared=True)
        )
        assert _used(tracker, "mp-1") == {}
        assert [(m.backend, m.used_bytes) for m in tracker.get_shared()] == [
            ("s3", 900)
        ]

    def test_two_instances_reporting_one_pool_count_it_once(self) -> None:
        # The whole point of the shared flag: one bucket mounted twice is
        # still one bucket. Summing per reporter would double it.
        tracker = MemoryUsageTracker()
        tracker.consume(
            _batch(1, Tier.L2, "s3", CacheEventType.STORE, _store(1, 900), shared=True)
        )
        tracker.consume(
            _batch(
                1,
                Tier.L2,
                "s3",
                CacheEventType.STORE,
                _store(1, 900),
                shared=True,
                instance_id="mp-2",
            )
        )
        assert [m.used_bytes for m in tracker.get_shared()] == [900]

    def test_shared_owner_is_not_reported_as_an_instance(self) -> None:
        tracker = MemoryUsageTracker()
        tracker.consume(
            _batch(1, Tier.L2, "s3", CacheEventType.STORE, _store(1, 900), shared=True)
        )
        assert tracker.get_instances() == ()


class TestFenceInstance:
    def test_drops_l1_and_keeps_l2(self) -> None:
        # L1 lives in the reporting process and dies with it; L2 outlives
        # the reporter and leaves only through DELETE.
        tracker = MemoryUsageTracker()
        tracker.consume(
            _batch(1, Tier.L1, "dram", CacheEventType.STORE, _store(1, 100))
        )
        tracker.consume(_batch(2, Tier.L2, "fs", CacheEventType.STORE, _store(2, 500)))
        tracker.fence_instance("mp-1")
        assert _used(tracker, "mp-1") == {("l2", "fs"): 500}

    def test_does_not_touch_other_instances(self) -> None:
        tracker = MemoryUsageTracker()
        tracker.consume(
            _batch(1, Tier.L1, "dram", CacheEventType.STORE, _store(1, 100))
        )
        tracker.consume(
            _batch(
                1,
                Tier.L1,
                "dram",
                CacheEventType.STORE,
                _store(2, 700),
                instance_id="mp-2",
            )
        )
        tracker.fence_instance("mp-1")
        assert _used(tracker, "mp-1") == {}
        assert _used(tracker, "mp-2") == {("l1", "dram"): 700}

    def test_does_not_drop_shared_pools(self) -> None:
        tracker = MemoryUsageTracker()
        tracker.consume(
            _batch(1, Tier.L2, "s3", CacheEventType.STORE, _store(1, 900), shared=True)
        )
        tracker.fence_instance("mp-1")
        assert [m.used_bytes for m in tracker.get_shared()] == [900]

    def test_is_idempotent_and_safe_for_unknown_instances(self) -> None:
        tracker = MemoryUsageTracker()
        tracker.consume(
            _batch(1, Tier.L1, "dram", CacheEventType.STORE, _store(1, 100))
        )
        tracker.fence_instance("mp-1")
        tracker.fence_instance("mp-1")
        tracker.fence_instance("never-seen")
        assert _used(tracker, "mp-1") == {}

    def test_refilling_after_a_fence_tracks_normally(self) -> None:
        tracker = MemoryUsageTracker()
        tracker.consume(
            _batch(1, Tier.L1, "dram", CacheEventType.STORE, _store(1, 100))
        )
        tracker.fence_instance("mp-1")
        tracker.consume(
            _batch(
                1,
                Tier.L1,
                "dram",
                CacheEventType.STORE,
                _store(1, 300),
                incarnation=2,
            )
        )
        assert _used(tracker, "mp-1") == {("l1", "dram"): 300}


class TestReads:
    def test_unknown_instance_reads_empty(self) -> None:
        assert MemoryUsageTracker().get_for_instance("nobody") == ()

    def test_modules_are_sorted_by_tier_then_backend(self) -> None:
        tracker = MemoryUsageTracker()
        for seq, (tier, backend) in enumerate(
            [(Tier.L2, "s3"), (Tier.L1, "dram"), (Tier.L2, "fs")], start=1
        ):
            tracker.consume(
                _batch(seq, tier, backend, CacheEventType.STORE, _store(seq, 10))
            )
        assert [
            (m.tier.value, m.backend) for m in tracker.get_for_instance("mp-1")
        ] == [
            ("l1", "dram"),
            ("l2", "fs"),
            ("l2", "s3"),
        ]


@pytest.mark.parametrize("tier", [Tier.L1, Tier.L2])
def test_compartment_disappears_when_fully_freed(tier: Tier) -> None:
    tracker = MemoryUsageTracker()
    tracker.consume(_batch(1, tier, "b", CacheEventType.STORE, _store(1, 100)))
    tracker.consume(_batch(2, tier, "b", CacheEventType.DELETE, _sizeless(1)))
    assert tracker.get_for_instance("mp-1") == ()
