# SPDX-License-Identifier: Apache-2.0
"""Tests for the coordinator's fleet memory-pressure endpoints."""

# Standard
from typing import cast

# Third Party
from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

# First Party
from lmcache.v1.distributed.api import ObjectKey, Tier
from lmcache.v1.mp_coordinator.api import (
    CacheEventBatch,
    CacheEventEntry,
    CacheEventType,
)
from lmcache.v1.mp_coordinator.app import create_app
from lmcache.v1.mp_coordinator.config import MPCoordinatorConfig
from lmcache.v1.mp_coordinator.http_apis.dependencies import CoordinatorContext

GIB = 1 << 30


@pytest.fixture
def client() -> TestClient:
    """A coordinator with both background loops disabled."""
    app = create_app(
        MPCoordinatorConfig(health_check_interval=0, eviction_check_interval=0)
    )
    return TestClient(app)


def _ctx(client: TestClient) -> CoordinatorContext:
    """Return the coordinator context behind ``client``.

    ``TestClient.app`` is a bare ASGI callable, so the cast happens here once.

    Args:
        client: Test client wrapping a coordinator app.

    Returns:
        That app's :class:`CoordinatorContext`.
    """
    return cast("FastAPI", client.app).state.ctx


def _register(
    client: TestClient,
    instance_id: str,
    modules: list[dict[str, object]],
) -> None:
    """Register an instance declaring ``modules``."""
    response = client.post(
        "/instances",
        json={
            "instance_id": instance_id,
            "ip": "10.0.0.1",
            "http_port": 8000,
            "memory_modules": modules,
        },
    )
    assert response.status_code == 200


def _ingest(
    client: TestClient,
    instance_id: str,
    tier: Tier,
    backend: str,
    size_bytes: int,
    index: int,
    shared: bool = False,
    seq: int = 1,
) -> None:
    """Push one STORE batch through the ingest gate."""
    key = ObjectKey(
        chunk_hash=bytes([index]) * 32,
        model_name="model",
        kv_rank=0,
        cache_salt="tenant",
    )
    _ctx(client).event_gate.ingest(
        CacheEventBatch(
            instance_id=instance_id,
            incarnation=1,
            seq=seq,
            event_type=CacheEventType.STORE,
            tier=tier,
            backend=backend,
            shared=shared,
            ts=1.0,
            entries=[
                CacheEventEntry(key=key.to_encoded_object_key(), size_bytes=size_bytes)
            ],
        )
    )


def _module(body: dict, tier: str, backend: str) -> dict:
    """Pick one module out of an instance status body."""
    found = [
        m for m in body["modules"] if m["tier"] == tier and m["backend"] == backend
    ]
    assert len(found) == 1, f"expected one {tier}/{backend}, got {found}"
    return found[0]


class TestInstanceMemory:
    def test_joins_usage_to_declared_capacity(self, client: TestClient) -> None:
        _register(
            client,
            "mp-1",
            [{"tier": "l1", "backend": "dram", "capacity_bytes": 40 * GIB}],
        )
        _ingest(client, "mp-1", Tier.L1, "dram", 10 * GIB, index=1)

        body = client.get("/memory/mp-1").json()
        module = _module(body, "l1", "dram")
        assert module["used_bytes"] == 10 * GIB
        assert module["capacity_bytes"] == 40 * GIB
        assert module["usage_ratio"] == pytest.approx(0.25)
        assert body["registered"] is True
        assert body["declared_capacity"] is True

    def test_undeclared_capacity_reports_no_ratio(self, client: TestClient) -> None:
        # A null ratio means unknown, not empty.
        _register(client, "mp-1", [])
        _ingest(client, "mp-1", Tier.L2, "fs", 7 * GIB, index=1)

        body = client.get("/memory/mp-1").json()
        module = _module(body, "l2", "fs")
        assert module["used_bytes"] == 7 * GIB
        assert module["capacity_bytes"] == 0
        assert module["usage_ratio"] is None
        assert body["declared_capacity"] is False

    def test_declared_but_unused_module_is_reported_empty(
        self, client: TestClient
    ) -> None:
        # Declared but idle reads as 0%, not as unknown.
        _register(
            client,
            "mp-1",
            [{"tier": "l1", "backend": "dram", "capacity_bytes": 40 * GIB}],
        )
        module = _module(client.get("/memory/mp-1").json(), "l1", "dram")
        assert module["used_bytes"] == 0
        assert module["usage_ratio"] == pytest.approx(0.0)

    def test_ratio_above_one_is_not_clamped(self, client: TestClient) -> None:
        # Over-full signals a misconfigured cap; clamping would hide it.
        _register(
            client, "mp-1", [{"tier": "l1", "backend": "dram", "capacity_bytes": GIB}]
        )
        _ingest(client, "mp-1", Tier.L1, "dram", 3 * GIB, index=1)
        assert _module(client.get("/memory/mp-1").json(), "l1", "dram")[
            "usage_ratio"
        ] == pytest.approx(3.0)

    def test_unknown_instance_is_404(self, client: TestClient) -> None:
        assert client.get("/memory/nobody").status_code == 404

    def test_deregistered_instance_keeps_surviving_l2_bytes(
        self, client: TestClient
    ) -> None:
        _register(
            client,
            "mp-1",
            [{"tier": "l2", "backend": "fs", "capacity_bytes": 40 * GIB}],
        )
        _ingest(client, "mp-1", Tier.L2, "fs", 5 * GIB, index=1)
        assert client.delete("/instances/mp-1").status_code == 204

        body = client.get("/memory/mp-1").json()
        assert body["registered"] is False
        # Capacity went with the departed process; the bytes did not.
        assert body["declared_capacity"] is False
        module = _module(body, "l2", "fs")
        assert module["used_bytes"] == 5 * GIB
        assert module["usage_ratio"] is None


class TestFleetMemory:
    def test_lists_every_instance(self, client: TestClient) -> None:
        _register(
            client,
            "mp-1",
            [{"tier": "l1", "backend": "dram", "capacity_bytes": 40 * GIB}],
        )
        _register(
            client,
            "mp-2",
            [{"tier": "l1", "backend": "dram", "capacity_bytes": 80 * GIB}],
        )
        _ingest(client, "mp-1", Tier.L1, "dram", 10 * GIB, index=1)
        _ingest(client, "mp-2", Tier.L1, "dram", 60 * GIB, index=2)

        body = client.get("/memory").json()
        ratios = {
            entry["instance_id"]: _module(entry, "l1", "dram")["usage_ratio"]
            for entry in body["instances"]
        }
        assert ratios == {"mp-1": pytest.approx(0.25), "mp-2": pytest.approx(0.75)}

    def test_shared_pool_is_reported_once_not_per_mount(
        self, client: TestClient
    ) -> None:
        shared = {
            "tier": "l2",
            "backend": "s3",
            "capacity_bytes": 100 * GIB,
            "shared": True,
        }
        _register(client, "mp-1", [shared])
        _register(client, "mp-2", [shared])
        _ingest(client, "mp-1", Tier.L2, "s3", 25 * GIB, index=1, shared=True)
        _ingest(client, "mp-2", Tier.L2, "s3", 25 * GIB, index=1, shared=True, seq=1)

        body = client.get("/memory").json()
        assert len(body["shared_modules"]) == 1
        pool = body["shared_modules"][0]
        assert pool["used_bytes"] == 25 * GIB
        assert pool["usage_ratio"] == pytest.approx(0.25)
        # Counted once, never onto the mounting instances.
        for entry in body["instances"]:
            assert entry["modules"] == []

    def test_disagreeing_shared_capacity_reads_as_undeclared(
        self, client: TestClient
    ) -> None:
        # Picking either would make the answer depend on registration order.
        _register(
            client,
            "mp-1",
            [
                {
                    "tier": "l2",
                    "backend": "s3",
                    "capacity_bytes": 100 * GIB,
                    "shared": True,
                }
            ],
        )
        _register(
            client,
            "mp-2",
            [
                {
                    "tier": "l2",
                    "backend": "s3",
                    "capacity_bytes": 999 * GIB,
                    "shared": True,
                }
            ],
        )
        _ingest(client, "mp-1", Tier.L2, "s3", 25 * GIB, index=1, shared=True)

        pool = client.get("/memory").json()["shared_modules"][0]
        assert pool["capacity_bytes"] == 0
        assert pool["usage_ratio"] is None

    def test_empty_fleet(self, client: TestClient) -> None:
        assert client.get("/memory").json() == {
            "instances": [],
            "shared_modules": [],
        }


class TestRegistration:
    def test_re_registration_replaces_the_declaration(self, client: TestClient) -> None:
        # A server that dropped an adapter must not keep its old capacity.
        _register(
            client,
            "mp-1",
            [
                {"tier": "l1", "backend": "dram", "capacity_bytes": 40 * GIB},
                {"tier": "l2", "backend": "fs", "capacity_bytes": 90 * GIB},
            ],
        )
        _register(
            client,
            "mp-1",
            [{"tier": "l1", "backend": "dram", "capacity_bytes": 40 * GIB}],
        )
        backends = {
            (m["tier"], m["backend"])
            for m in client.get("/memory/mp-1").json()["modules"]
        }
        assert backends == {("l1", "dram")}

    def test_registration_without_modules_is_accepted(self, client: TestClient) -> None:
        # Backward compatibility with servers predating this field.
        response = client.post(
            "/instances",
            json={"instance_id": "mp-1", "ip": "10.0.0.1", "http_port": 8000},
        )
        assert response.status_code == 200
        assert client.get("/memory/mp-1").json()["declared_capacity"] is False

    def test_tier_all_is_rejected(self, client: TestClient) -> None:
        response = client.post(
            "/instances",
            json={
                "instance_id": "mp-1",
                "ip": "10.0.0.1",
                "http_port": 8000,
                "memory_modules": [
                    {"tier": "all", "backend": "dram", "capacity_bytes": GIB}
                ],
            },
        )
        assert response.status_code == 422

    def test_duplicate_compartment_is_rejected(self, client: TestClient) -> None:
        with pytest.raises(ValueError, match="duplicate capacity declaration"):
            client.post(
                "/instances",
                json={
                    "instance_id": "mp-1",
                    "ip": "10.0.0.1",
                    "http_port": 8000,
                    "memory_modules": [
                        {"tier": "l1", "backend": "dram", "capacity_bytes": GIB},
                        {"tier": "l1", "backend": "dram", "capacity_bytes": 2 * GIB},
                    ],
                },
            )
