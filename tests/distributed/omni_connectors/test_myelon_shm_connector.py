# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""API-conformance tests for MyelonShmConnector.

Mirrors the public-API tests in test_shm_connector.py but against the
ring-buffer-backed Myelon implementation. Skipped if `myelon_objstore`
is not built/installed.
"""
import os

import pytest

myelon_objstore = pytest.importorskip(
    "myelon_objstore",
    reason="myelon_objstore PyO3 binding not installed",
)
from vllm_omni.distributed.omni_connectors.connectors.myelon_shm_connector import (  # noqa: E402
    MyelonShmConnector,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.fixture()
def connector():
    c = MyelonShmConnector(
        {
            "stage_id": 0,
            "ring_name": f"/myelon-test-{os.getpid()}",
            "ring_bytes": 16 * 1024 * 1024,
            "max_object_size": 1024 * 1024,
            "n_readers": 1,
        }
    )
    yield c
    c.close()


class TestKeyBasedReadWrite:
    def test_put_then_get_by_key(self, connector):
        data = {"hello": "world", "n": 42}
        ok, size, meta = connector.put("s0", "s1", "test_key_1", data)
        assert ok
        assert size > 0
        assert "myelon" in meta
        assert "test_key_1" in connector._pending_keys

        # Use the metadata path (production usage) — Myelon doesn't
        # support pure by-key reads from a separate process; keys are
        # only writer-side. Same-process get with metadata=None falls
        # through to the writer's key index.
        result = connector.get("s0", "s1", "test_key_1", metadata=meta)
        assert result is not None
        obj, rsize = result
        assert obj == data
        assert rsize == size

    def test_get_nonexistent_key_returns_none(self, connector):
        # No put, no metadata → reader path returns None gracefully.
        result = connector.get("s0", "s1", "no_such_key_xyz", metadata=None)
        assert result is None

    def test_metadata_carries_address_id(self, connector):
        ok, _, meta = connector.put("s0", "s1", "k1", {"x": 1})
        assert ok
        m = meta["myelon"]
        assert "address" in m
        assert "monotonic_id" in m
        assert "ring_name" in m
        assert m["ring_name"].startswith("/myelon-test-")


class TestRoundTrip:
    def test_multiple_puts_and_gets(self, connector):
        items = []
        for i in range(8):
            data = {"i": i, "buf": b"x" * (1024 * (i + 1))}
            ok, _, meta = connector.put("s0", "s1", f"item-{i}", data)
            assert ok
            items.append((i, data, meta))

        # Read out of order to confirm random access by handle.
        for i, expected, meta in reversed(items):
            result = connector.get("s0", "s1", f"item-{i}", meta)
            assert result is not None
            obj, _ = result
            assert obj["i"] == i
            assert obj["buf"] == expected["buf"]

    def test_keyed_lookup_writer_side(self, connector):
        """In-process get with metadata=None falls through to writer key index."""
        data = {"v": "hello"}
        ok, _, _ = connector.put("s0", "s1", "fallback-key", data)
        assert ok
        result = connector.get("s0", "s1", "fallback-key", metadata=None)
        assert result is not None
        obj, _ = result
        assert obj == data


class TestLifecycle:
    def test_cleanup_clears_pending(self, connector):
        ok, _, _ = connector.put("s0", "s1", "to-cleanup", {"x": 1})
        assert ok
        assert "to-cleanup" in connector._pending_keys
        connector.cleanup("to-cleanup")
        assert "to-cleanup" not in connector._pending_keys

    def test_health_reports_metrics(self, connector):
        connector.put("s0", "s1", "h1", {"x": 1})
        h = connector.health()
        assert h["puts"] >= 1
        assert h["bytes_transferred"] > 0
        assert h["writer_attached"] is True
        assert h["name"] == "MyelonShmConnector"

    def test_close_releases_handles(self, connector):
        connector.put("s0", "s1", "c1", {"x": 1})
        # close idempotent: explicit close in fixture teardown should
        # not raise. We do not hold a reference here.
