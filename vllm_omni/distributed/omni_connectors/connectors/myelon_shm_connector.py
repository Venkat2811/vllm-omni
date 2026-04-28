# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Ring-buffer-backed shared-memory connector for vllm-omni.

Replaces SharedMemoryConnector's per-put `fcntl.flock` + new `/dev/shm/<key>`
inode pattern with a single persistent POSIX-SHM ring (Myelon's
`MyelonShmObjectStorage`). Each put is a ring `allocate` + memcpy; each get
is a ring read by `(address, monotonic_id)` handle. No per-put filesystem
syscalls, no per-key inode creation.

Connector role is decided lazily on first call:
  - The first `put()` on this instance creates the writer-side ring.
  - The first `get()` on this instance attaches as a reader (open by name).
  - In normal stage-split deployments only one role is exercised per
    process, so the unused role's ring is never opened.

Config keys consumed from `spec.extra` (all optional; sane defaults shown):
  ``ring_name``         — POSIX SHM segment name. Must match across stages
                          that share this ring. Default ``"/myelon-omni-shm"``.
  ``ring_bytes``        — Total ring capacity in bytes. Default 256 MiB.
  ``max_object_size``   — Largest single payload. Default 64 MiB.
  ``n_readers``         — Reader-rank count for slot reclamation. Default 1.
"""

from __future__ import annotations

import os
from typing import Any

from ..utils.logging import get_connector_logger
from .base import OmniConnectorBase

logger = get_connector_logger(__name__)


class MyelonShmConnector(OmniConnectorBase):
    """Drop-in replacement for SharedMemoryConnector that uses a Myelon ring."""

    # OmniSerializer handles bytes ↔ Python object; we don't bypass it.
    supports_raw_data = False

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.stage_id = config.get("stage_id", -1)
        self.device = config.get("device", "cuda:0")
        self.ring_name = config.get("ring_name") or f"/myelon-omni-{os.getpid()}"
        self.ring_bytes = int(config.get("ring_bytes", 256 * 1024 * 1024))
        self.max_object_size = int(config.get("max_object_size", 64 * 1024 * 1024))
        self.n_readers = int(config.get("n_readers", 1))

        self._writer: Any = None
        self._reader: Any = None
        self._pending_keys: set[str] = set()
        self._metrics: dict[str, int] = {
            "puts": 0,
            "gets": 0,
            "bytes_transferred": 0,
            "ring_writes": 0,
            "ring_reads": 0,
        }

    # ------------------------------------------------------------------ #
    # Lazy role attachment.
    # ------------------------------------------------------------------ #

    def _import_storage(self):
        try:
            from myelon_objstore import MyelonShmObjectStorage  # type: ignore
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "MyelonShmConnector requires the `myelon_objstore` PyO3 binding. "
                "Build it from `myelon-playground/crates/myelon-playground-py` "
                "(per RFC 0035 §5)."
            ) from e
        return MyelonShmObjectStorage

    def _ensure_writer(self) -> None:
        if self._writer is not None:
            return
        cls = self._import_storage()
        self._writer = cls.create_writer(
            max_object_size=self.max_object_size,
            n_readers=self.n_readers,
            ring_bytes=self.ring_bytes,
            name=self.ring_name,
        )
        logger.info(
            "MyelonShmConnector attached as writer (ring=%s, %d MiB, "
            "max_obj=%d MiB, n_readers=%d)",
            self.ring_name,
            self.ring_bytes // (1 << 20),
            self.max_object_size // (1 << 20),
            self.n_readers,
        )

    def _ensure_reader(self) -> None:
        if self._reader is not None:
            return
        cls = self._import_storage()
        self._reader = cls.open_reader(
            name=self.ring_name,
            ring_bytes=self.ring_bytes,
            n_readers=self.n_readers,
        )
        logger.info(
            "MyelonShmConnector attached as reader (ring=%s)", self.ring_name
        )

    # ------------------------------------------------------------------ #
    # OmniConnectorBase API.
    # ------------------------------------------------------------------ #

    def put(
        self,
        from_stage: str,
        to_stage: str,
        put_key: str,
        data: Any,
    ) -> tuple[bool, int, dict[str, Any] | None]:
        try:
            self._ensure_writer()
            payload = self.serialize_obj(data)
            size = len(payload)
            address, monotonic_id = self._writer.put(put_key, payload)
            metadata = {
                "myelon": {
                    "ring_name": self.ring_name,
                    "address": int(address),
                    "monotonic_id": int(monotonic_id),
                    "ring_bytes": self.ring_bytes,
                    "n_readers": self.n_readers,
                },
                "size": size,
            }
            self._pending_keys.add(put_key)
            self._metrics["puts"] += 1
            self._metrics["ring_writes"] += 1
            self._metrics["bytes_transferred"] += size
            return True, size, metadata
        except Exception as e:  # noqa: BLE001
            logger.error("MyelonShmConnector put failed for req %s: %s", put_key, e)
            return False, 0, None

    def get(
        self,
        from_stage: str,
        to_stage: str,
        get_key: str,
        metadata: dict[str, Any] | None = None,
    ) -> tuple[Any, int] | None:
        try:
            # Fast path: address+id supplied via put-side metadata.
            if metadata and "myelon" in metadata:
                handle = metadata["myelon"]
                # If the put was made by another process to a different ring,
                # update our ring identity before opening the reader.
                ring_name = handle.get("ring_name")
                if ring_name and ring_name != self.ring_name:
                    self.ring_name = ring_name
                    self.ring_bytes = int(handle.get("ring_bytes", self.ring_bytes))
                    self.n_readers = int(handle.get("n_readers", self.n_readers))
                self._ensure_reader()
                data_bytes = self._reader.get(
                    int(handle["address"]),
                    int(handle["monotonic_id"]),
                )
                size = int(metadata.get("size", len(data_bytes)))
                obj = self.deserialize_obj(bytes(data_bytes))
                self._metrics["gets"] += 1
                self._metrics["ring_reads"] += 1
                self._metrics["bytes_transferred"] += size
                self._pending_keys.discard(get_key)
                return obj, size

            # Slow path: by-key lookup. Only the writer side maintains the
            # key index; readers do not. Mirrors SharedMemoryConnector's
            # in-process key fallback.
            if self._writer is not None and self._writer.is_cached(get_key):
                address, monotonic_id = self._writer.get_cached(get_key)
                data_bytes = self._writer.get(int(address), int(monotonic_id))
                obj = self.deserialize_obj(bytes(data_bytes))
                size = len(data_bytes)
                self._metrics["gets"] += 1
                self._metrics["ring_reads"] += 1
                self._metrics["bytes_transferred"] += size
                self._pending_keys.discard(get_key)
                return obj, size

            return None
        except Exception as e:  # noqa: BLE001
            logger.error("MyelonShmConnector get failed for req %s: %s", get_key, e)
            return None

    def cleanup(self, request_id: str) -> None:
        # Ring slots are auto-reclaimed via `mark_consumed` once `n_readers`
        # have read each slot. No per-request inode/file teardown needed
        # (unlike SharedMemoryConnector which removes a /dev/shm/<key> file).
        self._pending_keys.discard(request_id)

    def health(self) -> dict[str, Any]:
        return {
            "name": "MyelonShmConnector",
            "ring_name": self.ring_name,
            "ring_bytes": self.ring_bytes,
            "max_object_size": self.max_object_size,
            "n_readers": self.n_readers,
            "pending_keys": len(self._pending_keys),
            "writer_attached": self._writer is not None,
            "reader_attached": self._reader is not None,
            **self._metrics,
        }

    def close(self) -> None:
        if self._writer is not None:
            try:
                self._writer.close()
            except Exception:  # noqa: BLE001
                logger.debug("Writer close raised", exc_info=True)
            self._writer = None
        if self._reader is not None:
            try:
                self._reader.close()
            except Exception:  # noqa: BLE001
                logger.debug("Reader close raised", exc_info=True)
            self._reader = None
