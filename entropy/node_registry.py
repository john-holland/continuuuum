"""Node registry for entropy ring: active nodes, warehouse (mezzed), persistence, and lifecycle logging."""
from __future__ import annotations

import secrets
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from unified_semantic_archiver.db import ContinuumDb


def _iso_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class NodeRegistry:
    """Registry of active and warehouse (mezzed) nodes with persistence and event logging."""

    def __init__(self, db: ContinuumDb):
        self._db = db

    def add_node(
        self,
        *,
        node_id: str | None = None,
        probe_target: str,
        tenant_id: str = "default",
    ) -> str:
        """Add a node to the active ring. Returns the node_id (generated if not provided)."""
        nid = node_id or f"node-{secrets.token_hex(8)}"
        self._db.entropy_ring_node_insert(
            node_id=nid,
            probe_target=probe_target,
            tenant_id=tenant_id or "default",
            status="active",
        )
        self._db.entropy_event_insert("added", nid)
        return nid

    def remove_node(self, node_id: str) -> None:
        """Remove a node from the ring (and warehouse if present)."""
        row = self._db.entropy_ring_node_get(node_id)
        if row:
            self._db.entropy_ring_node_update_status(node_id, "removed")
            self._db.entropy_event_insert("removed", node_id)
            self._db.entropy_ring_node_delete(node_id)
        self._db.entropy_warehouse_delete(node_id)

    def mark_mezzed(self, node_id: str) -> None:
        """Move node from active to warehouse (mezzed)."""
        row = self._db.entropy_ring_node_get(node_id)
        if not row or row.get("status") != "active":
            return
        probe_target = row["probe_target"]
        tenant_id = row.get("tenant_id") or "default"
        self._db.entropy_ring_node_delete(node_id)
        self._db.entropy_warehouse_insert(
            node_id=node_id,
            probe_target=probe_target,
            tenant_id=tenant_id,
        )
        self._db.entropy_event_insert("mezzed", node_id)

    def mark_refreshed(
        self,
        node_id: str,
        probe_target: str | None = None,
        tenant_id: str | None = None,
    ) -> None:
        """Move node from warehouse back to active (refreshed)."""
        wh = self.get_warehouse_node(node_id)
        if wh:
            probe_target = probe_target or wh.get("probe_target", "")
            tenant_id = tenant_id or wh.get("tenant_id", "default")
        else:
            probe_target = probe_target or ""
            tenant_id = tenant_id or "default"
        self._db.entropy_warehouse_delete(node_id)
        self._db.entropy_ring_node_insert(
            node_id=node_id,
            probe_target=probe_target,
            tenant_id=tenant_id or "default",
            status="active",
        )
        self._db.entropy_event_insert("refreshed", node_id)

    def list_active(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
        """List active ring nodes."""
        return self._db.entropy_ring_node_list(status="active", tenant_id=tenant_id)

    def list_warehouse(self, tenant_id: str | None = None) -> list[dict[str, Any]]:
        """List warehouse (mezzed) nodes."""
        return self._db.entropy_warehouse_list(tenant_id=tenant_id)

    def get_node(self, node_id: str) -> dict[str, Any] | None:
        """Get node from active ring."""
        return self._db.entropy_ring_node_get(node_id)

    def get_warehouse_node(self, node_id: str) -> dict[str, Any] | None:
        """Get node from warehouse by node_id."""
        rows = self._db.entropy_warehouse_list()
        for r in rows:
            if r.get("node_id") == node_id:
                return dict(r)
        return None

    def update_last_seen(self, node_id: str) -> None:
        """Update last_seen timestamp for node."""
        self._db.entropy_ring_node_update_status(
            node_id,
            "active",
            last_seen=_iso_utc(),
        )

    def warehouse_record_retry(self, node_id: str) -> None:
        """Record a retry attempt for a warehouse node."""
        self._db.entropy_warehouse_update_retry(node_id)
