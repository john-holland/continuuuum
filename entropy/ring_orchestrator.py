"""Ring orchestrator: fits ring for added/removed/mezzed nodes, warehouse rotation, run_round."""
from __future__ import annotations

import urllib.request
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Callable

from .center_target import CenterObjectTarget
from .node_registry import NodeRegistry

if TYPE_CHECKING:
    from unified_semantic_archiver.db import ContinuumDb


def _default_query_node(node_id: str, base_url: str, timeout: float = 2.0) -> bool:
    """Default health check: GET base_url/health for node. Returns True if reachable."""
    try:
        req = urllib.request.Request(f"{base_url.rstrip('/')}/health", method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False


class RingOrchestrator:
    """
    Queries each node, fits the ring for added/removed/mezzed nodes, drives sequential
    pass-along, keeps warehouse for mezzed nodes, persists and logs lifecycle.
    """

    def __init__(
        self,
        db: ContinuumDb,
        center: CenterObjectTarget | None = None,
        node_base_urls: dict[str, str] | None = None,
        query_node: Callable[[str, str], bool] | None = None,
        health_check_timeout_seconds: float = 2.0,
        staleness_seconds: float | None = None,
    ):
        self._db = db
        self._registry = NodeRegistry(db)
        self._center = center or CenterObjectTarget()
        self._node_base_urls = dict(node_base_urls or {})
        self._health_check_timeout = health_check_timeout_seconds
        self._staleness_seconds = staleness_seconds
        if query_node is not None:
            self._query_node = query_node
        else:
            self._query_node = lambda nid, url: _default_query_node(nid, url, self._health_check_timeout)
        self._ring_order: list[str] = []
        self._node_base_url_default = "http://127.0.0.1:0"

    def add_node(
        self,
        *,
        node_id: str | None = None,
        probe_target: str,
        tenant_id: str = "default",
        base_url: str | None = None,
    ) -> str:
        """Add node to registry and fit ring."""
        nid = self._registry.add_node(node_id=node_id, probe_target=probe_target, tenant_id=tenant_id)
        if base_url:
            self._node_base_urls[nid] = base_url
        self.fit_ring()
        return nid

    def remove_node(self, node_id: str) -> None:
        """Remove node and fit ring."""
        self._registry.remove_node(node_id)
        self._node_base_urls.pop(node_id, None)
        self.fit_ring()

    def fit_ring(self, tenant_id: str | None = None) -> list[str]:
        """Recompute ring order from active registry. Returns ordered node_ids."""
        active = self._registry.list_active(tenant_id=tenant_id)
        self._ring_order = [r["node_id"] for r in active]
        return self._ring_order.copy()

    def next_node_id(self, node_id: str) -> str | None:
        """Return the next node in the ring for sequential pass-along. Wraps to first if last."""
        if not self._ring_order:
            return None
        try:
            idx = self._ring_order.index(node_id)
            next_idx = (idx + 1) % len(self._ring_order)
            return self._ring_order[next_idx]
        except ValueError:
            return None

    def query_node(self, node_id: str) -> bool:
        """Health/telemetry query for node. Returns True if reachable."""
        url = self._node_base_urls.get(node_id, self._node_base_url_default)
        return self._query_node(node_id, url)

    def mark_mezzed(self, node_id: str) -> None:
        """Mark node as mezzed, move to warehouse, fit ring."""
        self._registry.mark_mezzed(node_id)
        self.fit_ring()

    def mark_refreshed(self, node_id: str, **kwargs: Any) -> None:
        """Move node from warehouse back to active, fit ring."""
        self._registry.mark_refreshed(node_id, **kwargs)
        self.fit_ring()

    def run_round(
        self,
        tenant_id: str = "default",
        try_warehouse: bool = True,
    ) -> dict[str, Any]:
        """
        Drive one round: rotate warehouse attempts, run sequential pass along active ring,
        collect center guess from CenterObjectTarget. Returns round summary.
        """
        self.fit_ring(tenant_id=tenant_id)
        summary: dict[str, Any] = {"active_count": len(self._ring_order), "warehouse_tried": 0}

        # Staleness check: mez nodes with last_seen older than staleness_seconds
        if self._staleness_seconds is not None:
            now = datetime.now(timezone.utc)
            for nid in self._ring_order[:]:
                row = self._registry.get_node(nid)
                if not row:
                    continue
                ls = row.get("last_seen")
                if not ls:
                    continue
                try:
                    ts = datetime.fromisoformat(ls.replace("Z", "+00:00"))
                    if (now - ts).total_seconds() > self._staleness_seconds:
                        self.mark_mezzed(nid)
                        self._ring_order = [x for x in self._ring_order if x != nid]
                except (ValueError, TypeError):
                    pass

        # Health check active nodes; mark unresponsive as mezzed
        for nid in self._ring_order[:]:
            if not self.query_node(nid):
                self.mark_mezzed(nid)
                self._ring_order = [x for x in self._ring_order if x != nid]

        # Try warehouse nodes (at least one value)
        if try_warehouse:
            warehouse = self._registry.list_warehouse(tenant_id=tenant_id)
            for wh in warehouse[:3]:  # Try up to 3 warehouse nodes per round
                nid = wh.get("node_id")
                if not nid:
                    continue
                summary["warehouse_tried"] += 1
                self._registry.warehouse_record_retry(nid)
                if self.query_node(nid):
                    self._registry.mark_refreshed(nid)
                    self.fit_ring(tenant_id=tenant_id)
                    break

        # Advance center with any collected guesses (or generate new entropy)
        new_center = self._center.advance_round()
        summary["center_advanced"] = True
        summary["center_digest"] = new_center[:8].hex() if new_center else ""
        return summary

    @property
    def center(self) -> CenterObjectTarget:
        return self._center

    @property
    def registry(self) -> NodeRegistry:
        return self._registry

    @property
    def ring_order(self) -> list[str]:
        return self._ring_order.copy()
