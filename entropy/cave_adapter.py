"""Cave adapter: spawn/stop Entropythief nodes, request_random, get_credits."""
from __future__ import annotations

import subprocess
import urllib.request
from typing import Any

from .credit_ledger import CreditLedger


def _http_json(url: str, method: str = "GET", data: dict | None = None) -> dict:
    req = urllib.request.Request(
        url,
        data=__import__("json").dumps(data).encode("utf-8") if data else None,
        headers={"Content-Type": "application/json"} if data else {},
        method=method,
    )
    with urllib.request.urlopen(req, timeout=10) as r:
        return __import__("json").loads(r.read().decode("utf-8"))


class CaveAdapter:
    """
    Adapter for Entropythief: starts petal nodes, requests randoms, gets credits.
    Usable from coinstroids and other consumers.
    """

    def __init__(self, continuum_url: str, credit_ledger: CreditLedger | None = None, db=None):
        self.continuum_url = continuum_url.rstrip("/")
        if db is not None and credit_ledger is None:
            credit_ledger = CreditLedger(db)
        self._ledger = credit_ledger
        self._processes: dict[str, subprocess.Popen] = {}

    def start_node(
        self,
        *,
        probe_target: str = "8.8.8.8:53",
        probe_class: str = "google_dns",
        tenant_id: str = "default",
        interval: float = 10.0,
        port: int = 0,
    ) -> str:
        """Spawn an entropythief petal node. Returns process handle (pid as string). Node registers itself with continuum."""
        import sys
        cmd = [
            sys.executable,
            "-m",
            "entropy.entropythief_node",
            "--continuum-url",
            self.continuum_url,
            "--probe-target",
            probe_target,
            "--probe-class",
            probe_class,
            "--tenant-id",
            tenant_id,
            "--interval",
            str(interval),
            "--port",
            str(port),
        ]
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        key = str(proc.pid)
        self._processes[key] = proc
        return key

    def stop_node(self, handle: str) -> None:
        """Stop a node by process handle (pid string from start_node)."""
        if handle in self._processes:
            self._processes[handle].terminate()
            try:
                self._processes[handle].wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._processes[handle].kill()
            del self._processes[handle]

    def list_nodes(self) -> list[dict[str, Any]]:
        """List active and warehouse nodes from continuum."""
        resp = _http_json(f"{self.continuum_url}/api/entropy/nodes")
        return resp.get("active", []) + resp.get("warehouse", [])

    def tombstone_node(self, node_id: str) -> None:
        """Mark node as mezzed (tombstoned). Call when game loses focus or client suspends."""
        _http_json(
            f"{self.continuum_url}/api/entropy/nodes/tombstone",
            "POST",
            {"node_id": node_id},
        )

    def mezz_node(self, node_id: str) -> None:
        """Synonym for tombstone_node. Mark node as mezzed via POST /api/entropy/nodes/mezz."""
        _http_json(
            f"{self.continuum_url}/api/entropy/nodes/mezz",
            "POST",
            {"node_id": node_id},
        )

    def request_random(
        self,
        tenant_id: str = "default",
        size_bytes: int = 32,
    ) -> dict[str, Any]:
        """Request random from chain. Returns {random, credits_spent, credits_balance} or error."""
        try:
            resp = _http_json(
                f"{self.continuum_url}/api/entropy/random",
                "POST",
                {"tenant_id": tenant_id, "bytes": size_bytes},
            )
            return resp
        except urllib.error.HTTPError as e:
            if e.code == 402:
                body = __import__("json").loads(e.read().decode("utf-8")) if e.fp else {}
                return {"error": "Insufficient credits", **body}
            raise
        except Exception as e:
            return {"error": str(e)}

    def get_credits(self, tenant_id: str = "default") -> dict[str, Any]:
        """Return balance, earned_total, spent_total."""
        return _http_json(f"{self.continuum_url}/api/entropy/credits?tenant={tenant_id}")
