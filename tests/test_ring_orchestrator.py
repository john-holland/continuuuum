"""Tests for RingOrchestrator, CenterObjectTarget, NodeRegistry, CreditLedger."""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest

_repo = Path(__file__).resolve().parent.parent
if str(_repo) not in sys.path:
    sys.path.insert(0, str(_repo))

pytest.importorskip("unified_semantic_archiver.db")
from unified_semantic_archiver.db import ContinuumDb, get_connection, init_schema

from entropy import CenterObjectTarget, NodeRegistry, RingOrchestrator
from entropy.credit_ledger import CreditLedger


@pytest.fixture
def entropy_db():
    """Fresh DB per test to avoid cross-test pollution."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    try:
        prev = os.environ.get("CONTINUUM_DB_PATH")
        os.environ["CONTINUUM_DB_PATH"] = path
        conn = get_connection(path)
        init_schema(conn)
        conn.close()
        db = ContinuumDb(path)
        yield db
    finally:
        if prev is not None:
            os.environ["CONTINUUM_DB_PATH"] = prev
        else:
            os.environ.pop("CONTINUUM_DB_PATH", None)
        try:
            os.unlink(path)
        except OSError:
            pass


def test_center_target_accept_guess_and_advance():
    center = CenterObjectTarget()
    center.accept_guess("n1", {"guess_bytes": (b"a" * 32).hex()})
    center.accept_guess("n2", {"guess_bytes": (b"b" * 32).hex()})
    before = center.get_authoritative_center()
    new_center = center.advance_round()
    assert len(new_center) == 32
    assert new_center != before


def test_center_mix_entropy_yesterday():
    center = CenterObjectTarget()
    h = b"prior"
    center.mix_entropy_yesterday(h)
    assert center.get_prior_entropy_hash() == h


def test_node_registry_add_list_mez_refresh(entropy_db):
    reg = NodeRegistry(entropy_db)
    nid = reg.add_node(probe_target="8.8.8.8:53", tenant_id="default")
    assert nid
    active = reg.list_active()
    assert len(active) == 1
    assert active[0]["node_id"] == nid
    reg.mark_mezzed(nid)
    assert len(reg.list_active()) == 0
    assert len(reg.list_warehouse()) == 1
    reg.mark_refreshed(nid)
    assert len(reg.list_active()) == 1


def test_ring_orchestrator_fit_ring(entropy_db):
    orch = RingOrchestrator(entropy_db)
    n1 = orch.add_node(probe_target="8.8.8.8:53")
    n2 = orch.add_node(probe_target="8.8.4.4:53")
    ring = orch.fit_ring()
    assert n1 in ring
    assert n2 in ring
    assert orch.next_node_id(n1) == n2
    assert orch.next_node_id(n2) == n1


def test_credit_ledger_earn_spend(entropy_db):
    ledger = CreditLedger(entropy_db)
    assert ledger.get_credits("t1")["balance"] == 0
    ledger.earn("t1", 100)
    assert ledger.get_credits("t1")["balance"] == 100
    assert ledger.spend("t1", 30) is True
    assert ledger.get_credits("t1")["balance"] == 70
    assert ledger.spend("t1", 100) is False
    assert ledger.get_credits("t1")["balance"] == 70


def test_ring_orchestrator_timeout_config(entropy_db):
    """RingOrchestrator accepts health_check_timeout_seconds and staleness_seconds."""
    orch = RingOrchestrator(
        entropy_db,
        health_check_timeout_seconds=5.0,
        staleness_seconds=120.0,
    )
    orch.add_node(probe_target="8.8.8.8:53")
    orch.fit_ring()
    assert orch.ring_order


def test_staleness_mez(entropy_db):
    """Nodes with last_seen older than staleness_seconds are mezzed during run_round."""
    from datetime import datetime, timedelta, timezone

    orch = RingOrchestrator(entropy_db, staleness_seconds=60.0)
    nid = orch.add_node(probe_target="8.8.8.8:53")
    # Set last_seen to 2 minutes ago via registry
    old_ts = (datetime.now(timezone.utc) - timedelta(seconds=120)).strftime("%Y-%m-%dT%H:%M:%SZ")
    entropy_db.entropy_ring_node_update_status(nid, "active", last_seen=old_ts)
    orch.fit_ring()
    assert nid in orch.ring_order
    # Health check would pass (mock) - staleness should still mez the node
    orch._query_node = lambda _nid, _url: True
    orch.run_round(try_warehouse=False)
    assert nid not in orch.ring_order
    assert len(orch.registry.list_warehouse()) == 1
    assert orch.registry.list_warehouse()[0]["node_id"] == nid
