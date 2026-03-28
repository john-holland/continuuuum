from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import pytest

# Ensure repo root is on path so serve_library can be imported
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

# Use temp DB for ring test so we don't touch a real DB
_tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
_tmp.close()
os.environ.setdefault("CONTINUUM_DB_PATH", _tmp.name)

from serve_library import (  # noqa: E402
    app,
    collect_entropy_probe_evidence,
    collect_probe_series,
)


def _required_evidence_fields() -> set[str]:
    return {
        "probe_target",
        "probe_class",
        "timestamp_source",
        "sample_window_seconds",
        "sample_count",
        "timeout_seconds",
        "rtt_ms_series",
        "rtt_mean_ms",
        "rtt_variance_ms2",
        "drift_series_ms",
        "confidence_bounds_ms",
        "failure_handling",
    }


def test_collect_probe_series_contract_shape():
    sample_count = int(os.environ.get("ENTROPY_TEST_SAMPLE_COUNT", "3"))
    timeout_seconds = float(os.environ.get("ENTROPY_TEST_TIMEOUT_SECONDS", "2.0"))
    sample_window_seconds = float(os.environ.get("ENTROPY_TEST_WINDOW_SECONDS", "5.0"))

    evidence = collect_probe_series(
        probe_target="8.8.8.8:53",
        probe_class="google_dns",
        sample_count=sample_count,
        timeout_seconds=timeout_seconds,
        sample_window_seconds=sample_window_seconds,
    )

    assert _required_evidence_fields().issubset(set(evidence.keys()))
    assert evidence["probe_class"] == "google_dns"
    assert evidence["sample_count"] >= sample_count
    assert len(evidence["rtt_ms_series"]) >= sample_count
    assert isinstance(evidence["confidence_bounds_ms"], list)
    assert len(evidence["confidence_bounds_ms"]) == 2


def test_live_external_vs_loopback_entropy_contract():
    sample_count = int(os.environ.get("ENTROPY_TEST_SAMPLE_COUNT", "3"))
    timeout_seconds = float(os.environ.get("ENTROPY_TEST_TIMEOUT_SECONDS", "2.0"))
    sample_window_seconds = float(os.environ.get("ENTROPY_TEST_WINDOW_SECONDS", "5.0"))

    evidence = collect_entropy_probe_evidence(
        sample_count=sample_count,
        timeout_seconds=timeout_seconds,
        sample_window_seconds=sample_window_seconds,
    )

    series = evidence["series"]
    comparison = evidence["comparison"]

    assert "google_dns_a" in series
    assert "google_dns_b" in series
    assert "akamai_edge" in series
    assert "localhost_loopback" in series

    for key in ("google_dns_a", "google_dns_b", "akamai_edge", "localhost_loopback"):
        assert _required_evidence_fields().issubset(set(series[key].keys()))
        assert series[key]["sample_count"] >= sample_count

    # Contract expectation: local loopback is a lower-latency control than external internet paths.
    assert comparison["external_mean_rtt_ms"] > comparison["localhost_mean_rtt_ms"]

    # External and local variance should both be non-negative and at least one external series should vary.
    assert comparison["external_variance_ms2"] >= 0.0
    assert comparison["localhost_variance_ms2"] >= 0.0
    assert (
        series["google_dns_a"]["rtt_variance_ms2"] > 0.0
        or series["google_dns_b"]["rtt_variance_ms2"] > 0.0
        or series["akamai_edge"]["rtt_variance_ms2"] > 0.0
    )


def test_ring_based_evidence_when_available():
    """When entropy API is available, ring topology returns valid shape (nodes, ring, next mapping)."""
    with app.test_client() as client:
        r = client.get("/api/entropy/nodes")
        if r.status_code == 503:
            pytest.skip("Entropy API requires USC with entropy schema")
        assert r.status_code == 200
        data = r.get_json()
        assert "active" in data
        assert "warehouse" in data
        assert isinstance(data["active"], list)
        assert isinstance(data["warehouse"], list)

        r2 = client.get("/api/entropy/ring")
        if r2.status_code == 503:
            pytest.skip("Entropy API requires USC with entropy schema")
        assert r2.status_code == 200
        ring_data = r2.get_json()
        assert "ring_order" in ring_data
        assert isinstance(ring_data["ring_order"], list)


def test_request_random_and_credits_when_available():
    """When entropy API is available, request_random spends credits and returns random; get_credits returns balance."""
    with app.test_client() as client:
        r = client.get("/api/entropy/credits?tenant=test_consumer")
        if r.status_code == 503:
            pytest.skip("Entropy API requires USC with entropy schema")
        assert r.status_code == 200
        creds = r.get_json()
        assert "balance" in creds
        assert "earned_total" in creds
        assert "spent_total" in creds

        reg = client.post(
            "/api/entropy/nodes/register",
            json={"probe_target": "8.8.8.8:53", "tenant_id": "test_consumer"},
        )
        if reg.status_code == 503:
            pytest.skip("Entropy API requires USC with entropy schema")

        run_r = client.post("/api/entropy/orchestrator/run-round")
        if run_r.status_code == 503:
            pytest.skip("Entropy API requires USC with entropy schema")
        assert run_r.status_code == 200

        rand_r = client.post(
            "/api/entropy/random",
            json={"tenant_id": "test_consumer", "bytes": 32},
        )
        if rand_r.status_code == 503:
            pytest.skip("Entropy API requires USC with entropy schema")
        assert rand_r.status_code == 200
        data = rand_r.get_json()
        assert "random" in data
        assert "credits_spent" in data
        assert "credits_balance" in data
        assert len(bytes.fromhex(data["random"])) == 32


def test_tombstone_endpoint_when_available():
    """Client tombstones node via POST /api/entropy/nodes/tombstone; node moves to warehouse."""
    with app.test_client() as client:
        r = client.get("/api/entropy/nodes")
        if r.status_code == 503:
            pytest.skip("Entropy API requires USC with entropy schema")

        reg = client.post(
            "/api/entropy/nodes/register",
            json={"probe_target": "8.8.8.8:53", "tenant_id": "tombstone_test"},
        )
        if reg.status_code == 503:
            pytest.skip("Entropy API requires USC with entropy schema")
        assert reg.status_code == 201
        node_id = reg.get_json()["node_id"]

        nodes_before = client.get("/api/entropy/nodes").get_json()
        active_ids = [n["node_id"] for n in nodes_before["active"]]
        assert node_id in active_ids

        tomb = client.post(
            "/api/entropy/nodes/tombstone",
            json={"node_id": node_id},
        )
        assert tomb.status_code == 200
        assert tomb.get_json() == {"ok": True, "node_id": node_id}

        nodes_after = client.get("/api/entropy/nodes").get_json()
        active_ids_after = [n["node_id"] for n in nodes_after["active"]]
        warehouse_ids = [n["node_id"] for n in nodes_after["warehouse"]]
        assert node_id not in active_ids_after
        assert node_id in warehouse_ids


def test_mezz_endpoint_synonym_when_available():
    """POST /api/entropy/nodes/mezz is synonym for tombstone; same behavior."""
    with app.test_client() as client:
        r = client.get("/api/entropy/nodes")
        if r.status_code == 503:
            pytest.skip("Entropy API requires USC with entropy schema")

        reg = client.post(
            "/api/entropy/nodes/register",
            json={"probe_target": "8.8.8.8:53", "tenant_id": "mezz_test"},
        )
        if reg.status_code == 503:
            pytest.skip("Entropy API requires USC with entropy schema")
        assert reg.status_code == 201
        node_id = reg.get_json()["node_id"]

        mezz = client.post(
            "/api/entropy/nodes/mezz",
            json={"node_id": node_id},
        )
        assert mezz.status_code == 200
        assert mezz.get_json() == {"ok": True, "node_id": node_id}

        nodes_after = client.get("/api/entropy/nodes").get_json()
        assert node_id not in [n["node_id"] for n in nodes_after["active"]]
        assert node_id in [n["node_id"] for n in nodes_after["warehouse"]]
