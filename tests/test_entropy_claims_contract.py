from __future__ import annotations

import os
import sys
from pathlib import Path

# Ensure repo root is on path so serve_library can be imported
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from serve_library import collect_entropy_probe_evidence, collect_probe_series  # noqa: E402


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
