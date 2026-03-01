# Entropy Claims and Evidence Policy

This policy governs public claims, contract assertions, and release gates for entropy quality messaging in Continuum.

## Policy Intent

- This policy defines a technical compliance guardrail.
- This policy does not constitute legal certification.
- Claims must remain conservative and evidence-bound.

## Approved Claim Language

Allowed claim pattern:

- "The system improves unpredictability under measured conditions using externally observed drift against local control baselines."

Required qualifiers:

- Results depend on live network conditions.
- Results vary by region, routing, and endpoint availability.
- Measurements provide probabilistic evidence, not absolute guarantees.

## Prohibited Claim Language

The following are prohibited in release-facing policy and contract text:

- assertions that system behavior "ensures entropy destabilization"
- "guaranteed random" or "guarantees randomness"
- "true random" as an absolute claim without qualifiers

## Probe Matrix and Measurement Contract

The service-level verification layer must compare external infrastructure probes against local control probes.

### External probe endpoints

- Google DNS endpoint A: `8.8.8.8:53`
- Google DNS endpoint B: `8.8.4.4:53`
- Akamai endpoint: `www.akamai.com:443`

### Control endpoint

- Local loopback control: `127.0.0.1` (locally hosted probe target)

## Required Evidence Fields

Every evidence bundle must include:

- `probe_target`
- `probe_class` (`google_dns`, `akamai_edge`, `localhost_loopback`)
- `timestamp_source`
- `sample_window_seconds`
- `sample_count`
- `timeout_seconds`
- `rtt_ms_series`
- `rtt_mean_ms`
- `rtt_variance_ms2`
- `drift_series_ms`
- `confidence_bounds_ms`
- `failure_handling`

## Minimum Validation Conditions

- Minimum sample count per probe class must be met.
- External classes must show non-degenerate variance across sample windows.
- Localhost control must remain measurable and materially lower-latency than external classes in the same run window.
- Any failed probe must be recorded with explicit error taxonomy.

## Terms and Contract Linkage

Terms and contract gates must state:

- infrastructure stability baselines are measured from common DNS/edge probe classes,
- entropy claims are bounded to measured conditions,
- no promise of universal or perpetual randomness quality is made.

## Audit and Freshness

- Probe measurements are always live in integration verification mode.
- Cached probe outputs are not valid for legal/compliance gate checks.
- Evidence artifacts must include generation timestamp and software version identity.
