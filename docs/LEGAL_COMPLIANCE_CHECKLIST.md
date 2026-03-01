# Legal Compliance Checklist (Entropythief)

This checklist is a technical compliance guardrail for Continuum + USC integrations. It is not legal certification and does not replace counsel review.

## Claim Language Gate

- [ ] Product language uses conservative phrasing: "improves unpredictability under measured conditions."
- [ ] Product language does not claim guaranteed randomness outcomes.
- [ ] Product language does not use absolute "true random" claims.
- [ ] Terms and contracts include explicit limitations and operating assumptions.

## License and Attribution Gate

- [ ] `LICENSE` remains MIT in Continuum and Apache-2.0 in USC.
- [ ] Any new dependency license is recorded in release notes or dependency docs.
- [ ] No third-party model, service, or API terms are copied into product terms without review.
- [ ] Cross-repo legal references are linked from docs and README files.

## Entropy Evidence Gate

- [ ] External probe matrix is documented with Google DNS and Akamai endpoints.
- [ ] Local control probe (localhost loopback) is documented.
- [ ] Policy defines sample window, sample count, timeout, and retry handling.
- [ ] Policy defines timestamp source and data retention behavior for evidence artifacts.
- [ ] Policy defines minimum quality checks (variance and sample sufficiency).

## ISP Bridge Stability Baseline

This section documents infrastructure stability used as a baseline, separate from the entropy process itself.

| Probe class | Endpoint example | Typical behavior | Stability interpretation |
|---|---|---|---|
| Google DNS | `8.8.8.8:53`, `8.8.4.4:53` | Low-to-moderate jitter, globally anycasted | External network baseline for widely used resolver infrastructure |
| Akamai edge | `www.akamai.com:443` | Region-dependent latency and route variation | External CDN/infrastructure baseline under real internet paths |
| Local control | `127.0.0.1` loopback | Very low jitter and low latency | Control channel that approximates local stack floor |

Required interpretation:

- [ ] Published wording frames external probes as infrastructure stability baselines.
- [ ] Published wording frames local loopback as control baseline.
- [ ] Published wording frames coin logic as entropy augmentation over baseline measurements, not as guaranteed destabilization.

## Terms and Contract Gate

- [ ] Terms of service reference the entropy claims/evidence policy.
- [ ] Contract language states verification depends on live network conditions.
- [ ] Contract language states external endpoint availability is outside project control.
- [ ] Contract language includes fallback handling for probe failures and partial evidence windows.

## CI and Release Gate

- [ ] `npm run ci:pact` validates policy/checklist wording and required sections.
- [ ] Generated GitHub and CircleCI workflows both include the legal/compliance pact job.
- [ ] Live probe integration tests execute service functions directly and produce deterministic evidence fields.
- [ ] Release is blocked if claim wording or evidence requirements fail.
