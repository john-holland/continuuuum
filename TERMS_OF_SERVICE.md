# Continuum Terms of Service (Project Draft)

This document is a project policy draft for development and release gating. It is not a substitute for licensed legal counsel.

## Service Character

Continuum is provided for development and operational workflows around library/document/media services and related research features.

## Entropy and Randomness Claims

When entropy-related features are enabled, Continuum makes only conservative, measured-condition claims:

- Continuum improves unpredictability under measured conditions when probe evidence requirements are satisfied.
- Continuum does not guarantee universal or perpetual randomness quality.
- Continuum does not represent outputs as legally certified "true random."

## Infrastructure Stability Baseline

Continuum may use common DNS/edge infrastructure measurements as a baseline:

- Google DNS endpoints
- Akamai edge endpoints
- localhost loopback control channel

These baseline measurements describe infrastructure behavior; they do not by themselves guarantee entropy quality.

## Contract Gate and Verification

Release-facing entropy claims are conditioned on passing the legal/compliance contract gate:

- approved claim wording checks,
- required evidence field checks,
- live probe integration checks against external and local control classes.

If these checks fail, entropy claims must not be published for that build.

## Availability and Third-Party Dependencies

External network services, routing, and endpoint availability are outside project control. Failures or degraded service from third-party infrastructure may affect measurements and feature behavior.

## Changes

These terms may be updated as part of project governance and compliance hardening.
