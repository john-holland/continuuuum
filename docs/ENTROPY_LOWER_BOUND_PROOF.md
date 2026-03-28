# Appendix: Entropy Lower-Bound Proof (Cross-Provider Instability)

This appendix provides an assumption-driven proof sketch that cross-provider timing instability can increase harvested entropy relative to localhost-only control measurements.

It is a lower-bound argument under explicit conditions. It is not a claim of absolute or universally guaranteed randomness.

## 1) Model and Notation

Let provider probes be indexed by \(p \in \{1,\dots,m\}\) and sampled at discrete times \(t\).

- External latency vector:
  \[
  \mathbf{X}_t = (X_{1,t},\dots,X_{m,t})^\top \in \mathbb{R}^m
  \]
- Local control latency:
  \[
  L_t \in \mathbb{R}
  \]

Decompose each external channel into predictable + residual parts:
\[
X_{p,t} = \mu_{p,t} + \eta_{p,t}
\]
where \(\mu_{p,t}\) is trend/forecastable structure and \(\eta_{p,t}\) is residual micro-instability.

Define common-mode projection and ring residual:
\[
P = \frac{1}{m}\mathbf{1}\mathbf{1}^\top,\quad
\mathbf{R}_t = (I-P)\mathbf{X}_t
\]
so \(\mathbf{R}_t\) removes ring-average/common-mode variation.

Let \(\widehat{\mathbf{R}}_t\) be the ring predictor (for example an LSTM-ring estimate), and define innovation:
\[
\mathbf{E}_t = \mathbf{R}_t - \widehat{\mathbf{R}}_t
\]

Harvested bits are produced from quantized innovations and extractor hashing.

## 2) Requirement Dependency Tree

Goal \(G\): harvested output has strictly larger entropy than localhost-only residual baseline.

Required predicates:

- \(R_1\): External instability dominates local control in at least one channel/window.
  \[
  \exists p:\ \operatorname{Var}(X_{p,t}) > \operatorname{Var}(L_t)
  \]
- \(R_2\): Cross-provider channels are not pure rank-1 lockstep.
  \[
  (I-P)\Sigma_X(I-P) \neq 0
  \]
  where \(\Sigma_X = \operatorname{Cov}(\mathbf{X}_t)\).
- \(R_3\): Predictor is not perfect; innovation covariance is non-degenerate.
  \[
  \operatorname{Cov}(\mathbf{E}_t \mid \mathcal{F}_{t-1}) \succeq \epsilon I,\ \epsilon>0
  \]
- \(R_4\): Extractor is sound (standard leftover-hash style requirement over sufficient min-entropy input).

Dependency implication:
\[
R_1 \land R_2 \land R_3 \land R_4 \Rightarrow G
\]

## 3) Lemmas

### Lemma 1: Ring refactor isolates non-common instability
\[
\Sigma_R = \operatorname{Cov}(\mathbf{R}_t) = (I-P)\Sigma_X(I-P)
\]
If \(R_2\) holds, then \(\Sigma_R\) has at least one positive eigenvalue, so residual ring coordinates have nonzero variance.

### Lemma 2: Innovation retains conditional uncertainty
Under \(R_3\), innovation covariance has a positive-definite lower bound in at least one residual subspace dimension. Therefore conditional differential entropy is bounded below:
\[
h(\mathbf{E}_t \mid \mathcal{F}_{t-1}) \ge \frac{1}{2}\log\!\big((2\pi e)^d \det(\epsilon I)\big)
\]
for residual dimension \(d\).

### Lemma 3: Quantization preserves positive entropy for finite step
For scalar/coordinate quantizer \(Q_\Delta\):
\[
H(Q_\Delta(\mathbf{E}_t)\mid \mathcal{F}_{t-1})
\gtrsim
h(\mathbf{E}_t\mid \mathcal{F}_{t-1}) - d\log\Delta - c
\]
for constant \(c\) induced by quantizer form. For finite \(\Delta\), entropy remains positive if the bound in Lemma 2 is sufficient.

### Lemma 4: External-vs-local entropy gap
If \(R_1\) holds over the same sampling window and controls, then under equal quantization and extraction policy:
\[
H(Q_\Delta(\mathbf{E}_t)) - H(Q_\Delta(L_t-\widehat{L}_t)) > 0
\]
for at least one admissible operating window.

## 4) Theorem (Entropy Gain Under Measured Conditions)

Given \(R_1\)–\(R_4\), extracted output from ring-innovation evidence has a positive lower bound on min-entropy and exceeds localhost-only baseline entropy under matched windowing and quantization assumptions.

Therefore, cross-provider instability is sufficient to increase entropy collection under measured conditions.

## 5) Operationalization in Continuum

The proof assumptions map to enforceable checks:

- \(R_1\): external probes (Google DNS/Akamai) vs localhost loopback comparison.
- \(R_2\): non-degenerate external variance and non-lockstep behavior in observed series.
- \(R_3\): innovation/residual checks from service-level evidence pipeline (forecast residual contract).
- \(R_4\): extractor policy + deterministic evidence format for auditability.

## 6) CI and Policy Mapping

The following project artifacts enforce or document proof assumptions:

- `docs/ENTROPY_CLAIMS_AND_EVIDENCE_POLICY.md`
- `docs/ENTROPYTHIEF_RING_ARCHITECTURE.md` — ring overlap implementation (CenterObjectTarget, RingOrchestrator)
- `docs/LEGAL_COMPLIANCE_CHECKLIST.md`
- `tools/ci-gen/pact-ci-generator.mjs`
- `tests/test_entropy_claims_contract.py` — probe evidence contract; ring-based evidence when entropy API available
- `serve_library.py` (live probe evidence collection functions)

These gates ensure release claims remain assumption-bound, evidence-backed, and reproducible.

## 7) Scope and Limits

- This is a lower-bound argument, not a proof of perfect randomness.
- Live network behavior can shift by geography, provider policy, and transient failures.
- Legal/compliance language must remain conservative and tied to measured evidence thresholds.
