# CI Generator

This repository uses a single-source CI generator (Node + Handlebars) to produce:

- `.github/workflows/ci.yml`
- `.circleci/config.yml`

## Why

One config model keeps GitHub Actions and CircleCI behavior aligned, including semantic variant test coverage.

## Source Files

- `tools/ci-gen/ci.config.json`
- `tools/ci-gen/templates/github-workflow.hbs`
- `tools/ci-gen/templates/circleci-config.hbs`
- `tools/ci-gen/generate-ci.mjs`

## Generate

```bash
npm install
npm run ci:generate
```

## Drift Check

Use this in CI or pre-commit to ensure generated files are current:

```bash
npm run ci:check
```

`ci:check` exits non-zero when generated outputs differ from committed outputs.

## Pact / Contract Check

Validate generated workflow files against published schemas and local generator contracts:

```bash
npm run ci:pact
```

This check validates:

- GitHub workflow against SchemaStore `github-workflow.json`
- CircleCI config against CircleCI YAML language server schema
- manual-only GitHub trigger contract (`workflow_dispatch`, no `push`/`pull_request`)
- semantic variant generation count consistency with `ci.config.json`
- required core CircleCI workflow jobs
- required GitHub workflow jobs from `ci.config.json`
- legal/compliance policy contract:
  - approved entropy claim wording only
  - prohibited guarantee wording blocked
  - required endpoint matrix entries (Google DNS, Akamai, localhost control)
  - required evidence fields in policy/checklist docs

The legal contract check reads:

- `TERMS_OF_SERVICE.md`
- `docs/ENTROPY_CLAIMS_AND_EVIDENCE_POLICY.md`
- `docs/LEGAL_COMPLIANCE_CHECKLIST.md`

## CircleCI Local

Validate Circle config:

```bash
npm run ci:circle:validate
```

Run a local Circle job:

```bash
npm run ci:circle:local
```

This requires CircleCI CLI and local Docker support.

## GitHub Actions Safety Posture

The generated GitHub workflow is intentionally manual-only:

- Trigger: `workflow_dispatch`
- No `push` or `pull_request` triggers

This keeps Actions output present and reviewable without automatically consuming hosted CI resources.

## Semantic Variant Matrix

The generator emits additional semantic variant runs from `semanticVariants` in `ci.config.json`:

- Multipart synonym limits
- Prompt language-version expressions:
  - `word@>1.x`
  - `word@>1.1.x`
  - `word@{1.1.x-2.x}`

The same intent is rendered into both CI platforms.

## Verification Commands

Run from Continuum repo root:

```bash
npm run ci:generate
npm run ci:check
npm run ci:pact
python -m pytest tests/test_entropy_claims_contract.py -v
```

Expected behavior:

- `ci:generate` updates `.github/workflows/ci.yml` and `.circleci/config.yml`.
- `ci:check` fails if generated files drift from source templates/config.
- `ci:pact` fails when:
  - required legal/compliance docs are missing,
  - prohibited guarantee wording appears in claim sections,
  - required evidence fields or endpoint matrix entries are missing,
  - required jobs are not emitted into GitHub and CircleCI workflows.
- `test_entropy_claims_contract.py` fails when:
  - live external probes cannot collect minimum samples,
  - evidence bundles are missing required fields,
  - external RTT baseline is not greater than localhost control RTT.
