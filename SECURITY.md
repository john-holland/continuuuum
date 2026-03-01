# Security

## Auth stance

- **By default the continuum server has no authentication.** Search and upload are open. This is acceptable for **local development** or when the server is not exposed to the internet.
- **If the server is exposed** (e.g. on a shared network or public URL), treat it as untrusted and add protection:
  - **Recommended:** Put the server behind a reverse proxy (e.g. nginx, Caddy) that enforces auth (e.g. HTTP basic, OAuth, or VPN).
  - **Optional:** Set `CONTINUUM_API_KEY` (see below) to require an API key on requests.

## Optional API key

If you set the environment variable **CONTINUUM_API_KEY**, the server will require that value on each request to library API routes:

- **Header:** `X-API-Key: <your-key>`
- **Query param:** `api_key=<your-key>`

If the key is set and the request does not provide a matching value, the server returns 401. If `CONTINUUM_API_KEY` is not set, no key is required (local use only).

## Summary

| Use case           | Recommendation                    |
|--------------------|------------------------------------|
| Local / dev only   | No auth; do not expose.            |
| Shared / exposed   | Reverse-proxy auth or CONTINUUM_API_KEY. |

## Compliance and Claims References

- Terms draft: `TERMS_OF_SERVICE.md`
- Entropy claims/evidence policy: `docs/ENTROPY_CLAIMS_AND_EVIDENCE_POLICY.md`
- Compliance checklist: `docs/LEGAL_COMPLIANCE_CHECKLIST.md`

These references define conservative wording requirements and evidence-bound verification expectations for entropy-related claims.
