# Continuuuum tenant (per game / local dev)

Library documents are scoped by **tenant_id**. The server reads the tenant from each request and filters all library endpoints by it.

## Local development

- Use tenant **`default`** when no tenant is configured. The server and CLI use `default` when the client does not send `X-Tenant-ID` (or query param `tenant`).
- This aligns with log-view-machine’s tenant fallback and keeps one clear “local” tenant.

## Production

- Tenants should be **stable lowercase slugs** (e.g. `game-slug`, `team-id`). Each game team or org uses one tenant so data is isolated.
- Clients (Unity Continuuuum windows, or Cave when it proxies to continuuuum) send the tenant via:
  - Header: **`X-Tenant-ID: <tenant>`**
  - Or query param: **`tenant=<tenant>`**

## Summary

| Context        | Tenant value   |
|----------------|----------------|
| Local dev      | `default`      |
| Production     | Stable slug per game/team |
