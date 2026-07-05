# Continuuuum rebrand (breaking changes)

The product/engine formerly named **Continuum** is now **Continuuuum** (three `u`s).

## Environment variables

| Old | New |
|-----|-----|
| `CONTINUUM_DB` / `CONTINUUM_DB_PATH` | `CONTINUUUUM_DB` / `CONTINUUUUM_DB_PATH` |
| `CONTINUUM_REPO` | `CONTINUUUUM_REPO` |
| `CONTINUUM_TENANT` | `CONTINUUUUM_TENANT` |
| `CONTINUUM_LIBRARY_BASE` | `CONTINUUUUM_LIBRARY_BASE` |
| `CONTINUUUUM_SHARED_STATIC` | (was `CONTINUUM_SHARED_STATIC`) |
| `CONTINUUM_API_URL` | `CONTINUUUUM_API_URL` |

## Python

- Package path: `Scripts/continuum_api` → `Scripts/continuuuum_api`
- USC ORM: `ContinuumDb` → `ContinuuuumDb` (`continuuuum_db.py`)

## Unity (Drawer 2)

- Assembly folder: `Assets/Continuum` → `Assets/Continuuuum`
- Namespaces: `Continuum.*` → `Continuuuum.*`

## HTTP static URLs

All `/static/shared/continuum-*` paths are now `/static/shared/continuuuum-*`.

## Local checkout

Rename your library server folder if still checked out as `continuum`:

```powershell
Rename-Item "C:\Users\John\continuum" "C:\Users\John\continuuuum"
```

## Database files

Existing `continuum.db` files remain valid; point `CONTINUUUUM_DB` at the old path or rename the file locally.
