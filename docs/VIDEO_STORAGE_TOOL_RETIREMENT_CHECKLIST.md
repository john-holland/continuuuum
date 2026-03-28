# video_storage_tool Retirement Checklist

Retire `video_storage_tool` only when every item is checked.

- [x] Every feature in `library/media_parity_matrix.json` has status `Parity-tested`.
- [x] Continuum endpoints implement every target API in the matrix.
- [x] USC provides all required core media primitives from the contract doc.
- [x] Bit-exact and lossy quality parity tests pass against USC+Continuum.
- [x] Stream cache mode (including budget + LRU) is validated in endpoint tests.
- [x] Settings parity is validated (all media knobs exposed in Continuum API).
- [x] Docs/reference workflows point to USC+Continuum, not `video_storage_tool`.
- [x] `video_storage_tool` has no unique runtime capability remaining.
