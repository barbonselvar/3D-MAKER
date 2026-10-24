# Vendored browser dependencies

Served from this directory so the studio never loads code from a CDN at runtime (blueprint P0-03).
Files are copied unmodified from the npm tarballs below; `SHA256SUMS` is checked by `tests/test_vendor.py`.

| Package | Version | Licence | Source | Files |
| --- | --- | --- | --- | --- |
| three | 0.169.0 | MIT | `npm pack three@0.169.0` | `build/three.module.js`, the six example modules the studio imports and their two `utils/` dependencies |
| @gradio/client | 2.7.1 (published 2026-09-29) | Apache-2.0 (`LICENSE` file; package.json says ISC) | `npm pack @gradio/client@2.7.1` | `dist/index.min.js` (self-contained ESM bundle) |

To upgrade: `npm pack <pkg>@<version>`, copy the same files, regenerate `SHA256SUMS` with
`cd app/static && find vendor -type f ! -name SHA256SUMS ! -name README.md | sort | xargs sha256sum > vendor/SHA256SUMS`,
and update this table.
