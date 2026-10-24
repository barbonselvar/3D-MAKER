# 3D Maker Studio

Turn one image into a textured 3D model. The browser studio uploads an image, sends it to an
image-to-3D engine on a GPU you choose, shows progress, and loads the validated GLB into an
interactive viewer. Local, non-AI tools (table builder, silhouette extrusion, GLB import) remain
available and are always labelled "not AI".

Plan and decisions: [`docs/PLAN.md`](docs/PLAN.md). GPU worker: [`worker/README.md`](worker/README.md).

## How it works

```
browser studio ──► public API (app.main, /api/v1) ──► SQLite job queue + object storage
                                      ▲
GPU worker (worker.main) ──► worker API (app.internal, /internal/v1, private port)
```

- **Generate** (AI mode) opens a choice of **Internal GPU** — a worker you run on your own NVIDIA
  GPU — or **External GPU**. Nothing is ever rerouted between them, and a failed job never
  falls back to a non-AI mesh.
- The primary engine is **TRELLIS.2-4B** (MIT). It needs an NVIDIA GPU with at least 24 GB.
- Every model passes validation gates before it can be downloaded: Khronos glTF-Validator,
  mesh sanity, PBR textures and UVs, and a Blender import in the worker image.
- Each model has a `manifest.json` that records the input hash, engine commit and weights
  revision, seed, GPU, timings, validation results and the licence notices that apply.
- The model is generated from one image, so the back and hidden parts are estimated, not
  reconstructed. The UI says this with every result.

External GPU today: the **Hugging Face public demo** of TRELLIS.2 runs from your browser after you
consent. It is shared and free, with quotas and no SLA, and your image leaves your device. A
dedicated paid provider (Phase 2) is shown as "not configured". No paid service is used.

## Run it

Requires Python 3.10+ and Node.js 18+ (for the glTF validator).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
npm ci --prefix worker            # Khronos glTF-Validator used by the worker
cp .env.example .env              # optional; see the comments in that file
uvicorn app.main:app --reload     # studio + public API on http://localhost:8000
```

To use an Internal GPU, also run the worker API on a private interface and start a worker on the
GPU machine. Both are described in `worker/README.md`:

```bash
MAKER_WORKER_TOKENS="gpu-1:<long random token>" uvicorn app.internal:app --host 127.0.0.1 --port 8001
```

The studio still works as plain static files (`python -m http.server 8000 --directory app/static`).
In that case only the local tools and the Hugging Face demo are available.

## API

Unchanged: `GET /api/health`, `POST /api/generate` (silhouette extrusion, not AI) and
`GET /api/files/{id}/{name}`.

New, under `/api/v1`: `POST /uploads`, `GET /gpu-targets`, `POST /jobs`, `GET /jobs/{id}`,
`GET /jobs/{id}/events` (Server-Sent Events), `POST /jobs/{id}/cancel`, `POST /jobs/{id}/retry`,
`GET /models/{id}`, `GET /models/{id}/files/{format}` (redirects to a signed link valid for
10 minutes). Errors are returned as `{"error": {"code", "message", "retryable"}}`.

## Tests

```bash
ruff check . && python scripts/check_js.py && pytest -q
```

CI (`.github/workflows/ci.yml`) runs the same commands, including the browser tests in headless
Chromium.

## What is tested, and what is not

**Tested** (automated, run on every push):

- The existing API (health, silhouette GLB, path traversal, bad input), and output cleanup after 24 h.
- Studio regression tests in a real browser, with all off-site requests blocked: boot, silhouette,
  table, wireframe/rotate, GLB/OBJ/STL/PLY downloads.
- Textured models are never tinted by the colour tool. Imported or AI GLBs download as the
  original bytes. Derived exports keep the model's own scale.
- Vendored Three.js 0.169.0 and @gradio/client 2.7.1 match their recorded checksums.
- Upload sanitising: type detected from file bytes, size and 40 MP limits, decompression bomb,
  EXIF/GPS and ICC stripping, orientation.
- Signed URLs. Every arrow of the job state machine. The job API and SSE. Owner scoping.
  Consent and no-reroute rules.
- GPU probe parsing (on synthetic `nvidia-smi`/`rocm-smi` outputs) and engine eligibility.
- Validation gates V1/V3/V4 on good and deliberately broken GLBs.
- The worker loop end to end, using a **mock engine** that returns a fixture GLB: upload → job →
  worker → validation → signed download, plus cancel, errors and isolation.
- The Generate panel in the browser against that mock worker.
- The TRELLIS.2 adapter's wiring, against a **mock** of the upstream API.
- The Hugging Face demo flow, against a **mock** gradio client.

**Not tested** (needs hardware or access this project has not had yet):

- **Real AI inference.** No NVIDIA GPU with 24 GB or more has been available. The TRELLIS.2
  adapter and the worker Docker image are written and pinned, but neither has been run or built.
- A real Blender import (gate V2), and the BiRefNet segmenter.
- The live Hugging Face Space (unreachable from the development environment).
- The evaluation set and the ninja acceptance test (P1-11).

Mock code exists only under `tests/`, and a test fails if production code imports it.

## Limitations

- Single-owner mode: `MAKER_API_TOKEN` protects the API, but the browser UI does not yet send
  it. Full authentication is ticket P4-01.
- Licences: TRELLIS.2 is MIT. Its texture baker (nvdiffrast) is licensed for non-commercial use
  only, and DINOv3 requires attribution. Do not launch commercially until P5-06 is resolved; see
  `worker/README.md`.
- FBX, BLEND and USD are not supported. OBJ/STL/PLY are converted in the browser and may lose
  materials.
- The repository has no LICENSE file yet; the owner must choose one.
