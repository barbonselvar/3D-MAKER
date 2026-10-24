# Internal GPU worker

> **Windows PC with a smaller NVIDIA GPU (for example a 6 GB laptop)?** Use the native Windows setup with
> Stable Fast 3D instead: [`docs/WINDOWS.md`](../docs/WINDOWS.md). This page covers the Linux/Docker worker
> for TRELLIS.2 on GPUs with 24 GB or more.

Runs image-to-3D jobs for the API: claims a job, segments the image, runs the engine, cleans the
mesh, validates the GLB (gates V1–V4), and uploads the GLB plus `manifest.json`. It never
produces a non-AI stand-in: every failure is reported with an error code.

## Status

| Part | Status |
| --- | --- |
| Job loop, heartbeats, cancel, error mapping (`main.py`) | Tested with a mock engine (`tests/test_worker_integration.py`) |
| Capability probe (`gpu_probe.py`) | Tested on synthetic `nvidia-smi`/`rocm-smi` outputs; **not run on real GPUs** |
| Validation V1/V3/V4 (`validate.py`) | Tested on good and known-bad fixtures with the real Khronos validator |
| Validation V2 Blender import | Wrapper tested with a mock Blender; **real Blender not run** |
| Segmenter rules (`segment.py`) | Licence gate, alpha bypass, coverage limits tested with a mock model; **BiRefNet not run** |
| Cleanup (`postprocess.py`) | Tested (CPU) |
| TRELLIS.2 adapter (`engines/trellis2.py`) | Wiring tested against a mock of the upstream API; **NOT VERIFIED: no ≥ 24 GB NVIDIA GPU was available** |
| Docker image (`Dockerfile`) | **NOT BUILT: no Docker daemon or GPU was available** |

## What is pinned

| Component | Pin |
| --- | --- |
| Base image | `nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04@sha256:622e78a1…` (manifest list) |
| TRELLIS.2 code | `microsoft/TRELLIS.2@75fbf0183001ed9876c8dbb35de6b68552ee08bd` (2026-06-05) |
| TRELLIS.2 weights | `microsoft/TRELLIS.2-4B` at `TRELLIS2_WEIGHTS_REVISION` (**required** env; a Hugging Face commit hash) |
| nvdiffrast / nvdiffrec / CuMesh / FlexGEMM / utils3d | exact commits in the Dockerfile `ARG`s (upstream installs branch heads) |
| Python packages | `constraints-gpu.txt` (torch 2.6.0 + cu124, flash-attn 2.7.3, and the rest) |
| Blender | 4.2.3 LTS, checked against blender.org's SHA-256 list at build time |
| Node.js / glTF-Validator | 22.11.0 (SHA-256 pinned) / `gltf-validator@2.0.0-dev.3.10` (lockfile) |

Not yet pinned: the DINOv3 conditioning model and the BiRefNet segmenter are loaded by name
(set `SEGMENTER_REVISION` to pin the segmenter). Record both revisions after the first GPU run.

## Licences that apply to every output (risk R1)

- TRELLIS.2 code and weights: MIT.
- DINOv3 (image conditioning): gated, custom licence; outputs must credit "Built with DINOv3".
  Accept the licence on Hugging Face and pass `HF_TOKEN`.
- nvdiffrast (texture baking): NVIDIA Source Code License — **non-commercial**. Replace or license
  before any commercial launch (ticket P5-06).
- Segmenter: default `ZhengPeng7/BiRefNet` (weights licence not yet verified by this project).
  `briaai/RMBG-2.0` (CC BY-NC 4.0) is refused unless `SEGMENTER_ALLOW_NONCOMMERCIAL=1`.
  TRELLIS.2's own background remover is never instantiated.

Each `manifest.json` lists these notices.

## Verify on a GPU host (P1-02 / P1-04 definition of done)

Requirements: Linux, NVIDIA GPU with ≥ 24 GB (A100/H100 verified upstream; L40S/A6000 expected),
NVIDIA driver supporting CUDA 12.4, Docker with the NVIDIA container toolkit, ~60 GB disk.

```bash
# 1. Build (expect 1–2 h: flash-attn and the CUDA extensions compile). Keep the log.
docker build -f worker/Dockerfile -t 3d-maker-worker:dev . 2>&1 | tee worker-build.log

# 2. Capability probe: must report backend "cuda" and the GPU's free VRAM.
docker run --rm --gpus all 3d-maker-worker:dev --probe-only

# 3. Smoke inference: writes out/trellis2-4b.glb and its manifest, exits 0 only if V1–V4 pass.
docker run --rm --gpus all -e HF_TOKEN -e TRELLIS2_WEIGHTS_REVISION=<commit> \
  -v maker-models:/models -v "$PWD/out:/out" 3d-maker-worker:dev --smoke

# 4. Record: image digest, build log, manifest.json (timings, peak VRAM, validation) -> commit them.
docker inspect --format '{{index .RepoDigests 0}}' 3d-maker-worker:dev
```

Then replace `constraints-gpu.txt` with `pip freeze` from the built image.

## Run as the internal worker

The API host runs the worker API on a private interface (never the public listener):

```bash
MAKER_WORKER_TOKENS="gpu-1:<long random token>" uvicorn app.internal:app --host 10.0.0.5 --port 8001
```

The GPU host runs:

```bash
docker run -d --gpus all --restart unless-stopped \
  -e MAKER_INTERNAL_URL=http://10.0.0.5:8001 -e MAKER_WORKER_TOKEN=<token> \
  -e HF_TOKEN -e TRELLIS2_WEIGHTS_REVISION=<commit> -v maker-models:/models 3d-maker-worker:dev
```

The worker heartbeats every 15 s; the Generate panel offers "Internal GPU" only while a heartbeat
is under 45 s old and the probe shows an engine that fits. One job runs per GPU at a time.

## Environment

| Variable | Meaning |
| --- | --- |
| `MAKER_INTERNAL_URL`, `MAKER_WORKER_TOKEN` | Worker API address and this worker's token |
| `WORKER_ENGINES` | Comma list, default `trellis2-4b` |
| `TRELLIS2_WEIGHTS_REVISION` | Required Hugging Face commit of `microsoft/TRELLIS.2-4B` |
| `HF_TOKEN` | Needed for gated weights (DINOv3) |
| `SEGMENTER`, `SEGMENTER_REVISION`, `SEGMENTER_ALLOW_NONCOMMERCIAL` | Segmenter choice, pin, and the non-commercial opt-in |
| `SEGMENT_CONTRAST_BOOST=1` | Segment a contrast-boosted copy (dark subjects); textures keep original colours |
| `BLENDER_BIN`, `VALIDATION_REQUIRE_BLENDER` | Gate V2; required in the image |

## Mock code

`tests/stub_engine.py` returns a fixture GLB so the plumbing can be tested without a GPU. It lives
only under `tests/`, is not in `engines.PRODUCTION_ENGINES`, and its output is marked
`"mock": true` in the manifest. A test fails if production code imports it.
