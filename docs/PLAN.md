# 3D-MAKER — reconciled development plan

Sources combined here:

- **Implementation prompt** (`CLAUDE_CODE_3D_MAKER_IMPLEMENTATION_PROMPT.txt`) — the product requirements.
- **Engineering Blueprint** (24 Oct 2026) — the proposed architecture, engine choice, backlog and acceptance gates.

Rule used for every conflict: keep the product goal from the prompt, take the safest technically feasible
implementation from the blueprint, and never drop a feature silently — anything deferred is listed in
"Deferred, not dropped" below.

## 1. Audit check (2026-10-24, commit 574b602)

The blueprint's repository audit was re-verified against the code before anything was changed:

| Blueprint claim | Re-check | Result |
| --- | --- | --- |
| 6 tracked files, browser-only studio calling the HF TRELLIS.2 Space | `git ls-files`, read every file | Confirmed |
| `/api/health` → `ai_enabled:false`; circle → 60-face watertight GLB; traversal → 404; noisy JPEG → 422 | FastAPI TestClient smoke test | Confirmed |
| `node --check` passes on the inline module | extracted script + `node --check` | Confirmed |
| Colour tool tints AI textures; export re-serialises a rescaled scene | read `#color` and export handlers | Confirmed |
| Unpinned `@gradio/client`, Three.js from esm.sh at runtime | read imports | Confirmed |
| `#wire`/`#color` handlers assigned twice; `oldSelect`, `sourceCanvas`, `hasAlpha` unused; `subprocess` unused | read code, `ruff` (4 errors) | Confirmed |
| No LICENSE, tests, CI, Dockerfile, DB, queue, GPU code | `ls` | Confirmed |
| GitHub issues / PRs | GitHub API | None exist |

Environment of this session: no NVIDIA/AMD GPU, no Blender, Hugging Face hosts blocked by the network
policy, npm registry reachable, Chromium + Playwright available. Consequences are listed under blockers.

## 2. Contradictions and how they are resolved

| # | Prompt says | Blueprint says | Resolution |
| --- | --- | --- | --- |
| C1 | Job states: `queued, preparing, running, postprocessing, exporting, completed, failed, cancelled` | `queued, assigned, running, validating, succeeded, failed, cancelled, expired` | **Union, prompt names win.** `queued → assigned → preparing → running → postprocessing → validating → exporting → completed`; terminals `failed`, `cancelled`, `expired`. `assigned` and `expired` come from the blueprint (needed for worker claims and stale queues); `validating` keeps the blueprint's V1–V4 gate visible; `completed` replaces `succeeded`. A finer `stage` string (segment, infer, extract, cleanup, bake, validate, export) is kept for progress UI. |
| C2 | Phase 1 = homepage UX (two buttons) before backend work | Nothing cosmetic before the vertical slice; homepage is P3-03 | **Blueprint order** (user instruction: real AI over cosmetics). The two-button homepage still ships: Generate panel in P1-10, final layout in P3-03. Not dropped. |
| C3 | One dedicated branch | One branch per ticket | Single designated branch `claude/3d-maker-blueprint-impl-vhvzu4` (session constraint), **one commit per ticket** with the ticket id in the subject so each is reviewable/revertible. |
| C4 | Internal GPU detection: CUDA / ROCm / MPS | Probe only `nvidia-smi` + CUDA | **Probe all three**; the engine table decides eligibility. TRELLIS.2 and Hunyuan3D need CUDA; ROCm and MPS are reported but only qualify for SF3D (MPS experimental upstream) once P5-05 lands; otherwise a clear `NO_SUITABLE_GPU` reason. |
| C5 | "Internal GPU with CPU fallback" | No practical CPU path for TRELLIS.2 | CPU fallback = SF3D labelled "Preview quality", or refusal. Never silhouette/procedural. |
| C6 | Signed artifact access + job ownership in backend-contracts phase | Auth in P4-01, S3 presign in P4-02 | **Pulled forward the cheap parts:** HMAC-signed, 10-minute download URLs in the local storage backend now; `owner_id` on every row and every query filtered by owner now (single local owner until P4-01 adds real auth). |
| C7 | Real PBR, "treat glow as emissive when possible" | TRELLIS.2 outputs base colour/roughness/metallic/opacity only; emissive eyes = optional heuristic, off by default | Promise the four channels. Emissive eyes = P5-03, opt-in, labelled heuristic. |
| C8 | Mocks allowed if labelled | Stub engine must be impossible in production | Both: `tests/stub_engine.py` lives only under `tests/`, is named a mock in output/manifest, and a test asserts the production engine registry cannot load it. |
| C9 | "Exactly two primary buttons" and "preserve existing features" | Extras move to "More tools" | Two primary buttons; table builder, silhouette, GLB import under "More tools"; HF demo becomes an External provider choice. |
| C10 | Evaluate Hunyuan3D 2.x "and newer" | 2.5/3.0/3.1 have no open weights | Compare 2.0/2.1 only; re-check before P5-04. |
| C11 | "Meshy-inspired" | Retopology, rigging, text-to-3D out of scope | Out of scope until the slice passes evaluation; retopology placeholder stays in pipeline stage 8. |
| C12 | Optional OBJ/STL/FBX/BLEND only after conversion tests | Same, Blender-gated (V7) | Agreed. Existing client-side OBJ/STL/PLY exporters are real conversions (not renamed files) and stay for local modes and imported/AI models, now exported at original scale (P0-02). |
| C13 | Owner must choose a LICENSE (blueprint P0-04) | — | **Not decided by the agent.** Choosing a licence is the owner's legal decision; P0-04 ships everything except the LICENSE file and lists it as a blocker. |

## 3. Requirements present in only one document

Added to the backlog so neither document loses content:

| From | Requirement | Placed in |
| --- | --- | --- |
| Prompt | Viewer lighting presets, frame-to-fit/bounding box, screenshot, mobile controls, loading/error states | New ticket **P3-04** (viewer polish) |
| Prompt | Accessibility (keyboard, focus, ARIA, contrast) | Acceptance criterion on every UI ticket; audit in **P4-08** |
| Prompt | `.env.example` with no secrets; reproducible setup scripts | **P0-05** (`.env.example`), P1-02, P4-07 |
| Prompt | Concurrency limits per worker/user | P1-07 (one job per GPU), P4-04 (3 concurrent per user) |
| Prompt | Webhooks as applicable for external providers | P2-01 (Modal: polling; webhook only if provider supports it) |
| Prompt | Remove/disable misleading fallback success paths | Already true in the UI ("No local substitute"); enforced by API: no engine → error code, tested |
| Blueprint | Licence trap (RMBG-2.0 CC BY-NC, nvdiffrast non-commercial, DINOv3 attribution) | P1-03, P5-06, manifest licence list |
| Blueprint | Spend ledger before invoking a provider, monthly cap | P2-03 |
| Blueprint | Evaluation set (12 inputs) and gates V1–V8, H1 | P1-11 |
| Blueprint | Supply-chain pinning (vendored Three.js, pinned gradio client) | P0-03 |
| Blueprint | Upload retention (30 days), output TTL (24 h) | P0-06, P4-05 |

## 4. Merged phase map

| Blueprint phase | Covers prompt phases | GPU needed? |
| --- | --- | --- |
| P0 Hygiene | Prompt 0 (audit/baseline) + bug fixes | No |
| P1 Internal GPU slice | Prompt 2 (contracts), 3 (worker), 4 (real inference), part of 5 & 7 | P1-02/03/04, P1-10 DoD, P1-11 baseline: **yes**; rest no |
| P2 External GPU | Prompt 8 | Paid provider — **needs owner approval** |
| P3 View + homepage | Prompt 1 (homepage), 6 (viewer) | No |
| P4 Hardening | Prompt 9 | No |
| P5 Formats & quality | Prompt 5, 7 | Partly |

## 5. Deferred, not dropped

Homepage two-button layout (P3-03) · gallery (P3-01/02) · viewer presets & screenshot (P3-04) ·
External GPU/Modal (P2, needs approval) · auth, S3, Postgres, rate limits, retention, metrics, compose (P4) ·
OBJ/STL/PLY via Blender, FBX, BLEND, emissive eyes, Hunyuan3D-2.1, SF3D, nvdiffrast replacement (P5) ·
quad retopology, rigging, text-to-3D (out of scope).

## 6. Blockers needing the owner

1. **GPU**: no NVIDIA ≥ 24 GB GPU is available to this session. P1-02/03/04 are written but NOT VERIFIED; P1-10's
   end-to-end DoD and P1-11's baseline need a GPU host or approval to use a paid provider (P2).
2. **LICENSE**: owner must pick one (MIT/Apache-2.0/proprietary…).
3. **Commercial intent**: decides whether RMBG-2.0/nvdiffrast are acceptable (R1).
4. **Ninja reference image**: needed in `eval/inputs/`.
5. **External GPU budget and provider account**: needed before any P2 work touches a paid service.
6. **Blender**: not installed here; Blender import checks (V2/V7) are written but run only where `BLENDER_BIN` exists.

## 7. Status (2026-10-24, branch `claude/3d-maker-blueprint-impl-vhvzu4`)

| Ticket | Status | Evidence |
| --- | --- | --- |
| P0-01 colour tool | Done | `tests/ui/test_textured_models.py` |
| P0-02 original GLB bytes | Done | SHA-256 of download = fixture |
| P0-03 pinned/vendored JS | Done | `tests/test_vendor.py`, UI suite runs offline |
| P0-04 dead code, leftover file | Done, **except LICENSE (owner decision)** | ruff clean |
| P0-05 tests + CI | Done | GitHub Actions green |
| P0-06 output TTL | Done | fake-clock test |
| P1-01 GPU probe + gpu-targets | Done (synthetic outputs; no real GPU) | `tests/test_gpu_probe.py` |
| P1-02 worker image | **Written, NOT BUILT** (no Docker daemon/GPU) | `worker/Dockerfile`, `worker/README.md` |
| P1-03 segmenter | Rules done and tested; model **NOT RUN** | `tests/test_segment_cleanup.py` |
| P1-04 TRELLIS.2 adapter | **Written, NOT VERIFIED**; wiring tested against a mock of upstream | `tests/test_trellis2_adapter.py` |
| P1-05 DB, state machine, job routes | Done | `tests/test_jobs_state_machine.py`, `tests/test_jobs_api.py` |
| P1-06 storage + uploads | Done | `tests/test_uploads.py`, `tests/test_storage.py` |
| P1-07 worker loop + internal API | Done (mock engine) | `tests/test_worker_integration.py` |
| P1-08 validation | V1/V3/V4 done; V2 wrapper tested with mock Blender | `tests/test_validate.py` |
| P1-09 ES modules | Done | UI suite passes before and after |
| P1-10 Generate panel | UI done and tested against the mock worker; **GPU end-to-end pending** | `tests/ui/test_generate_panel.py` |
| P1-11 evaluation baseline | **Not started: needs GPU, Blender, ninja image** | — |

Next, in order: (1) on a ≥ 24 GB NVIDIA host, follow `worker/README.md` "Verify on a GPU host",
(2) P1-11 harness and baseline, (3) P3 gallery + two-button homepage (no GPU needed),
(4) P2 External GPU only after owner approval of provider and budget, (5) P4 hardening before
any public exposure.

### Native Windows local GPU (requested 2026-10-24)

| Item | Status | Evidence |
| --- | --- | --- |
| Stable Fast 3D engine (`worker/engines/sf3d.py`) | Written against SF3D `ff21fc4`; wiring tested against a mock of upstream; **NOT VERIFIED on a GPU** | `tests/test_sf3d_adapter.py` |
| Engine statuses: Ready, Unsupported, Out of memory, Not installed (plus Not tested yet, Self-test failed) | Done | `tests/test_gpu_probe.py`, `tests/ui/test_generate_panel.py` |
| GPU self-test on the user's machine; texture limit; out-of-memory step-down | Done (mock engine) | `tests/test_selftest.py` |
| Memory options: 512–2048 px textures, low-VRAM chunked grid query, bf16, CPU background removal, sequential jobs, idle offload | Done; chunking equivalence (to floating-point rounding) tested with NumPy stand-ins | `tests/test_sf3d_adapter.py` |
| Windows probe (nvidia-smi locations, `[N/A]`, driver CUDA vs PyTorch CUDA) | Done (synthetic outputs) | `tests/test_gpu_probe.py` |
| One-command local stack | Done | `tests/test_local_stack.py` |
| `scripts/windows/setup.ps1`, `start.ps1` | Written; parse/lint clean; check phase executed with simulated tools and on Windows CI; **full install not run on a real PC** | `tests/test_windows_scripts.py`, CI `windows` job |
| Windows dependency lock (CPython 3.11, win_amd64) | Resolved; test suite passes with the same pins | `worker/windows/constraints-win-py311.txt` |
| CPU background removal (U²-Net) | **Really run** | `tests/test_segment_cleanup.py::test_rembg_u2net_real_on_cpu` |
| TRELLIS.2 | Unchanged; shows Unsupported on a 6 GB GPU | — |

Next: the owner runs `docs/WINDOWS.md` → "Verify on your PC" on the RTX 4050 laptop and sends back
`probe.json`, `selftest.json` and one model manifest.
