# 3D-MAKER on Windows with a local NVIDIA GPU

Generate textured 3D models from one image on your own Windows 10/11 PC. Everything runs natively:
no WSL, no Docker, no cloud, no paid API. The first local engine is **Stable Fast 3D (SF3D)**. It is
fast, needs much less memory than TRELLIS.2, and produces a textured mesh (base colour plus normal
map) as a GLB file that Blender can open.

## Status: what is and is not verified

| Part | Status |
| --- | --- |
| Setup and start scripts | Parse cleanly and pass PSScriptAnalyzer, including PowerShell 5.1 syntax rules. The check phase runs in CI on real Windows (no GPU) and on Linux with simulated tools. **The full install has not been run on a real Windows PC.** |
| Dependency set (Python 3.11, 64-bit) | Every pinned package has a Windows wheel; the full test suite passes with these exact versions (Linux, Python 3.11, and the Windows CI job) |
| Background removal (U²-Net on the CPU) | **Really run** in tests, on Linux and on the Windows CI job |
| Validation, job API, progress, cancel, downloads, UI | Tested end to end with a **mock** engine |
| Stable Fast 3D on a GPU | **NOT VERIFIED.** No GPU was available while this was built. Whether it fits in 6 GB is decided on your PC by the self-test. Nothing is assumed. |
| Compiling SF3D's two C++/CUDA extensions on Windows | **NOT VERIFIED.** Upstream calls Windows support "experimental". |

Please run the "Verify on your PC" procedure below and send back the results. That is the only way
to confirm the last two rows.

## What you need

| Item | Notes |
| --- | --- |
| Windows 10 or 11, 64-bit | |
| NVIDIA GPU with a current driver | Tested target: RTX 4050 Laptop GPU (6 GB). The driver must support CUDA 12.4 or newer (`nvidia-smi` shows "CUDA Version"). |
| Python **3.11**, 64-bit | One SF3D dependency (`gpytoolbox 0.2.0`) has Windows wheels only up to Python 3.11. Python 3.12 or 3.13 will not work. |
| Git, Node.js 18+ | Git downloads SF3D. Node runs the Khronos glTF validator; every model must pass it before you can download it. |
| Visual Studio 2022 Build Tools, "Desktop development with C++" | Compiles SF3D's two extensions |
| CUDA Toolkit **12.4** | Compiles the GPU texture baker. Without it, SF3D builds a CPU-only baker that then fails on the GPU; setup detects this. |
| Hugging Face account | SF3D's weights are gated. Accept the licence once at https://huggingface.co/stabilityai/stable-fast-3d and create a **read** token at https://huggingface.co/settings/tokens. |
| Disk space | Plan for roughly 20 GB in total (rough estimate, not measured): Build Tools, CUDA Toolkit, PyTorch and the model weights. |

Install the tools with winget from a normal PowerShell window, or let setup do it with `-InstallPrerequisites`:

```powershell
winget install -e --id Python.Python.3.11
winget install -e --id Git.Git
winget install -e --id OpenJS.NodeJS.LTS
winget install -e --id Microsoft.VisualStudio.2022.BuildTools --override "--wait --passive --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"
```

Install the CUDA Toolkit 12.4 from https://developer.nvidia.com/cuda-12-4-1-download-archive
(Windows → x86_64 → your Windows version → exe (local)). In the installer, choose **Custom**. If your
driver is newer than the one bundled with the toolkit, untick **Driver components**.

## Install

```powershell
git clone https://github.com/barbonselvar/3D-MAKER.git C:\3d-maker
cd C:\3d-maker
git checkout claude/3d-maker-blueprint-impl-vhvzu4
powershell -ExecutionPolicy Bypass -File scripts\windows\setup.ps1
```

Use a short folder path such as `C:\3d-maker`; compiling with MSVC can fail on long paths.

Setup runs nine numbered steps and prints **OK**, **WARN** or **FAIL** with the exact fix for each:

1. Windows, GPU, driver version and the CUDA version the driver supports.
2. Python 3.11, Git, Node.js.
3. Visual Studio Build Tools and the CUDA Toolkit. If anything is missing, setup lists it and stops before changing anything.
4. Creates `.venv-gpu` in the repository.
5. Installs PyTorch 2.6.0 (CUDA 12.4) and all pinned packages, then confirms PyTorch can see your GPU.
6. Downloads SF3D at commit `ff21fc4` and compiles `texture_baker` (with its CUDA kernel) and `uv_unwrapper`.
7. Signs in to Hugging Face (asks for the token, input hidden) and downloads the SF3D weights. It records
   the exact revision in `%LOCALAPPDATA%\3D-MAKER\worker.env`, then switches to offline mode.
8. Downloads the CPU background-removal model (U²-Net, about 170 MB) and runs an installation check.
9. **GPU self-test.** A real generation on a test photo, described below.

`-CheckOnly` runs only steps 1–3 and changes nothing. Re-running setup is safe.

## Start and use

```powershell
powershell -ExecutionPolicy Bypass -File scripts\windows\start.ps1
```

This starts everything on `127.0.0.1` only (the studio and API on port 8000, and the worker on port
8001 with a random per-run token) and opens the studio in your browser. Then:

1. Drop an image onto the upload area. The best results come from a single object on a plain
   background, or a PNG with transparency.
2. Choose **AI reconstruction**, then press **Generate**.
3. In **Local GPU**, *Stable Fast 3D* should show **Ready** with the measured limits, for example
   "textures up to 1024 px · 4.8 GB peak VRAM" (that example is illustrative, not a measurement).
4. Pick a texture size (512 uses the least memory) and press **Start on Local GPU**.
5. Follow the progress steps. When the model appears, orbit it in the viewer and press **↓ GLB**.
   You get the exact file that passed validation.
6. In Blender: *File → Import → glTF 2.0 (.glb/.gltf)*.

Stop the program with **Ctrl+C** in the PowerShell window.

## What the statuses mean

| Status | Meaning | What to do |
| --- | --- | --- |
| **Ready** | The self-test produced a valid textured GLB on this GPU. The texture-size list only offers sizes that worked. | Generate |
| **Not tested yet** | The worker is still running its self-test, which takes from a few seconds up to a minute or two | Wait, then reopen Generate |
| **Out of memory** | Even 512 px with low-VRAM mode ran out of GPU memory | Close games, browsers, video apps and anything else using the GPU, then `start.ps1 -SelfTest` |
| **Not installed** | Something is missing; the reason names it (weights, extensions, PyTorch…) | Re-run `setup.ps1` |
| **Unsupported** | This hardware cannot run the engine, for example TRELLIS.2 needs at least 24 GB | Use Stable Fast 3D |
| **Self-test failed** | A non-memory error; the message shows it | See Troubleshooting |

### How memory is handled on a 6 GB GPU

Upstream says SF3D uses "about 6 GB VRAM" at its default 1024 px texture. That is your whole card, so
nothing is assumed. At startup the worker tries these settings from best to most frugal, and keeps the
first one whose output passes validation:

1. 2048 px
2. 2048 px + low-VRAM mode
3. 1024 px
4. 1024 px + low-VRAM mode
5. 512 px + low-VRAM mode

The other memory-saving measures:
- Background removal runs on the CPU.
- One image is processed at a time.
- The transformer runs in bf16, as upstream does.
- Low-VRAM mode evaluates SF3D's 3D grid in chunks. The result is the same up to floating-point
  rounding; only the peak memory changes.
- `SF3D_OFFLOAD_IDLE=1` in `worker.env` moves the model to system RAM between jobs, freeing VRAM for
  other programs. This makes each job slower to start.

If a real job later runs out of memory, the worker steps one setting down and remembers it. The
results are cached in `%LOCALAPPDATA%\3D-MAKER\selftest.json`. The self-test re-runs automatically
after a driver, GPU, PyTorch or engine update; you can also force it with `start.ps1 -SelfTest`.

## Verify on your PC (please send back the results)

Run these commands in `C:\3d-maker` after setup has finished.

```powershell
# 1. What the worker sees
.\.venv-gpu\Scripts\python.exe -m worker.gpu_probe > probe.json

# 2. Real GPU self-test (repeat it once with nothing else open, to compare)
powershell -ExecutionPolicy Bypass -File scripts\windows\start.ps1 -SelfTest > selftest.json

# 3. One complete generation through the UI
powershell -ExecutionPolicy Bypass -File scripts\windows\start.ps1
```

**Expected results:**
- `probe.json` shows `"backend": "cuda"` and your GPU name.
- `selftest.json` shows `"status": "passed"` with `max_texture_size`, `low_vram`, `peak_vram_mb` and
  `seconds`, and an `attempts` list showing which settings ran out of memory.
- In the studio: Generate shows Stable Fast 3D **Ready**. A photo of a single object (a shoe, a mug, a
  toy) completes all steps up to **Export**, and the downloaded GLB opens in Blender with its colours.

**Please send back:**
- `probe.json` and `selftest.json`.
- The `manifest.json` of your model. Open
  `http://127.0.0.1:8000/api/v1/models/<model id>/files/manifest` while the program runs; the model
  id is in the job response, and the manifest is also listed in `GET /api/v1/models/<id>`.
- A screenshot of the result and of the model in Blender.
- Any red **FAIL** text from setup.

## Troubleshooting

| Symptom | Fix |
| --- | --- |
| `nvidia-smi was not found` | Install the NVIDIA driver from https://www.nvidia.com/Download/index.aspx, then reboot |
| `The driver supports CUDA 12.x, but PyTorch here needs CUDA 12.4` | Update the NVIDIA driver |
| `PyTorch cannot see the NVIDIA GPU` | A CPU-only PyTorch got installed. Delete `.venv-gpu` and run setup again. |
| `texture_baker was built without its CUDA kernel` | Install the CUDA Toolkit 12.4 (step 3 shows the link), open a new window, run setup again |
| `unsupported Microsoft Visual Studio version` from nvcc | Setup already adds `-allow-unsupported-compiler` for MSVC 14.40+. If it still fails, add the *MSVC v143 – VS 2022 C++ x64/x86 build tools (v14.39)* component in the Visual Studio Installer. |
| `error: Microsoft Visual C++ 14.0 or greater is required` | Install the Build Tools with "Desktop development with C++" |
| Compiler errors mentioning paths over 260 characters | Use a short folder such as `C:\3d-maker`, or enable long paths (`LongPathsEnabled` in the registry, admin required) |
| `Stable Fast 3D is a gated model…` | Accept the licence at https://huggingface.co/stabilityai/stable-fast-3d with the same account as your token |
| `DLL load failed` for `onnxruntime` or `torch` | Install the latest Microsoft Visual C++ Redistributable (x64) |
| Status **Out of memory** | Close GPU-heavy apps, choose 512 px, then run `start.ps1 -SelfTest`. On laptops, plug in power: some laptops limit the GPU on battery. |
| Every job fails with `V1 glTF-Validator skipped` | Node.js is missing. Install it, then run `npm ci --prefix worker`. |
| `Could not start the server on 127.0.0.1:8000` | Another program uses the port. Use `start.ps1 -Port 8100`. |
| `running scripts is disabled on this system` | Use `powershell -ExecutionPolicy Bypass -File …` as shown |
| The studio shows "Not tested yet" for a long time | Watch the PowerShell window for `[selftest]` lines; the first run also loads the model |

## Privacy and cost

- The worker API listens only on `127.0.0.1` and uses a random token for each run.
- After setup, `worker.env` sets `HF_HUB_OFFLINE=1` and `HF_HUB_DISABLE_TELEMETRY=1`, so generating
  never contacts Hugging Face.
- Images and models stay in `%LOCALAPPDATA%\3D-MAKER\data`.
- The only external option in the studio is the Hugging Face *public demo*. It needs explicit consent
  every time and is never used automatically. No paid service exists in this setup.

## Licences

| Component | Licence | What it means for you |
| --- | --- | --- |
| Stable Fast 3D weights and code | Stability AI Community License | Free for research and non-commercial use, and for commercial use by individuals and organisations under US$1M annual revenue. Commercial use requires registering with Stability AI (https://stability.ai/community-license). Above US$1M you need an enterprise licence. You own your outputs. If you distribute the model or a product using it, include the licence and the notice "This Stability AI Model is licensed under the Stability AI Community License, Copyright © Stability AI Ltd. All Rights Reserved", and show "Powered by Stability AI". The studio displays this on SF3D results. |
| rembg | MIT | |
| U²-Net weights | Apache-2.0 | |
| PyTorch | BSD-3-Clause | |
| Khronos glTF-Validator | Apache-2.0 | |
| TRELLIS.2 (24 GB GPUs only) | MIT, with non-commercial parts | See `worker/README.md` |

## Uninstall

Delete `.venv-gpu` and `third_party` in the repository folder, `%LOCALAPPDATA%\3D-MAKER`, the model
cache `%USERPROFILE%\.cache\huggingface\hub\models--stabilityai--stable-fast-3d`, and `%USERPROFILE%\.u2net`.
Uninstall the Build Tools, CUDA Toolkit, Python and Node.js from Windows Settings if nothing else uses them.

## TRELLIS.2

TRELLIS.2 stays available as the high-quality engine for GPUs with 24 GB or more: the Linux Docker
worker in `worker/README.md`, or a cloud GPU you approve later. On a 6 GB GPU it shows **Unsupported**,
with the reason.
