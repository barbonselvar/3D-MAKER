# 3D Maker — Hybrid Image-to-3D Studio

A runnable starter for an image-to-3D application. **Current implementation is silhouette extrusion, NOT AI image-to-3D reconstruction.** It creates a genuine mesh from a segmented image outline, previews it in Three.js, and exports GLB/OBJ/STL/PLY.

## Quick start

Requires Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Open http://localhost:8000. Internet is needed to load the Three.js browser modules from esm.sh.

## Current limitations
- One-view silhouette extrusion cannot infer hidden geometry or accurate depth.
- Plain background or transparent PNG recommended. Automatic corner-color segmentation is approximate.
- OBJ/STL/PLY exports do not include source-image textures.
- No FBX, BLEND, USD, USDZ yet. Blender CLI conversion can be added as an optional backend.
- No local AI model or cloud provider is bundled, and no API keys are required.
- Generated files are stored under the OS temp directory; configure `MAKER_OUTPUT` to change this. Add cleanup and authentication before public deployment.

## Planned hybrid architecture
1. Local AI adapter: isolated worker for a separately installed open-source image-to-3D model, with GPU availability checks and model license review.
2. Cloud AI adapter: provider credentials via environment variables, never client-side.
3. Job queue, progress tracking, retries, GPU/CPU modes.
4. Texture baking, mesh decimation/remeshing, Blender conversion for FBX/BLEND, and optional USD exports.
5. Persistent project library, multi-image reconstruction, model editor and PBR materials.

## GitHub
Create an empty `3d-maker` repository, then from this folder:

```bash
git init
git add .
git commit -m "Initial 3D Maker studio MVP"
git branch -M main
git remote add origin https://github.com/barbonselvar/3d-maker.git
git push -u origin main
```
