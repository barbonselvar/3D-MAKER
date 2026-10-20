# 3D Maker Studio

A responsive image-to-3D silhouette workspace with drag-and-drop upload, orbit/zoom controls, depth adjustment, sample shapes, wireframe, auto rotation, material colors, and GLB/OBJ/STL/PLY downloads.

## Browser studio

Serve `app/static` with any static web server, for example:

```bash
python -m http.server 8000 --directory app/static
```

Open http://localhost:8000. The studio processes images locally in your browser; it does not upload source images. Internet access is needed for Three.js modules from esm.sh and optional Google Fonts. WebGL support is required.

## Optional Python API

The original FastAPI silhouette backend is retained for API clients. The redesigned studio uses its own browser engine, independently of this API.

Requires Python 3.10+:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

The studio is served at http://localhost:8000. API endpoints: `GET /api/health`, `POST /api/generate` (multipart `image` and `depth`), and `GET /api/files/{id}/{filename}`. API output is stored under the OS temporary directory, configurable with `MAKER_OUTPUT`. Add authentication and cleanup before exposing the API publicly.

## Limitations

- This is silhouette extrusion, not AI image-to-3D reconstruction; hidden geometry and true depth are not inferred.
- The browser engine extrudes a 96×96 image mask, so outlines have stepped edges. The Python API uses contour extrusion.
- Transparent PNGs or isolated objects on plain backgrounds work best. Background separation is approximate.
- Exported meshes have no source-image texture. Material color is included where the target format supports it.
- No local AI model, cloud AI provider, FBX, BLEND, USD, or USDZ integration is bundled.

## Planned hybrid architecture

Local GPU model adapters, optional cloud AI adapters with server-side credentials, job queues, texture baking, remeshing, and Blender conversion remain future work.
