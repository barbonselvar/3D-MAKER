# 3D Maker Studio

A responsive image-to-3D silhouette workspace with drag-and-drop upload, orbit/zoom controls, depth adjustment, sample shapes, wireframe, auto rotation, material colors, and GLB/OBJ/STL/PLY downloads.

## AI reconstruction (TRELLIS.2)

The studio now defaults to Microsoft TRELLIS.2 via its public Hugging Face Gradio API. Select an image, check the external-processing consent box, choose 512 or 1024 resolution, then Generate with AI. The integration creates a session, preprocesses the image, generates geometry/materials, extracts a textured GLB, and imports it into the viewer. Queue status and cancellation are supported. No token is bundled or requested by this frontend.

This is an external shared GPU demo, not a dedicated hosted model. Availability, authentication requirements, quota and waiting times are controlled by Hugging Face/Microsoft. If access is denied, open the official service link to inspect its current requirements. No local silhouette is silently substituted after an AI error. Images sent in AI mode leave the device; do not upload private or sensitive images.

Official references:
- https://huggingface.co/spaces/microsoft/TRELLIS.2
- https://huggingface.co/docs/hub/spaces-api-endpoints

## Other modes

- **Table builder:** an adjustable four-leg 3D table template, with width/depth/height controls. This does not reconstruct the uploaded image and is not AI.
- **Silhouette extrusion:** the earlier local 96×96 mask extrusion; useful for flat cutouts rather than furniture.
- **Import GLB:** load a model generated elsewhere (80 MB maximum), inspect it, and export it. GLB preserves textures; OBJ/STL/PLY may omit materials/textures.

## Validation

JavaScript syntax, HTML control IDs, table dimensions/parts, and all four local exporters were checked. The live TRELLIS.2 API endpoint signatures were checked. A complete remote AI generation and browser interaction were not verified in the development environment due to network/browser availability.

## Browser studio

Serve `app/static` with any static web server, for example:

```bash
python -m http.server 8000 --directory app/static
```

Open http://localhost:8000. Local modes process images in your browser. AI mode uploads the image to the external service after consent. Internet access is needed for Three.js modules from esm.sh and optional Google Fonts. WebGL support is required.

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

- Local silhouette mode cannot infer hidden geometry. AI reconstruction estimates hidden surfaces and may produce artifacts or inaccurate dimensions.
- The browser engine extrudes a 96×96 image mask, so outlines have stepped edges. The Python API uses contour extrusion.
- Transparent PNGs or isolated objects on plain backgrounds work best. Background separation is approximate.
- Exported meshes have no source-image texture. Material color is included where the target format supports it.
- No local GPU model or dedicated cloud GPU is bundled. TRELLIS.2 uses its external shared demo. FBX, BLEND, USD, and USDZ are not supported.

## Planned hybrid architecture

Local GPU model adapters, optional cloud AI adapters with server-side credentials, job queues, texture baking, remeshing, and Blender conversion remain future work.
