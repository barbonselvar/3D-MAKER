# Deploy the public website to Vercel (free Hobby plan)

The public website is the **static studio only** (`app/static`). It runs entirely in the visitor's
browser and needs no server and no GPU:

- **Local tools work:** table builder, silhouette extrusion, GLB import, viewer, and GLB/OBJ/STL/PLY export.
- **Free Hugging Face public demo:** available after the visitor consents. The image goes from their
  browser to Hugging Face.
- **Local GPU:** the panel explains that it runs in 3D-MAKER installed on the visitor's own computer
  (`docs/WINDOWS.md`, then http://127.0.0.1:8000). A public HTTPS page cannot reach a visitor's
  local worker, because browsers block it.

Nothing private is deployed. `vercel.json` sets `framework: null`, no build, no install, no
functions, and `outputDirectory: app/static`. `.vercelignore` uploads only `app/static` and
`vercel.json`. The FastAPI backend, the private worker API (`/internal/v1`), the Windows GPU worker,
the scripts and all dependency files never leave the repository. `tests/test_vercel_config.py`
guards both files.

Checked offline with the real Vercel CLI (`vercel build`, v65.2.0): the build succeeds and produces
static output only (no serverless functions), and the security headers are applied.

## Steps (in the Vercel dashboard)

1. Sign in at https://vercel.com with GitHub (Hobby plan; no payment method needed).
2. **Add New… → Project → Import Git Repository.** If asked, install the Vercel GitHub app and grant
   it access to `barbonselvar/3D-MAKER` only.
3. **Project Name:** `3d-maker`.
4. **Framework Preset:** *Other*. **Root Directory:** leave as `./`.
5. Leave Build/Output/Install settings at their defaults: `vercel.json` sets them (no build,
   output `app/static`).
6. Add **no** environment variables. The public site needs none, and no secret belongs here.
7. **Deploy.** Production deploys come from `main`; other branches get preview URLs.

Hobby resources used: static files from the CDN only. No functions, no storage, no paid add-ons.
Do not enable paid features (Pro trial, extra bandwidth, analytics add-ons).

## After deploying

Your URL will be `https://3d-maker.vercel.app`, or `https://3d-maker-<suffix>.vercel.app` if that
name is taken. Check that:

- `/` loads, and **Table builder → Build** shows a table you can download as GLB.
- **AI reconstruction → Generate** shows the Local GPU explanation and the Hugging Face demo with
  its consent box.
- `/api/health`, `/api/v1/gpu-targets` and `/internal/v1/claim` return **404** (no backend is exposed).
