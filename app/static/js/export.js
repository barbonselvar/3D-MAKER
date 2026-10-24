// Downloads: original GLB bytes when the model came from a file/AI, otherwise client-side conversion.
import { GLTFExporter } from 'three/addons/exporters/GLTFExporter.js';
import { OBJExporter } from 'three/addons/exporters/OBJExporter.js';
import { PLYExporter } from 'three/addons/exporters/PLYExporter.js';
import { STLExporter } from 'three/addons/exporters/STLExporter.js';
import { toast } from './dom.js';
import { state } from './state.js';
import { withoutViewerFit } from './viewer.js';

function download(data, ext) {
  const blob = new Blob([data], { type: 'application/octet-stream' });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = '3d-maker-' + state.name.replace(/[^a-z0-9_-]/gi, '_') + '.' + ext;
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 10000);
}

const converters = {
  glb: async (m) => new GLTFExporter().parseAsync(m, { binary: true }),
  obj: (m) => new OBJExporter().parse(m),
  stl: (m) => new STLExporter().parse(m, { binary: true }),
  ply: (m) => new PLYExporter().parse(m, undefined, { binary: true }),
};

export function initExport() {
  document.querySelectorAll('[data-export]').forEach(
    (b) =>
      (b.onclick = async () => {
        if (!state.generated) return;
        const ext = b.dataset.export;
        b.disabled = true;
        try {
          if (ext === 'glb' && state.sourceBytes) {
            download(state.sourceBytes, 'glb');
            toast('Original GLB downloaded');
            return;
          }
          download(await withoutViewerFit(converters[ext]), ext);
          toast(ext.toUpperCase() + ' downloaded');
        } catch (e) {
          toast('Export failed: ' + e.message);
        } finally {
          b.disabled = false;
        }
      }),
  );
}
