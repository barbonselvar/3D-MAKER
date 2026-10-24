// Generation modes: AI (public demo), table builder and silhouette extrusion; plus GLB import.
import * as THREE from 'three';
import { cancelCurrent, onBusyChange, setBusy } from './busy.js';
import { $, setStatus, toast } from './dom.js';
import { openPanel } from './generate.js';
import { state } from './state.js';
import { loadGLB, mat, replaceObject } from './viewer.js';

function build(image, depth) {
  const n = 96;
  const c = document.createElement('canvas');
  c.width = c.height = n;
  const x = c.getContext('2d', { willReadFrequently: true });
  const scale = (n - 8) / Math.max(image.width, image.height);
  const w = image.width * scale;
  const h = image.height * scale;
  x.drawImage(image, (n - w) / 2, (n - h) / 2, w, h);
  const d = x.getImageData(0, 0, n, n).data;
  const mask = new Uint8Array(n * n);
  /* Alpha is determined inside the source bounds, excluding padding. */
  let transparent = false;
  for (let y = Math.ceil((n - h) / 2); y < (n + h) / 2 - 1; y++)
    for (let xx = Math.ceil((n - w) / 2); xx < (n + w) / 2 - 1; xx++) if (d[(y * n + xx) * 4 + 3] < 240) transparent = true;
  const corners = [
    [Math.ceil((n - w) / 2), Math.ceil((n - h) / 2)],
    [Math.floor((n + w) / 2) - 1, Math.ceil((n - h) / 2)],
    [Math.ceil((n - w) / 2), Math.floor((n + h) / 2) - 1],
    [Math.floor((n + w) / 2) - 1, Math.floor((n + h) / 2) - 1],
  ];
  const bg = [0, 1, 2].map((k) => corners.reduce((sum, [a, b]) => sum + d[(b * n + a) * 4 + k], 0) / 4);
  let count = 0;
  for (let i = 0; i < n * n; i++) {
    const o = i * 4;
    mask[i] = d[o + 3] > 90 && (transparent || Math.hypot(d[o] - bg[0], d[o + 1] - bg[1], d[o + 2] - bg[2]) > 42) ? 1 : 0;
    count += mask[i];
  }
  if (count < 12 || count > n * n * 0.9) throw Error('Could not isolate the object. Use a transparent PNG or a plain background.');
  const positions = [];
  const unit = 2.4 / n;
  const z = depth / 2;
  const quad = (a, b, c, d) => positions.push(...a, ...b, ...c, ...a, ...c, ...d);
  for (let y = 0; y < n; y++)
    for (let x = 0; x < n; x++) {
      if (!mask[y * n + x]) continue;
      const l = (x - n / 2) * unit;
      const r = l + unit;
      const t = (n / 2 - y) * unit;
      const b = t - unit;
      quad([l, b, z], [r, b, z], [r, t, z], [l, t, z]);
      quad([r, b, -z], [l, b, -z], [l, t, -z], [r, t, -z]);
      if (x === 0 || !mask[y * n + x - 1]) quad([l, b, -z], [l, b, z], [l, t, z], [l, t, -z]);
      if (x === n - 1 || !mask[y * n + x + 1]) quad([r, b, z], [r, b, -z], [r, t, -z], [r, t, z]);
      if (y === 0 || !mask[(y - 1) * n + x]) quad([l, t, z], [r, t, z], [r, t, -z], [l, t, -z]);
      if (y === n - 1 || !mask[(y + 1) * n + x]) quad([l, b, -z], [r, b, -z], [r, b, z], [l, b, z]);
    }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geo.computeVertexNormals();
  geo.center();
  return geo;
}

function tableModel() {
  const group = new THREE.Group();
  const wood = new THREE.MeshStandardMaterial({ color: 0x964c27, roughness: 0.52, metalness: 0.05 });
  const w = +$('#tw').value;
  const d = +$('#td').value;
  const h = +$('#th').value;
  function box(a, b, c, x, y, z) {
    const m = new THREE.Mesh(new THREE.BoxGeometry(a, b, c), wood);
    m.position.set(x, y, z);
    group.add(m);
  }
  box(w, 0.12, d, 0, h / 2, 0);
  for (const x of [-1, 1])
    for (const z of [-1, 1]) {
      const leg = new THREE.Mesh(new THREE.CylinderGeometry(0.065, 0.09, h, 20), wood);
      leg.position.set(x * (w / 2 - 0.15), 0, z * (d / 2 - 0.15));
      group.add(leg);
      const foot = new THREE.Mesh(new THREE.SphereGeometry(0.105, 16, 10), wood);
      foot.scale.y = 0.45;
      foot.position.set(leg.position.x, -h / 2 + 0.04, leg.position.z);
      group.add(foot);
    }
  box(w - 0.18, 0.18, 0.07, 0, h / 2 - 0.14, d / 2 - 0.12);
  box(w - 0.18, 0.18, 0.07, 0, h / 2 - 0.14, -d / 2 + 0.12);
  box(0.07, 0.18, d - 0.18, w / 2 - 0.12, h / 2 - 0.14, 0);
  box(0.07, 0.18, d - 0.18, -w / 2 + 0.12, h / 2 - 0.14, 0);
  return group;
}

export function updateMode() {
  const m = $('#mode').value;
  $('#aisettings').hidden = m !== 'ai';
  $('#tablesettings').hidden = m !== 'table';
  $('#localsettings').hidden = m !== 'silhouette';
  $('#modehelp').textContent =
    m === 'ai'
      ? 'AI reconstructs shape and PBR materials on an Internal GPU (your own worker) or an External GPU.'
      : m === 'table'
        ? 'Build a true 3D table with separated legs. Adjustable template, not image reconstruction.'
        : 'Flat silhouette with thickness. This mode cannot reconstruct furniture.';
  $('#generate').textContent = m === 'ai' ? '✦ Generate' : m === 'table' ? 'Build editable table' : 'Extrude silhouette';
  $('#generate').disabled = state.busy || (m !== 'table' && !state.img);
}

async function generate() {
  if (state.busy) return;
  const mode = $('#mode').value;
  if (mode !== 'table' && !state.img) return;
  if (mode === 'ai') return openPanel(); // choose Internal GPU or External GPU
  setBusy(true);
  try {
    if (mode === 'table') {
      state.name = 'table';
      replaceObject(tableModel(), 'Editable table', 'Local table template · not AI');
      setStatus('Four-leg 3D table created. Adjust its proportions above.');
    } else {
      await new Promise((r) => setTimeout(r, 30));
      replaceObject(new THREE.Mesh(build(state.img, +$('#depth').value), mat), state.name, 'Silhouette extrusion · not AI');
      setStatus('Silhouette ready. For full geometry, choose AI reconstruction.');
    }
  } catch (e) {
    setStatus(e.message);
  } finally {
    setBusy(false);
  }
}

async function importGLB(e) {
  const f = e.target.files[0];
  if (!f) return;
  if (state.busy) return toast('Wait for the current job.');
  if (f.size > 80 * 1024 * 1024) return toast('Maximum GLB size is 80 MB.');
  try {
    state.name = f.name.replace(/\.glb$/i, '');
    const bytes = await f.arrayBuffer();
    replaceObject(await loadGLB(bytes.slice(0)), state.name, 'Imported GLB', bytes);
    setStatus('Model imported. Ready to explore and export.');
  } catch (err) {
    setStatus('GLB import failed: ' + err.message);
  }
}

export function initModes() {
  $('#depth').oninput = () => ($('#depthval').value = $('#depth').value);
  $('#mode').onchange = updateMode;
  onBusyChange(updateMode);
  updateMode();
  for (const id of ['tw', 'td', 'th'])
    $('#' + id).oninput = () => {
      if (!state.busy && $('#mode').value === 'table') {
        state.name = 'table';
        replaceObject(tableModel(), 'Editable table', 'Local table template · not AI');
      }
    };
  $('#cancelai').onclick = cancelCurrent;
  $('#importbtn').onclick = () => $('#importfile').click();
  $('#importfile').onchange = importGLB;
  $('#generate').onclick = generate;
}
