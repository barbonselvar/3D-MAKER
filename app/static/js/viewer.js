// Three.js viewer: scene, camera, controls, model replacement and viewer tools.
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { $, setStatus } from './dom.js';
import { state } from './state.js';

const canvas = $('#view');
let renderer;
try {
  renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
} catch (e) {
  setStatus('WebGL is unavailable. Enable hardware acceleration to use the studio.');
  throw e;
}
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setClearColor(0, 0);
renderer.outputColorSpace = THREE.SRGBColorSpace;

const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(38, 1, 0.01, 100);
camera.position.set(3, 2.1, 4.5);
const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;
controls.autoRotate = true;
controls.autoRotateSpeed = 0.7;
controls.minDistance = 1.2;
controls.maxDistance = 12;
scene.add(new THREE.HemisphereLight(0xeaffdb, 0x25332d, 3));
const light = new THREE.DirectionalLight(0xffffff, 4);
light.position.set(2, 4, 5);
scene.add(light);
const rim = new THREE.DirectionalLight(0xd5fa6b, 2);
rim.position.set(-3, 1, -2);
scene.add(rim);
const grid = new THREE.GridHelper(12, 36, 0x455446, 0x2c382e);
grid.position.y = -1.35;
scene.add(grid);

// Default material for local (non-textured) meshes; also used by the placeholder knot.
export const mat = new THREE.MeshStandardMaterial({ color: 0xd5fa6b, metalness: 0.3, roughness: 0.32 });
let mesh = new THREE.Mesh(new THREE.TorusKnotGeometry(0.7, 0.24, 160, 24), mat);
scene.add(mesh);

function resize() {
  const r = canvas.parentElement.getBoundingClientRect();
  renderer.setSize(r.width, r.height, false);
  camera.aspect = r.width / r.height;
  camera.updateProjectionMatrix();
}
new ResizeObserver(resize).observe(canvas.parentElement);
resize();
renderer.setAnimationLoop(() => {
  controls.update();
  renderer.render(scene, camera);
});

const materialsOf = (o) => (Array.isArray(o.material) ? o.material : [o.material]);

export function disposeObject(obj) {
  obj.traverse((o) => {
    if (o.geometry) o.geometry.dispose();
    if (o.material) {
      for (const m of materialsOf(o))
        if (m !== mat) {
          for (const k of ['map', 'normalMap', 'roughnessMap', 'metalnessMap']) m[k]?.dispose();
          m.dispose();
        }
    }
  });
}

function hasTexture(obj) {
  let found = false;
  obj.traverse((o) => {
    if (o.material) for (const m of materialsOf(o)) if (Object.values(m).some((v) => v && v.isTexture)) found = true;
  });
  return found;
}

export async function loadGLB(buffer) {
  const gltf = await new GLTFLoader().parseAsync(buffer, '');
  return gltf.scene;
}

/* The model is kept untouched inside a pivot group; only the pivot gets the viewer fit (scale/centre),
   so exports keep the model's own units and origin. */
export function replaceObject(obj, label, method, bytes = null) {
  scene.remove(mesh);
  disposeObject(mesh);
  mesh = new THREE.Group();
  mesh.add(obj);
  scene.add(mesh);
  state.sourceBytes = bytes;
  const box = new THREE.Box3().setFromObject(mesh);
  const size = box.getSize(new THREE.Vector3());
  const max = Math.max(size.x, size.y, size.z);
  if (!Number.isFinite(max) || max <= 0) throw Error('Model has no visible geometry');
  mesh.scale.multiplyScalar(2.6 / max);
  box.setFromObject(mesh);
  mesh.position.sub(box.getCenter(new THREE.Vector3()));
  box.setFromObject(mesh);
  grid.position.y = box.min.y - 0.03;
  let faces = 0;
  mesh.traverse((o) => {
    if (o.isMesh) faces += (o.geometry.index?.count || o.geometry.attributes.position.count) / 3;
  });
  state.generated = true;
  const textured = hasTexture(mesh);
  $('#color').disabled = textured;
  $('#color').title = textured ? 'Colour is off for textured models (it would tint the texture)' : 'Change material color';
  $('#modelname').textContent = label;
  $('#stats').textContent = Math.round(faces).toLocaleString() + ' triangles · ' + method;
  document.querySelectorAll('[data-export]').forEach((b) => (b.disabled = false));
  $('#reset').click();
}

/* Runs fn(model) with the viewer-only fit removed, because exporters read world matrices. */
export async function withoutViewerFit(fn) {
  const fit = [mesh.position.clone(), mesh.scale.clone()];
  mesh.position.set(0, 0, 0);
  mesh.scale.set(1, 1, 1);
  mesh.updateMatrixWorld(true);
  try {
    return await fn(mesh.children[0]);
  } finally {
    mesh.position.copy(fit[0]);
    mesh.scale.copy(fit[1]);
    mesh.updateMatrixWorld(true);
  }
}

export function initViewerTools() {
  $('#reset').onclick = () => {
    camera.position.set(3, 2.1, 4.5);
    controls.target.set(0, 0, 0);
    controls.update();
  };
  $('#rotate').onclick = (e) => {
    controls.autoRotate = !controls.autoRotate;
    e.currentTarget.setAttribute('aria-pressed', String(controls.autoRotate));
  };
  // Apply viewer controls to every mesh, including imported AI scenes.
  $('#wire').onclick = (e) => {
    const enabled = e.currentTarget.getAttribute('aria-pressed') !== 'true';
    mesh.traverse((o) => {
      if (o.isMesh) for (const m of materialsOf(o)) m.wireframe = enabled;
    });
    mat.wireframe = enabled;
    e.currentTarget.setAttribute('aria-pressed', String(enabled));
  };
  let ci = 0;
  $('#color').onclick = () => {
    if ($('#color').disabled) return;
    const color = [0xd5fa6b, 0xc3adff, 0xffac7c, 0xffffff][++ci % 4];
    mesh.traverse((o) => {
      if (o.isMesh) for (const m of materialsOf(o)) if (!m.map) m.color?.setHex(color);
    });
  };
}
