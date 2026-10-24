// Generate panel: choose Internal GPU or External GPU, follow the job, load the validated result.
import { api, ApiFailure } from './api.js';
import { setBusy } from './busy.js';
import { $, setStatus } from './dom.js';
import * as hfdemo from './hfdemo.js';
import { state } from './state.js';
import { disposeObject, loadGLB, replaceObject } from './viewer.js';

const TERMINAL = new Set(['completed', 'failed', 'cancelled', 'expired']);
const STAGES = [['segment', 'Segment'], ['infer', 'Infer'], ['extract', 'Extract'], ['bake', 'Bake'],
  ['cleanup', 'Clean'], ['validate', 'Validate'], ['export', 'Export']];
const STATUS_TEXT = { queued: 'Queued', assigned: 'Starting on the GPU', preparing: 'Preparing', running: 'Generating',
  postprocessing: 'Processing the mesh', validating: 'Validating', exporting: 'Saving' };
const PROVIDER_NAMES = { 'external:modal': 'Dedicated GPU provider', 'external:hf-demo': 'Hugging Face public demo' };
// When the page is served without the API (static hosting), the browser-run public demo is still offered.
const STATIC_HF_DEMO = { id: 'external:hf-demo', kind: 'external', available: true, price_estimate_usd: 0,
  notes: 'Shared Hugging Face Space: free quota, queues, no SLA; the image leaves your device.' };
const JOB_KEY = '3dmaker.activeJob';
const GUIDE_URL = 'https://github.com/barbonselvar/3D-MAKER/blob/main/docs/WINDOWS.md';

let targets = null;
let currentJob = null;

const gb = (mb) => (mb == null ? '?' : Math.round(mb / 1024) + ' GB');

function remember(jobId) {
  try {
    jobId ? localStorage.setItem(JOB_KEY, jobId) : localStorage.removeItem(JOB_KEY);
  } catch {
    // storage unavailable (private mode): resuming after reload is a convenience only
  }
}

function setCard(card, enabled) {
  card.setAttribute('aria-disabled', String(!enabled));
}

const QUALITY = { full: 'Full quality', preview: 'Fast, lower detail', untextured: 'Untextured' };
const ATTRIBUTION = { sf3d: 'Powered by Stability AI' }; // required by the Stability AI Community License

function engineFacts(e) {
  const facts = [];
  if (e.max_texture_size) facts.push(`textures up to ${e.max_texture_size} px`);
  if (e.peak_vram_mb) facts.push(`${(e.peak_vram_mb / 1024).toFixed(1)} GB peak VRAM`);
  if (e.seconds) facts.push(`~${Math.round(e.seconds)} s in the self-test`);
  return facts.join(' · ');
}

function renderTextureOptions() {
  const t = targets?.find((x) => x.id === 'internal');
  const engine = t?.engines.find((e) => e.id === selectedEngine());
  const select = $('#int-texture');
  select.replaceChildren();
  if (!engine) return;
  const max = engine.max_texture_size;
  const sizes = (max ? [512, 1024, 2048] : [1024, 2048, 4096]).filter((n) => !max || n <= max);
  for (const n of sizes) select.append(new Option(`${n} px${n === 512 ? ' · least memory' : ''}`, String(n)));
  select.value = String(max ? sizes[sizes.length - 1] : 2048);
}

function selectedEngine() {
  return document.querySelector('input[name=engine]:checked')?.value || null;
}

function renderInternal() {
  const status = $('#int-status');
  const button = $('#start-internal');
  const list = $('#int-engines');
  const t = targets?.find((x) => x.id === 'internal');
  status.classList.toggle('bad', !t?.available);
  button.disabled = !t?.available;
  setCard($('#card-internal'), !!t?.available);
  list.replaceChildren();
  $('#int-options').hidden = !t?.available;
  if (!targets) {
    // Public website (static hosting) or the local server is not running. A public HTTPS page cannot reach a
    // visitor's 127.0.0.1 worker (browsers block it), so point to the locally installed studio instead.
    status.textContent = 'Local GPU runs in 3D-MAKER installed on your own computer, not on this website. '
      + 'Install it with the Windows guide, then open http://127.0.0.1:8000 for Local GPU generation. ';
    const guide = Object.assign(document.createElement('a'), {
      href: GUIDE_URL, target: '_blank', rel: 'noopener', textContent: 'Windows setup guide ↗' });
    guide.className = 'guide-link';
    status.append(guide);
    return;
  }
  const d = t.device;
  status.textContent = d
    ? `${d.name || 'GPU'} · ${gb(d.vram_total_mb)}${d.vram_free_mb != null ? ' (' + gb(d.vram_free_mb) + ' free)' : ''}${d.driver ? ' · driver ' + d.driver : ''}`
    : t.reason;
  let first = null;
  for (const e of t.engine_status || []) {
    const li = document.createElement('li');
    li.className = 'engine';
    li.dataset.engine = e.id;
    const head = document.createElement('label');
    const radio = Object.assign(document.createElement('input'), { type: 'radio', name: 'engine', value: e.id, disabled: e.status !== 'ready' });
    radio.onchange = renderTextureOptions;
    const name = document.createElement('span');
    name.textContent = `${e.label} · ${QUALITY[e.quality] || e.quality}`;
    const chip = document.createElement('span');
    chip.className = `chip ${e.status}`;
    chip.textContent = e.status_label;
    head.append(radio, name, chip);
    const detail = document.createElement('small');
    detail.textContent = e.status === 'ready' ? engineFacts(e) : e.reason || '';
    li.append(head, detail);
    list.append(li);
    if (e.status === 'ready' && !first) first = radio;
  }
  if (first) first.checked = true;
  if (!t.available && t.device) status.textContent += '. ' + 'No engine is ready on this GPU yet (see below).';
  renderTextureOptions();
}

function selectedProvider() {
  return document.querySelector('input[name=provider]:checked')?.value || null;
}

function updateExternalButton() {
  const id = selectedProvider();
  const t = (targets || [STATIC_HF_DEMO]).find((x) => x.id === id);
  $('#consenttext').textContent = id ? `I agree to send this image to the ${PROVIDER_NAMES[id] || id}.` : 'Choose a provider first.';
  const button = $('#start-external');
  button.disabled = !(t?.available && $('#cloudconsent').checked);
  button.textContent = !t ? 'Start' : t.price_estimate_usd === 0 ? 'Start · free public demo' : 'Start';
}

function renderExternal() {
  const list = $('#ext-providers');
  list.replaceChildren();
  const external = targets ? targets.filter((t) => t.kind === 'external') : [STATIC_HF_DEMO];
  let first = null;
  for (const t of external) {
    const label = document.createElement('label');
    label.className = 'provider';
    label.setAttribute('aria-disabled', String(!t.available));
    const input = Object.assign(document.createElement('input'), { type: 'radio', name: 'provider', value: t.id, disabled: !t.available });
    input.onchange = updateExternalButton;
    const text = document.createElement('span');
    text.textContent = PROVIDER_NAMES[t.id] || t.id;
    const small = document.createElement('small');
    small.textContent = t.available ? t.notes || '' : t.reason || 'Not available.';
    text.append(small);
    label.append(input, text);
    list.append(label);
    if (t.available && !first) first = input;
  }
  if (first) first.checked = true;
  setCard($('#card-external'), external.some((t) => t.available));
  $('#cloudconsent').checked = false;
  updateExternalButton();
}

export async function openPanel() {
  const dialog = $('#genpanel');
  $('#int-status').textContent = 'Checking the local GPU…';
  $('#start-internal').disabled = true;
  if (!dialog.open) dialog.showModal();
  try {
    targets = (await api('/api/v1/gpu-targets')).targets;
  } catch {
    targets = null;
  }
  renderInternal();
  renderExternal();
}

function showStages(job) {
  const list = $('#jobstages');
  if (!job) {
    list.hidden = true;
    return;
  }
  const at = STAGES.findIndex(([key]) => key === job.stage);
  list.replaceChildren(
    ...STAGES.map(([key, label], i) => {
      const li = document.createElement('li');
      li.textContent = label;
      li.dataset.stage = key;
      if (job.status === 'completed' || (at >= 0 && i < at)) li.className = 'done';
      else if (i === at && !TERMINAL.has(job.status)) li.className = 'active';
      return li;
    }),
  );
  list.hidden = false;
}

function describe(job, startedAt) {
  const seconds = Math.round((Date.now() - startedAt) / 1000);
  const where = job.status === 'queued' && job.queue_position ? ` (position ${job.queue_position})` : '';
  return `${STATUS_TEXT[job.status] || job.status}${where} · ${seconds}s`;
}

/* Resolves with the final job object. Uses Server-Sent Events, falling back to polling every 3 s. */
function follow(jobId, first) {
  return new Promise((resolve) => {
    const startedAt = Date.now();
    let job = first;
    let source = null;
    let poll = null;
    const show = () => {
      setStatus(describe(job, startedAt));
      showStages(job);
    };
    const tick = setInterval(show, 1000);
    const finish = (final) => {
      source?.close();
      clearInterval(poll);
      clearInterval(tick);
      showStages(final);
      resolve(final);
    };
    const startPolling = () => {
      if (poll) return;
      poll = setInterval(async () => {
        try {
          job = await api(`/api/v1/jobs/${jobId}`);
          show();
          if (TERMINAL.has(job.status)) finish(job);
        } catch {
          // transient error: keep polling
        }
      }, 3000);
    };
    show();
    if (!('EventSource' in window)) return startPolling();
    source = new EventSource(`/api/v1/jobs/${jobId}/events`);
    source.addEventListener('stage', (ev) => {
      const e = JSON.parse(ev.data);
      job = { ...job, status: e.status, stage: e.stage ?? job.stage, progress: e.progress ?? job.progress,
        queue_position: e.status === 'queued' ? job.queue_position : null };
      show();
    });
    source.addEventListener('done', (ev) => finish(JSON.parse(ev.data)));
    source.addEventListener('error', (ev) => {
      if (ev.data) return finish(JSON.parse(ev.data)); // terminal job event from the server
      source.close(); // network problem: fall back to polling
      startPolling();
    });
  });
}

async function sourceBlob() {
  if (state.file) return state.file;
  const c = document.createElement('canvas');
  c.width = state.img.width;
  c.height = state.img.height;
  c.getContext('2d').drawImage(state.img, 0, 0);
  return new Promise((r) => c.toBlob(r, 'image/png'));
}

async function loadResult(job) {
  setStatus('Loading your model…');
  const model = await api(`/api/v1/models/${job.model_id}`);
  const glb = model.files.find((f) => f.format === 'glb' && f.validated);
  if (!glb) throw new ApiFailure('FORMAT_NOT_AVAILABLE', 'The model has no validated GLB.', 404);
  const response = await fetch(glb.url);
  if (!response.ok) throw new ApiFailure('DOWNLOAD_FAILED', 'Could not download the model.', response.status);
  const bytes = await response.arrayBuffer();
  const object = await loadGLB(bytes.slice(0));
  state.name = state.sourceName;
  const credit = ATTRIBUTION[job.engine] ? ` · ${ATTRIBUTION[job.engine]}` : '';
  replaceObject(object, state.name, `${job.engine} · Local GPU · ${job.params.texture_size} px · seed ${job.params.seed}${credit}`, bytes);
  setStatus('Model ready. Generated from one image: the back and hidden parts are estimated.');
}

async function cancelInternal() {
  if (!currentJob) return;
  setStatus('Cancellation requested…');
  try {
    await api(`/api/v1/jobs/${currentJob}/cancel`, { method: 'POST' });
  } catch (e) {
    setStatus(e.message);
  }
}

async function track(jobId, first) {
  currentJob = jobId;
  remember(jobId);
  try {
    const job = await follow(jobId, first);
    if (job.status === 'completed') await loadResult(job);
    else if (job.status === 'cancelled') setStatus('Job cancelled. No model was generated.');
    else setStatus(`${job.error?.message || 'The job failed.'} No substitute model was generated.`);
  } finally {
    currentJob = null;
    remember(null);
  }
}

async function runInternal() {
  const engine = selectedEngine();
  const textureSize = +$('#int-texture').value;
  setBusy(true, cancelInternal);
  showStages(null);
  try {
    setStatus('Uploading image…');
    const form = new FormData();
    form.append('image', await sourceBlob(), state.file?.name || 'image.png');
    const upload = await api('/api/v1/uploads', { method: 'POST', body: form });
    setStatus('Creating job…');
    const created = await api('/api/v1/jobs', {
      method: 'POST',
      json: { upload_id: upload.upload_id, target: 'internal', engine,
        params: { resolution: +$('#quality').value, texture_size: textureSize } },
    });
    await track(created.job_id, created.job);
  } catch (e) {
    setStatus(e.message);
  } finally {
    setBusy(false);
  }
}

async function runHfDemo() {
  setBusy(true, hfdemo.cancel);
  showStages(null);
  try {
    state.name = state.sourceName;
    const bytes = await hfdemo.generate(state.img, $('#quality').value);
    const object = await loadGLB(bytes.slice(0));
    if (hfdemo.isCancelled()) {
      disposeObject(object);
      throw Error('Cancelled');
    }
    replaceObject(object, state.name, 'TRELLIS.2 AI · Hugging Face public demo', bytes);
    setStatus('AI model ready. GLB preserves textures; other formats may omit them.');
  } catch (e) {
    setStatus(
      hfdemo.isCancelled()
        ? 'AI job cancelled.'
        : 'AI unavailable: ' + (e.message || String(e)) +
            '. Try the official service link for sign-in or GPU quota information. No local substitute was generated.',
    );
  } finally {
    setBusy(false);
  }
}

export async function initGenerate() {
  $('#genclose').onclick = () => $('#genpanel').close();
  $('#cloudconsent').onchange = updateExternalButton;
  $('#start-internal').onclick = () => {
    $('#genpanel').close();
    runInternal();
  };
  $('#start-external').onclick = () => {
    const provider = selectedProvider();
    if (!$('#cloudconsent').checked || provider !== 'external:hf-demo') return;
    $('#genpanel').close();
    runHfDemo();
  };
  // Leaving the page never cancels a job: pick up where we left off.
  let saved = null;
  try {
    saved = localStorage.getItem(JOB_KEY);
  } catch {
    saved = null;
  }
  if (!saved) return;
  try {
    const job = await api(`/api/v1/jobs/${saved}`);
    if (TERMINAL.has(job.status)) return remember(null);
    setBusy(true, cancelInternal);
    await track(saved, job);
  } catch {
    remember(null);
  } finally {
    if (state.busy) setBusy(false);
  }
}
