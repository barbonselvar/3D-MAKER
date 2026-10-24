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

function renderInternal() {
  const status = $('#int-status');
  const button = $('#start-internal');
  const t = targets?.find((x) => x.id === 'internal');
  status.classList.toggle('bad', !t?.available);
  button.disabled = !t?.available;
  setCard($('#card-internal'), !!t?.available);
  if (!targets) {
    status.textContent = 'The 3D Maker server is not reachable, so no internal GPU can be used. Start it with: uvicorn app.main:app';
  } else if (!t.available) {
    status.textContent = t.reason;
  } else {
    const d = t.device || {};
    const e = t.engines[0];
    const quality = { full: 'Full quality', preview: 'Preview quality', untextured: 'Untextured' }[e.quality] || e.quality;
    status.textContent = `${d.name || 'GPU'} · ${gb(d.vram_free_mb)} free${d.driver ? ' · driver ' + d.driver : ''}. Runs ${e.label} (${quality}).`;
  }
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
  $('#int-status').textContent = 'Checking for a GPU worker…';
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
  replaceObject(object, state.name, `${model.manifest?.engine?.id || job.engine} · Internal GPU · seed ${job.params.seed}`, bytes);
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
  const engine = targets.find((t) => t.id === 'internal').engines[0].id;
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
      json: { upload_id: upload.upload_id, target: 'internal', engine, params: { resolution: +$('#quality').value } },
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
