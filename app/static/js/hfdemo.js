// Public demo provider: Microsoft's shared TRELLIS.2 Space on Hugging Face (external, quota-limited, no SLA).
import { setStatus } from './dom.js';

let activeJob = null;
let aiClient = null;
let cancelled = false;

export const isCancelled = () => cancelled;

export function cancel() {
  cancelled = true;
  activeJob?.cancel();
  setStatus('Cancellation requested…');
}

async function runJob(endpoint, args, label) {
  if (cancelled) throw Error('Cancelled');
  setStatus(label);
  activeJob = aiClient.submit(endpoint, args);
  let data;
  for await (const msg of activeJob) {
    if (cancelled) {
      activeJob.cancel();
      throw Error('Cancelled');
    }
    if (msg.type === 'status') {
      if (msg.stage === 'error') throw Error(msg.message || 'AI service rejected the request');
      setStatus(label + (msg.position != null ? ' · Queue ' + msg.position : ''));
    }
    if (msg.type === 'data') data = msg.data;
  }
  if (!data) throw Error('AI service returned no result');
  return data;
}

/* Sends img to the Space and returns the generated GLB bytes. Caller must have the user's consent. */
export async function generate(img, resolution) {
  cancelled = false;
  try {
    setStatus('Connecting to Microsoft TRELLIS.2…');
    const { Client, handle_file } = await import('../vendor/gradio-client@2.7.1/index.min.js');
    if (cancelled) throw Error('Cancelled');
    aiClient = await Client.connect('microsoft/TRELLIS.2');
    const c = document.createElement('canvas');
    const ratio = Math.min(1, 1024 / Math.max(img.width, img.height));
    c.width = Math.round(img.width * ratio);
    c.height = Math.round(img.height * ratio);
    c.getContext('2d').drawImage(img, 0, 0, c.width, c.height);
    const blob = await new Promise((r) => c.toBlob(r, 'image/png'));
    await runJob('/start_session', [], 'Starting AI session');
    const prep = await runJob('/preprocess_image', [handle_file(blob)], 'Removing background');
    if (!prep[0]) throw Error('Background preprocessing returned no image');
    await runJob(
      '/image_to_3d',
      [prep[0], Math.floor(Math.random() * 2147483647), resolution, 7.5, 0.7, 12, 5, 7.5, 0.5, 12, 3, 1, 0, 12, 3],
      'AI reconstructing geometry and materials',
    );
    const out = await runJob('/extract_glb', [100000, 1024], 'Extracting textured GLB');
    const model = out.find((x) => x && typeof x === 'object' && x.url);
    if (!model?.url) throw Error('AI service returned no model URL');
    const url = new URL(model.url);
    if (url.protocol !== 'https:') throw Error('Invalid model URL');
    setStatus('Loading your AI model…');
    const response = await fetch(url);
    if (!response.ok) throw Error('Could not download the AI model');
    return await response.arrayBuffer();
  } finally {
    activeJob = null;
    try {
      aiClient?.close();
    } catch {
      // ignore close errors on an already-closed session
    }
    aiClient = null;
  }
}
