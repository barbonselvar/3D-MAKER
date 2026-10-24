// Source image selection: file picker, drag and drop, and the two built-in samples.
import { $, setStatus } from './dom.js';
import { state } from './state.js';

let imgURL;
let inputVersion = 0;
let onReady = () => {};

function selectImage(source, label, file = null) {
  const version = ++inputVersion;
  const next = new Image();
  next.onload = () => {
    if (version !== inputVersion) return;
    if (next.width * next.height > 40000000) {
      setStatus('Image dimensions are too large. Resize below 40 megapixels.');
      return;
    }
    state.img = next;
    state.file = file;
    state.name = label.replace(/\.[^.]+$/, '');
    state.sourceName = state.name;
    $('#thumb').src = source;
    $('#thumb').classList.remove('hidden');
    $('#uploadcopy').classList.add('hidden');
    onReady();
    setStatus('Ready: ' + label);
  };
  next.onerror = () => setStatus('Could not read this image. Try PNG, JPG, or WEBP.');
  next.src = source;
}

function accept(file) {
  if (state.busy) return;
  if (!file) return;
  if (!['image/png', 'image/jpeg', 'image/webp'].includes(file.type)) {
    setStatus('Please choose PNG, JPG, or WEBP.');
    return;
  }
  if (file.size > 12 * 1024 * 1024) {
    setStatus('Choose an image smaller than 12 MB.');
    return;
  }
  if (imgURL) URL.revokeObjectURL(imgURL);
  imgURL = URL.createObjectURL(file);
  selectImage(imgURL, file.name, file);
}

function drawSample(kind) {
  const c = document.createElement('canvas');
  c.width = c.height = 256;
  const x = c.getContext('2d');
  x.fillStyle = '#d5fa6b';
  x.beginPath();
  if (kind === 'vase') {
    x.moveTo(100, 35);
    x.lineTo(156, 35);
    x.bezierCurveTo(143, 85, 220, 120, 177, 218);
    x.lineTo(79, 218);
    x.bezierCurveTo(36, 120, 113, 85, 100, 35);
  } else {
    [[142, 20], [65, 140], [117, 140], [97, 236], [192, 108], [140, 108]].forEach(([a, b], i) =>
      i ? x.lineTo(a, b) : x.moveTo(a, b),
    );
  }
  x.closePath();
  x.fill();
  return c.toDataURL();
}

export function initUpload({ onImageReady }) {
  onReady = onImageReady;
  $('#file').onchange = (e) => accept(e.target.files[0]);
  $('#drop').onclick = () => $('#file').click();
  $('#drop').onkeydown = (e) => {
    if (e.key === 'Enter' || e.key === ' ') {
      e.preventDefault();
      $('#file').click();
    }
  };
  $('#drop').ondragover = (e) => {
    e.preventDefault();
    $('#drop').classList.add('drag');
  };
  $('#drop').ondragleave = () => $('#drop').classList.remove('drag');
  $('#drop').ondrop = (e) => {
    e.preventDefault();
    $('#drop').classList.remove('drag');
    accept(e.dataTransfer.files[0]);
  };
  document.querySelectorAll('[data-sample]').forEach(
    (b) =>
      (b.onclick = () => {
        if (state.busy) return;
        selectImage(drawSample(b.dataset.sample), b.dataset.sample + '.png');
      }),
  );
}
