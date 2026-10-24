// Tiny DOM helpers shared by every studio module.
export const $ = (s) => document.querySelector(s);
export const statusEl = $('#status');

export function setStatus(text) {
  statusEl.textContent = text;
}

export function toast(text) {
  const el = $('#toast');
  el.textContent = text;
  el.hidden = false;
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => (el.hidden = true), 3200);
}
