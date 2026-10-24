// Shared "a job is running" state: locks inputs and routes the Cancel button to the running job.
import { $ } from './dom.js';
import { state } from './state.js';

let cancelHandler = null;
const listeners = [];

export function onBusyChange(fn) {
  listeners.push(fn);
}

export function setBusy(on, cancel = null) {
  state.busy = on;
  cancelHandler = on ? cancel : null;
  for (const id of ['#mode', '#generate', '#file', '#importbtn']) $(id).disabled = on;
  $('#cancelai').hidden = !(on && cancel);
  listeners.forEach((fn) => fn());
}

export function cancelCurrent() {
  cancelHandler?.();
}
