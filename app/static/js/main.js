// Studio entry point: wires the modules together.
import { initExport } from './export.js';
import { initModes, updateMode } from './modes.js';
import { initUpload } from './upload.js';
import { initViewerTools } from './viewer.js';

initViewerTools();
initUpload({ onImageReady: updateMode });
initModes();
initExport();
