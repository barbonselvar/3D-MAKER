// Mutable studio state shared between modules.
export const state = {
  busy: false, // a generation job is running
  img: null, // HTMLImageElement of the selected source image
  file: null, // original File when picked from disk (uploaded as-is); null for samples
  name: 'model', // base name for downloads
  sourceName: 'model', // name of the selected source image
  generated: false, // a model (not the placeholder) is in the viewer
  sourceBytes: null, // original GLB bytes for imported/AI models; null for local modes
};
