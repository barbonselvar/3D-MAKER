// Runs the Khronos glTF-Validator on one file and prints its JSON report.
// Usage: node worker/validators/gltf_validate.mjs model.glb
import { readFile } from 'node:fs/promises';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const validator = require('gltf-validator');

const path = process.argv[2];
if (!path) {
  console.error('usage: gltf_validate.mjs <file.glb>');
  process.exit(2);
}
const data = new Uint8Array(await readFile(path));
const report = await validator.validateBytes(data, {
  maxIssues: 200,
  // GLBs from the pipeline are self-contained; any external reference is an error.
  externalResourceFunction: (uri) => Promise.reject(new Error(`external resource not allowed: ${uri}`)),
});
process.stdout.write(JSON.stringify(report));
