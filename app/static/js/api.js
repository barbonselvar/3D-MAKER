// Fetch wrapper for the 3D Maker API: JSON in/out, structured errors surfaced as ApiFailure.
export class ApiFailure extends Error {
  constructor(code, message, status) {
    super(message);
    this.code = code;
    this.status = status;
  }
}

export async function api(path, { method = 'GET', json, body } = {}) {
  const init = { method, headers: {} };
  if (json !== undefined) {
    init.headers['content-type'] = 'application/json';
    init.body = JSON.stringify(json);
  } else if (body !== undefined) init.body = body;
  let response;
  try {
    response = await fetch(path, init);
  } catch {
    throw new ApiFailure('API_UNREACHABLE', 'The 3D Maker server is not reachable. Start it with: uvicorn app.main:app', 0);
  }
  const data = await response.json().catch(() => null);
  if (!response.ok) {
    const err = data?.error;
    throw new ApiFailure(err?.code || 'HTTP_' + response.status, err?.message || `Request failed (${response.status}).`, response.status);
  }
  return data;
}
