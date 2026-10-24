"""Run everything on this computer with one command (native Windows or Linux, no Docker/WSL):

    python -m app.local_stack            # studio + API at http://127.0.0.1:8000, GPU worker in-process

Starts the public studio/API on 127.0.0.1:8000, the worker API on 127.0.0.1:8001 (never on a
network interface) with a fresh random worker token for this run, and the GPU worker with the
requested engines (default: sf3d). Nothing leaves the machine and no paid service is used.
"""
from __future__ import annotations

import argparse
import logging
import os
import secrets
import threading
import time
import webbrowser

import uvicorn

from .core.config import Settings
from .core.context import Context
from .internal import create_internal_app
from .main import create_app

log = logging.getLogger('maker.local')


def start_server(app, host: str, port: int) -> uvicorn.Server:
    server = uvicorn.Server(uvicorn.Config(app, host=host, port=port, log_level='warning'))
    threading.Thread(target=server.run, daemon=True, name=f'uvicorn-{port}').start()
    deadline = time.monotonic() + 20
    while not server.started:
        if time.monotonic() > deadline or (server.should_exit and not server.started):
            raise RuntimeError(f'Could not start the server on {host}:{port}. Is the port already in use?')
        time.sleep(0.05)
    return server


def run(port: int = 8000, worker_port: int = 8001, engines: list[str] | None = None, worker: bool = True,
        open_browser: bool = False, settings: Settings | None = None, classes: dict | None = None,
        stop: threading.Event | None = None, caps: dict | None = None) -> None:
    settings = settings or Settings.from_env()
    token = secrets.token_urlsafe(32)  # valid for this run only; never written to disk
    settings.worker_tokens[token] = 'local-gpu'
    context = Context.create(settings)
    servers = [start_server(create_app(context=context), '127.0.0.1', port),
               start_server(create_internal_app(context=context), '127.0.0.1', worker_port)]
    url = f'http://127.0.0.1:{port}/'
    print(f'[3d-maker] Studio and API: {url}')
    print(f'[3d-maker] Worker API: http://127.0.0.1:{worker_port} (local only)')
    if open_browser:
        webbrowser.open(url)
    stop = stop or threading.Event()
    try:
        if worker:
            from worker.main import serve  # noqa: PLC0415 - GPU stack is optional for --no-worker

            serve(f'http://127.0.0.1:{worker_port}', token, engines or ['sf3d'], caps=caps, classes=classes,
                  stop=stop, poll_interval=1)
        else:
            stop.wait()
    except KeyboardInterrupt:
        print('[3d-maker] Stopping.')
    finally:
        for server in servers:
            server.should_exit = True


def main() -> None:
    # Settings written by scripts/windows/setup.ps1 (secret key, data dir, pinned weights); env vars win.
    from worker.windows.setup_helpers import load_env  # noqa: PLC0415

    load_env()
    ap = argparse.ArgumentParser(description='Run the 3D Maker studio, API and local GPU worker on this computer.')
    ap.add_argument('--port', type=int, default=int(os.environ.get('MAKER_PORT', 8000)))
    ap.add_argument('--worker-port', type=int, default=int(os.environ.get('MAKER_WORKER_PORT', 8001)))
    ap.add_argument('--engines', default=os.environ.get('WORKER_ENGINES', 'sf3d'))
    ap.add_argument('--no-worker', action='store_true', help='studio and API only (no local GPU)')
    ap.add_argument('--open', action='store_true', help='open the studio in the default browser')
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(name)s: %(message)s')
    run(args.port, args.worker_port, [e.strip() for e in args.engines.split(',') if e.strip()],
        worker=not args.no_worker, open_browser=args.open)


if __name__ == '__main__':
    main()
