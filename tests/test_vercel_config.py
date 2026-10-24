"""Guards for the public Vercel deployment: static studio only, no backend, no worker, no secrets."""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_vercel_json_is_static_only():
    cfg = json.loads((ROOT / 'vercel.json').read_text(encoding='utf-8'))
    assert cfg['framework'] is None  # never auto-detect FastAPI (app/main.py) as a serverless backend
    assert cfg['outputDirectory'] == 'app/static'
    assert cfg['buildCommand'] == '' and cfg['installCommand'] == ''
    assert not {'functions', 'rewrites', 'builds', 'routes', 'crons'} & set(cfg)  # nothing dynamic, nothing proxied
    headers = {h['key']: h['value'] for rule in cfg['headers'] for h in rule['headers']}
    assert headers['X-Content-Type-Options'] == 'nosniff' and headers['X-Frame-Options'] == 'DENY'


def _uploaded(path: str, rules: list[str]) -> bool:
    """Minimal gitignore evaluation for the anchored allow-list used in .vercelignore."""
    keep = True
    for rule in rules:
        negate, pattern = rule.startswith('!'), rule.lstrip('!')
        regex = '^' + re.escape(pattern.lstrip('/')).replace(r'\*', '[^/]*') + '(/.*)?$'
        if re.match(regex, path):
            keep = negate
    return keep


def test_vercelignore_uploads_only_the_static_studio():
    rules = [r.strip() for r in (ROOT / '.vercelignore').read_text(encoding='utf-8').splitlines()
             if r.strip() and not r.startswith('#')]
    assert rules[0] == '/*'
    for path in ('app/static/index.html', 'app/static/js/main.js', 'app/static/vendor/SHA256SUMS', 'vercel.json'):
        assert _uploaded(path, rules), path
    for path in ('app/main.py', 'app/internal.py', 'app/core/config.py', 'worker/main.py', 'worker/Dockerfile',
                 'scripts/windows/setup.ps1', 'tests/conftest.py', 'requirements.txt', '.env', 'docs/WINDOWS.md'):
        assert not _uploaded(path, rules), path
