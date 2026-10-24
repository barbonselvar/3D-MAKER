"""Syntax-check every browser module with `node --check`.

Inline <script type="module"> blocks in app/static/*.html are extracted to temp
files first; standalone app/static/js/*.js modules are checked directly.
"""
import re
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATIC = ROOT / 'app' / 'static'


def main() -> int:
    targets: list[Path] = sorted((STATIC / 'js').glob('**/*.js')) if (STATIC / 'js').is_dir() else []
    tmp = Path(tempfile.mkdtemp())
    for html in sorted(STATIC.glob('*.html')):
        for i, body in enumerate(re.findall(r'<script type="module">(.*?)</script>', html.read_text(), re.S)):
            out = tmp / f'{html.stem}-{i}.mjs'
            out.write_text(body)
            targets.append(out)
    failed = 0
    for path in targets:
        result = subprocess.run(['node', '--check', str(path)], capture_output=True, text=True)
        status = 'ok' if result.returncode == 0 else 'FAIL'
        print(f'{status}  {path.relative_to(ROOT) if path.is_relative_to(ROOT) else path.name}')
        if result.returncode:
            failed += 1
            print(result.stderr)
    if not targets:
        print('no browser modules found')
        return 1
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
