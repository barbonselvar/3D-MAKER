"""Windows portability guards: Python's default text encoding there is cp1252, not UTF-8."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = [p for d in ('app', 'worker', 'scripts', 'tests') for p in (ROOT / d).rglob('*.py')]


def test_text_io_always_names_an_encoding():
    offenders = []
    for path in SOURCES:
        for n, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
            calls = re.findall(r'\.(?:read_text|write_text)\((?:[^()]|\([^()]*\))*\)', line)
            calls += [c for c in re.findall(r'subprocess\.run\(.*', line) if 'text=True' in c]
            if any('encoding=' not in c for c in calls):
                offenders.append(f'{path.relative_to(ROOT)}:{n}: {line.strip()}')
    assert not offenders, 'add encoding=... (Windows defaults to cp1252):\n' + '\n'.join(offenders)
