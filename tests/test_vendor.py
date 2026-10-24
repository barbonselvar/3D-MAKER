"""Supply-chain checks for the browser bundle (P0-03)."""
import hashlib
import re
from pathlib import Path

STATIC = Path(__file__).resolve().parents[1] / 'app' / 'static'


def test_vendored_files_match_checksums():
    lines = (STATIC / 'vendor' / 'SHA256SUMS').read_text(encoding='utf-8').split('\n')
    entries = [line.split(None, 1) for line in lines if line.strip()]
    assert entries
    for digest, rel in entries:
        assert hashlib.sha256((STATIC / rel).read_bytes()).hexdigest() == digest, rel


def browser_sources():
    files = list(STATIC.glob('*.html')) + list((STATIC / 'js').glob('**/*.js'))
    return {f: f.read_text(encoding='utf-8') for f in files}


def test_no_runtime_cdn_code():
    # Google Fonts stays (stylesheet only); executable code must be local.
    cdn = re.compile(r'https?://(esm\.sh|cdn\.jsdelivr\.net|unpkg\.com|cdn\.skypack\.dev|ga\.jspm\.io)', re.I)
    for path, text in browser_sources().items():
        assert not cdn.search(text), f'CDN script URL in {path.name}'


def test_every_versioned_import_is_vendored():
    for path, text in browser_sources().items():
        for ref in re.findall(r'\./vendor/[^\'"\s)]+', text):
            if ref.endswith('/'):
                continue
            assert (STATIC / ref[2:]).is_file(), f'{path.name} imports missing {ref}'
