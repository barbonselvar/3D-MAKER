"""P0-06: /api/generate outputs expire after 24 h."""
import os

from app import main


def make_folder(name: str, mtime: float):
    folder = main.OUTPUT / name
    folder.mkdir(exist_ok=True)
    (folder / 'model.glb').write_bytes(b'x')
    os.utime(folder, (mtime, mtime))
    return folder


def test_cleanup_removes_only_expired_folders():
    now = 2_000_000_000.0  # fake clock
    old = make_folder('a' * 32, now - main.OUTPUT_TTL_SECONDS - 1)
    fresh = make_folder('b' * 32, now - main.OUTPUT_TTL_SECONDS + 60)
    assert main.cleanup_outputs(now=now) >= 1
    assert not old.exists()
    assert fresh.exists()


def test_cleanup_ignores_stray_files():
    stray = main.OUTPUT / 'note.txt'
    stray.write_text('keep')
    os.utime(stray, (0, 0))
    main.cleanup_outputs(now=2_000_000_000.0)
    assert stray.exists()


def test_blender_available_requires_real_binary(monkeypatch):
    monkeypatch.setenv('BLENDER_BIN', '/nonexistent/blender')
    assert main.blender_available() is False
    monkeypatch.setenv('BLENDER_BIN', 'sh')  # any executable on PATH
    assert main.blender_available() is True
    monkeypatch.delenv('BLENDER_BIN')
    assert main.blender_available() is False
