"""Single source of time so tests can use a fake clock."""
import time
from datetime import datetime, timezone


def now() -> float:
    return time.time()


def iso(ts: float | None) -> str | None:
    return None if ts is None else datetime.fromtimestamp(ts, timezone.utc).isoformat().replace('+00:00', 'Z')
