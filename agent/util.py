from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from . import config


def parse_ts(value):
    """ISO-8601 string (or epoch number) to epoch seconds."""
    if isinstance(value, (int, float)):
        return float(value)
    dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def iso(epoch):
    return datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat()


def local_dt(epoch):
    return datetime.fromtimestamp(epoch, tz=ZoneInfo(config.LOCAL_TZ))


def is_night(epoch):
    h = local_dt(epoch).hour
    return h >= config.NIGHT_START_HOUR or h < config.NIGHT_END_HOUR


def hhmm(epoch):
    return local_dt(epoch).strftime("%H:%M")
