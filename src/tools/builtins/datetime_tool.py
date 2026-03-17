from __future__ import annotations

from datetime import datetime, timezone
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


def get_datetime(timezone: str = "UTC") -> dict:
    """Return the current date and time in the requested timezone."""
    try:
        tz = ZoneInfo(timezone)
    except ZoneInfoNotFoundError:
        tz = ZoneInfo("UTC")
        timezone = "UTC"

    now = datetime.now(tz)
    return {
        "datetime": now.isoformat(),
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S"),
        "timezone": timezone,
        "day_of_week": now.strftime("%A"),
    }
