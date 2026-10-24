"""Sunrise / sunset (NOAA / Almanac algorithm). Accurate to a few minutes.

Validated for Indian longitudes. For locations where the event falls on a different UTC calendar
day than the local solar date, results can be off by a day, which does not occur in India.
"""
import math
from datetime import date, datetime, timedelta, timezone


def _event_utc_hours(day, lat, lon, rising, zenith=90.833):
    n = day.timetuple().tm_yday
    lng_hour = lon / 15.0
    t = n + ((6 - lng_hour) / 24.0 if rising else (18 - lng_hour) / 24.0)
    m = 0.9856 * t - 3.289
    rad = math.radians
    l = (m + 1.916 * math.sin(rad(m)) + 0.020 * math.sin(rad(2 * m)) + 282.634) % 360.0
    ra = math.degrees(math.atan(0.91764 * math.tan(rad(l)))) % 360.0
    ra += (math.floor(l / 90.0) * 90.0) - (math.floor(ra / 90.0) * 90.0)
    ra /= 15.0
    sin_dec = 0.39782 * math.sin(rad(l))
    cos_dec = math.cos(math.asin(sin_dec))
    cos_h = (math.cos(rad(zenith)) - sin_dec * math.sin(rad(lat))) / (cos_dec * math.cos(rad(lat)))
    if cos_h > 1 or cos_h < -1:
        return None  # sun never rises / never sets (polar)
    h = (360.0 - math.degrees(math.acos(cos_h))) if rising else math.degrees(math.acos(cos_h))
    h /= 15.0
    local_t = h + ra - 0.06571 * t - 6.622
    return (local_t - lng_hour) % 24.0


def sun_times(day, lat, lon):
    """-> (sunrise_utc, sunset_utc) for the local solar date `day`, or None for polar cases."""
    r = _event_utc_hours(day, lat, lon, True)
    s = _event_utc_hours(day, lat, lon, False)
    if r is None or s is None:
        return None
    base = datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
    return base + timedelta(hours=r), base + timedelta(hours=s)


def local_solar_date(t, lon):
    return (t.astimezone(timezone.utc) + timedelta(hours=lon / 15.0)).date()


def is_dark(t, lat, lon, margin_min=0.0):
    """True between sunset and next sunrise (shifted by margin_min). Unknown (polar): False."""
    t = t.astimezone(timezone.utc)
    st = sun_times(local_solar_date(t, lon), lat, lon)
    if st is None:
        return False
    rise, sset = st
    m = timedelta(minutes=margin_min)
    return t < rise - m or t > sset + m
