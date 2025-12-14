import os
import requests
from datetime import datetime, timedelta
def _get_proxies():
    """Build proxies dict from environment variables."""
    http = os.getenv("HTTP_PROXY") or os.getenv("http_proxy")
    https = os.getenv("HTTPS_PROXY") or os.getenv("https_proxy")
    if not http and not https:
        return None
    proxies = {}
    if http:
        proxies["http"] = http
    if https:
        proxies["https"] = https
    return proxies

def _get_kwargs():
    return {"timeout": 15, "proxies": _get_proxies()}



def fetch_holidays(country_code: str, year: int):
    """Query public holidays via Nager.Date API.
    country_code examples: CN, US, etc.
    """
    try:
        url = f"https://date.nager.at/api/v3/PublicHolidays/{year}/{country_code}"
        r = requests.get(url, **_get_kwargs())
        r.raise_for_status()
        data = r.json()
        # return set of ISO dates
        return {item.get("date") for item in data if item.get("date")}
    except Exception:
        return set()


def fetch_weather_daily(lat: float, lon: float, start_date: str, days: int = 7):
    """Query weather forecast via Open-Meteo (free, no key).
    Returns list of dicts with date and temperature/precipitation.
    """
    try:
        start = datetime.fromisoformat(start_date)
        end = (start + timedelta(days=days)).date().isoformat()
        url = (
            "https://api.open-meteo.com/v1/forecast?"
            f"latitude={lat}&longitude={lon}&daily=temperature_2m_max,temperature_2m_min,precipitation_sum&"
            f"start_date={start.date().isoformat()}&end_date={end}&timezone=auto"
        )
        r = requests.get(url, **_get_kwargs())
        r.raise_for_status()
        data = r.json()
        daily = []
        dates = data.get("daily", {}).get("time", [])
        tmax = data.get("daily", {}).get("temperature_2m_max", [])
        tmin = data.get("daily", {}).get("temperature_2m_min", [])
        prcp = data.get("daily", {}).get("precipitation_sum", [])
        for i in range(min(len(dates), days)):
            daily.append({
                "date": dates[i],
                "tmax": tmax[i] if i < len(tmax) else None,
                "tmin": tmin[i] if i < len(tmin) else None,
                "precip": prcp[i] if i < len(prcp) else None,
            })
        return daily
    except Exception:
        return []


def fetch_epidemic(country_code: str, start_date: str, days: int = 7):
    """Epidemic signal via Our World in Data latest snapshot (lightweight JSON).
    Returns list of dicts with date and new_cases_smoothed for future days (use last known as proxy).
    """
    try:
        url = "https://covid.ourworldindata.org/data/owid-covid-data-latest.json"
        r = requests.get(url, **_get_kwargs())
        r.raise_for_status()
        data = r.json()
        key = country_code.upper()
        if key not in data:
            return []
        latest = data[key]
        new_cases = latest.get("new_cases_smoothed", 0) or 0
        start = datetime.fromisoformat(start_date).date()
        out = []
        for i in range(days):
            d = (start + timedelta(days=i)).isoformat()
            out.append({"date": d, "new_cases": new_cases})
        return out
    except Exception:
        return []


def compute_event_multipliers(future_days, holidays_set, weather_daily, epidemic_daily):
    """Compute simple risk multipliers for demand variance based on events.
    - Holidays: increase demand and variance (e.g., +10%).
    - Heavy precipitation: may reduce same-day demand for out-of-home purchases.
    - Epidemic spikes: may increase online demand volatility.
    Returns dict per date with multiplier.
    """
    multipliers = {}
    weather_map = {w["date"]: w for w in weather_daily} if weather_daily else {}
    epi_map = {e["date"]: e for e in epidemic_daily} if epidemic_daily else {}
    for item in future_days:
        d = item["date"]
        m = 1.0
        if d in holidays_set:
            m += 0.10
        w = weather_map.get(d)
        if w and w.get("precip") is not None and w.get("precip") > 10:
            m += 0.05  # more variance due to weather
        e = epi_map.get(d)
        if e and e.get("new_cases", 0) > 1000:
            m += 0.10
        multipliers[d] = m
    return multipliers
