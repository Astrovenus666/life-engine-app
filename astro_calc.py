# astro_calc.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import swisseph as swe

# ----------------------------
# Constants
# ----------------------------
SIGNS = ["Aries","Taurus","Gemini","Cancer","Leo","Virgo","Libra","Scorpio","Sagittarius","Capricorn","Aquarius","Pisces"]
SIGN_GLYPH = {
    "Aries":"♈","Taurus":"♉","Gemini":"♊","Cancer":"♋","Leo":"♌","Virgo":"♍",
    "Libra":"♎","Scorpio":"♏","Sagittarius":"♐","Capricorn":"♑","Aquarius":"♒","Pisces":"♓"
}

NAK_NAMES = [
    "Ashwini","Bharani","Krittika","Rohini","Mrigashirsha","Ardra","Punarvasu","Pushya","Ashlesha",
    "Magha","Purva Phalguni","Uttara Phalguni","Hasta","Chitra","Swati","Vishakha","Anuradha","Jyeshtha",
    "Mula","Purva Ashadha","Uttara Ashadha","Shravana","Dhanishta","Shatabhisha","Purva Bhadrapada","Uttara Bhadrapada","Revati"
]

VIM_ORDER = ["Ketu","Venus","Sun","Moon","Mars","Rahu","Jupiter","Saturn","Mercury"]
VIM_YEARS = {"Ketu":7, "Venus":20, "Sun":6, "Moon":10, "Mars":7, "Rahu":18, "Jupiter":16, "Saturn":19, "Mercury":17}
CYCLE_YEARS = 120.0
NAK_SIZE = 13 + 20/60  # 13°20'
DAYS_PER_YEAR = 365.24219879

PLANETS = [
    ("Sun", swe.SUN),
    ("Moon", swe.MOON),
    ("Mars", swe.MARS),
    ("Mercury", swe.MERCURY),
    ("Jupiter", swe.JUPITER),
    ("Venus", swe.VENUS),
    ("Saturn", swe.SATURN),
    ("Uranus", swe.URANUS),
    ("Rahu", swe.MEAN_NODE),
]

# ----------------------------
# Data models
# ----------------------------
@dataclass
class PlanetPos:
    name: str
    lon: float
    sign: str
    deg_in_sign: float
    lon_speed: float
    retro: bool

@dataclass
class DashaPeriod:
    level: str   # "MD"/"AD"/"PD"/"SD"/"PR"
    lord: str
    start: datetime
    end: datetime

# ----------------------------
# Utility
# ----------------------------
def norm360(x: float) -> float:
    x %= 360.0
    return x if x >= 0 else x + 360.0

def sign_of(lon: float) -> tuple[str, float]:
    lon = norm360(lon)
    si = int(lon // 30)
    return SIGNS[si], lon - si * 30.0

def swe_julday_utc(dt_utc: datetime) -> float:
    return swe.julday(
        dt_utc.year, dt_utc.month, dt_utc.day,
        dt_utc.hour + dt_utc.minute/60 + dt_utc.second/3600
    )

def years_to_td(years: float) -> timedelta:
    return timedelta(days=years * DAYS_PER_YEAR)

def order_from_lord(lord: str) -> list[str]:
    i = VIM_ORDER.index(lord)
    return VIM_ORDER[i:] + VIM_ORDER[:i]

def set_sidereal_mode(mode: str = "KRISHNAMURTI") -> None:
    mode = (mode or "").upper().strip()
    if mode in ("KP", "KRISHNAMURTI", "KRISHNAMURTI_AYANAMSHA"):
        swe.set_sid_mode(swe.SIDM_KRISHNAMURTI)
    else:
        swe.set_sid_mode(swe.SIDM_LAHIRI)

def _calc_ut_xx(jd: float, pid: int, flags: int):
    res = swe.calc_ut(jd, pid, flags)
    if isinstance(res, tuple) and len(res) == 2 and hasattr(res[0], "__len__"):
        return res[0]
    return res

# ----------------------------
# Astronomy
# ----------------------------
def calc_sidereal_planets(dt_utc: datetime, sid_mode: str = "KRISHNAMURTI") -> list[PlanetPos]:
    set_sidereal_mode(sid_mode)
    jd = swe_julday_utc(dt_utc)
    # FLG_SPEED is REQUIRED for xx[3] (longitude speed) to be populated.
    # Without it, speed reads as 0 and retrograde detection never triggers.
    flags = swe.FLG_SWIEPH | swe.FLG_SIDEREAL | swe.FLG_SPEED

    out: list[PlanetPos] = []
    rahu_lon = None
    rahu_speed = None

    for name, pid in PLANETS:
        xx = _calc_ut_xx(jd, pid, flags)
        lon = norm360(float(xx[0]))
        lon_speed = float(xx[3])
        retro = lon_speed < 0.0

        if name == "Rahu":
            rahu_lon = lon
            rahu_speed = lon_speed

        s, d = sign_of(lon)
        out.append(PlanetPos(name=name, lon=lon, sign=s, deg_in_sign=d, lon_speed=lon_speed, retro=retro))

    # Ketu = Rahu + 180
    if rahu_lon is not None:
        ketu_lon = norm360(rahu_lon + 180.0)
        s, d = sign_of(ketu_lon)
        ks = float(rahu_speed) if rahu_speed is not None else 0.0
        out.append(PlanetPos(name="Ketu", lon=ketu_lon, sign=s, deg_in_sign=d, lon_speed=ks, retro=(ks < 0.0)))

    return out

def calc_houses(dt_utc: datetime, lat: float, lon_east: float, sid_mode: str = "KRISHNAMURTI") -> dict:
    """
    Placidus houses. Swiss gives tropical cusps; convert to sidereal by subtracting ayanamsa.
    Robust across different swe.houses_ex() return shapes.
    """
    set_sidereal_mode(sid_mode)
    jd = swe_julday_utc(dt_utc)

    cusps, ascmc = swe.houses_ex(jd, lat, lon_east, b'P', swe.FLG_SWIEPH)
    ayan = float(swe.get_ayanamsa_ut(jd))

    cusp_vals = list(cusps)

    if len(cusp_vals) >= 13:
        trop = [float(cusp_vals[i]) for i in range(1, 13)]
    elif len(cusp_vals) == 12:
        trop = [float(cusp_vals[i]) for i in range(0, 12)]
    else:
        trop = [float(c) for c in cusp_vals[:12]]
        while len(trop) < 12:
            trop.append(trop[-1] if trop else 0.0)

    cusps_sid = [0.0] + [norm360(c - ayan) for c in trop]  # index 1..12

    asc_sid = norm360(float(ascmc[0]) - ayan)
    mc_sid  = norm360(float(ascmc[1]) - ayan)

    asc_sign, asc_deg = sign_of(asc_sid)
    mc_sign, mc_deg = sign_of(mc_sid)

    cusp_info = {}
    for h in range(1, 13):
        lon = cusps_sid[h]
        s, d = sign_of(lon)
        cusp_info[h] = {"lon": lon, "sign": s, "deg": d}

    return {
        "ayanamsa": ayan,
        "cusps_sid": cusps_sid,
        "cusp_info": cusp_info,
        "asc_sid": asc_sid, "asc_sign": asc_sign, "asc_deg": asc_deg,
        "mc_sid": mc_sid, "mc_sign": mc_sign, "mc_deg": mc_deg,
    }

# ----------------------------
# Vimshottari
# ----------------------------
def nakshatra_from_moon(moon_lon_sid: float) -> dict:
    moon_lon_sid = norm360(moon_lon_sid)
    idx = int(moon_lon_sid / NAK_SIZE)
    idx = max(0, min(26, idx))
    frac = (moon_lon_sid - idx * NAK_SIZE) / NAK_SIZE
    name = NAK_NAMES[idx]
    lord = VIM_ORDER[idx % 9]
    balance_years = (1.0 - frac) * VIM_YEARS[lord]
    elapsed_years = VIM_YEARS[lord] - balance_years
    return {
        "idx": idx, "name": name, "lord": lord,
        "frac": float(frac),
        "balance_years": float(balance_years),
        "elapsed_years": float(elapsed_years),
    }

def build_md_periods_from_birth(birth_local: datetime, start_md: str, elapsed_in_md_years: float, max_years: float = 120.0) -> list[DashaPeriod]:
    md_years = float(VIM_YEARS[start_md])
    md_start = birth_local - years_to_td(elapsed_in_md_years)
    md_end = md_start + years_to_td(md_years)

    periods: list[DashaPeriod] = [DashaPeriod("MD", start_md, md_start, md_end)]

    i = (VIM_ORDER.index(start_md) + 1) % 9
    while (periods[-1].end - md_start).total_seconds() < (max_years * DAYS_PER_YEAR * 86400.0):
        lord = VIM_ORDER[i]
        dur = float(VIM_YEARS[lord])
        s = periods[-1].end
        e = s + years_to_td(dur)
        periods.append(DashaPeriod("MD", lord, s, e))
        i = (i + 1) % 9

    return periods

def build_subperiods(parent: DashaPeriod, level: str) -> list[DashaPeriod]:
    parent_years = (parent.end - parent.start).total_seconds() / 86400.0 / DAYS_PER_YEAR
    t = parent.start
    out: list[DashaPeriod] = []
    for sub_lord in order_from_lord(parent.lord):
        sub_years = parent_years * float(VIM_YEARS[sub_lord]) / CYCLE_YEARS
        s = t
        e = s + years_to_td(sub_years)
        out.append(DashaPeriod(level, sub_lord, s, e))
        t = e
    out[-1].end = parent.end
    return out

# ----------------------------
# Divisional charts (Vargas) — VERIFIED against RVA reference (D9, D10)
# ----------------------------
# General rule for a varga of N divisions:
#   span = 30 / N  degrees per part
#   part = floor(deg_in_sign / span)            -> 0..N-1
#   varga_deg = (deg_in_sign - part*span)/span * 30   (scaled into a full 30 sign)
#   varga_sign = (start_sign + part) mod 12
# where start_sign depends on the varga's classical rule.
_MOVABLE = {"Aries", "Cancer", "Libra", "Capricorn"}
_FIXED   = {"Taurus", "Leo", "Scorpio", "Aquarius"}
# (dual/common signs: everything else)

def _varga_start_d9(sign_idx: int, sign_name: str) -> int:
    # D9: movable -> from same sign; fixed -> 9th from it; dual -> 5th from it
    if sign_name in _MOVABLE:
        return sign_idx
    if sign_name in _FIXED:
        return (sign_idx + 8) % 12
    return (sign_idx + 4) % 12

def _varga_start_d10(sign_idx: int, sign_name: str) -> int:
    # D10: odd sign -> from same; even sign -> from 9th
    # (Aries is sign 1 = odd, index 0)
    return sign_idx if (sign_idx % 2 == 0) else (sign_idx + 8) % 12

# Registry: code -> (num_divisions, start_rule)
VARGA_RULES = {
    "D1":  (1,  lambda i, n: i),
    "D9":  (9,  _varga_start_d9),
    "D10": (10, _varga_start_d10),
}

def varga_position(lon: float, varga: str) -> tuple[str, float, float]:
    """Map an absolute sidereal longitude to (varga_sign, varga_deg_in_sign, varga_abs_lon)."""
    lon = norm360(lon)
    sign_idx = int(lon // 30)
    deg_in_sign = lon - sign_idx * 30.0
    n, start_rule = VARGA_RULES[varga]
    if n == 1:
        return SIGNS[sign_idx], deg_in_sign, lon
    span = 30.0 / n
    part = int(deg_in_sign // span)
    if part >= n:
        part = n - 1
    frac = (deg_in_sign - part * span) / span
    start = start_rule(sign_idx, SIGNS[sign_idx])
    v_idx = (start + part) % 12
    v_deg = frac * 30.0
    return SIGNS[v_idx], v_deg, v_idx * 30.0 + v_deg

def divisional_chart(chart: dict, varga: str) -> dict:
    """
    Return a new chart dict (planets + houses) remapped to the given varga.
    D1 returns the chart unchanged. Planets AND the ascendant are remapped, so
    house placement stays correct. View-only: not used by the graph engine.
    """
    if varga == "D1":
        return chart

    new_planets = []
    for p in chart["planets"]:
        vs, vd, vlon = varga_position(p.lon, varga)
        new_planets.append(PlanetPos(
            name=p.name, lon=vlon, sign=vs, deg_in_sign=vd,
            lon_speed=p.lon_speed, retro=p.retro,
        ))

    h = chart["houses"]
    asc_vs, asc_vd, asc_vlon = varga_position(h["asc_sid"], varga)
    mc_vs, mc_vd, mc_vlon = varga_position(h.get("mc_sid", 0.0), varga)

    # Whole-sign cusps from the varga ascendant (divisional charts use whole-sign houses)
    asc_idx = SIGNS.index(asc_vs)
    cusp_info = {}
    cusps_sid = [0.0]
    for house in range(1, 13):
        s_idx = (asc_idx + (house - 1)) % 12
        lon_c = s_idx * 30.0
        cusps_sid.append(lon_c)
        cusp_info[house] = {"lon": lon_c, "sign": SIGNS[s_idx], "deg": 0.0}

    new_houses = dict(h)  # copy, then override the varga-specific fields
    new_houses.update({
        "asc_sid": asc_vlon, "asc_sign": asc_vs, "asc_deg": asc_vd,
        "mc_sid": mc_vlon, "mc_sign": mc_vs, "mc_deg": mc_vd,
        "cusps_sid": cusps_sid, "cusp_info": cusp_info,
    })
    # Divisional charts are reference-only: drop any BCP overlay so it never
    # implies the activation logic runs on a varga.
    new_houses.pop("bcp_house", None)
    new_houses.pop("bcp_age", None)

    out = dict(chart)
    out["planets"] = new_planets
    out["houses"] = new_houses
    return out

# ----------------------------
# Public API
# ----------------------------
def compute_birth_chart(birth_local: datetime, lat: float, lon_east: float, sid_mode: str = "KRISHNAMURTI") -> dict:
    birth_utc = birth_local.astimezone(ZoneInfo("UTC"))
    planets = calc_sidereal_planets(birth_utc, sid_mode=sid_mode)
    houses = calc_houses(birth_utc, lat, lon_east, sid_mode=sid_mode)

    moon = next(p for p in planets if p.name == "Moon")
    nak = nakshatra_from_moon(moon.lon)

    md_periods = build_md_periods_from_birth(
        birth_local=birth_local,
        start_md=nak["lord"],
        elapsed_in_md_years=nak["elapsed_years"],
        max_years=120.0
    )

    return {
        "planets": planets,
        "houses": houses,
        "nakshatra": nak,
        "md_periods": md_periods,
        "birth_local": birth_local,
        "birth_utc": birth_utc,
        "sid_mode": sid_mode,
    }

def _target_date(birth_local: datetime, year: int, month: int | None,
                 day: int | None = None,
                 hh: int | None = None, mm: int | None = None, ss: int | None = None) -> datetime:
    """Build the local target date for a given year (and optional month/day/time),
    keeping the birth day-of-month and birth time-of-day unless explicitly overridden.
    Falls back safely if the day overflows a short month (e.g. day 31 in a 30-day month)."""
    import calendar
    m = int(month) if month else birth_local.month
    last_day = calendar.monthrange(int(year), m)[1]
    if day is not None:
        d = min(int(day), last_day)
    else:
        d = min(birth_local.day, last_day)
    H = int(hh) if hh is not None else birth_local.hour
    M = int(mm) if mm is not None else birth_local.minute
    S = int(ss) if ss is not None else birth_local.second
    return birth_local.replace(year=int(year), month=m, day=d, hour=H, minute=M, second=S, microsecond=0)

def compute_transit_chart(birth_local: datetime, lat: float, lon_east: float, transit_year: int, sid_mode: str = "KRISHNAMURTI", month: int | None = None, day: int | None = None, hh: int | None = None, mm: int | None = None, ss: int | None = None) -> dict:
    dt_local = _target_date(birth_local, transit_year, month, day, hh, mm, ss)
    dt_utc = dt_local.astimezone(ZoneInfo("UTC"))
    planets = calc_sidereal_planets(dt_utc, sid_mode=sid_mode)
    houses = calc_houses(dt_utc, lat, lon_east, sid_mode=sid_mode)
    return {"planets": planets, "houses": houses, "dt_local": dt_local, "dt_utc": dt_utc, "sid_mode": sid_mode}

def compute_progressed_chart(birth_local: datetime, lat: float, lon_east: float, target_year: int, sid_mode: str = "KRISHNAMURTI", month: int | None = None, day: int | None = None, hh: int | None = None, mm: int | None = None, ss: int | None = None) -> dict:
    target_local = _target_date(birth_local, target_year, month, day, hh, mm, ss)
    age_days = (target_local - birth_local).total_seconds() / 86400.0
    progressed_local = birth_local + timedelta(days=age_days / DAYS_PER_YEAR)
    progressed_utc = progressed_local.astimezone(ZoneInfo("UTC"))
    planets = calc_sidereal_planets(progressed_utc, sid_mode=sid_mode)
    houses = calc_houses(progressed_utc, lat, lon_east, sid_mode=sid_mode)
    return {"planets": planets, "houses": houses, "dt_local": progressed_local, "dt_utc": progressed_utc, "sid_mode": sid_mode}

def compute_solar_arc_chart(birth_local: datetime, lat: float, lon_east: float, target_year: int, sid_mode: str = "KRISHNAMURTI", month: int | None = None, day: int | None = None, hh: int | None = None, mm: int | None = None, ss: int | None = None) -> dict:
    """
    Solar Arc (simple):
      arc = progressed_sun_lon (secondary) - natal_sun_lon
      directed_lon = natal_lon + arc  (all forward)
    Retro is not used for directed positions -> retro=False.
    """
    natal = compute_birth_chart(birth_local, lat, lon_east, sid_mode=sid_mode)
    prog = compute_progressed_chart(birth_local, lat, lon_east, target_year, sid_mode=sid_mode, month=month, day=day, hh=hh, mm=mm, ss=ss)

    natal_sun = next(p for p in natal["planets"] if p.name == "Sun")
    prog_sun = next(p for p in prog["planets"] if p.name == "Sun")
    arc = norm360(prog_sun.lon - natal_sun.lon)

    directed_planets: list[PlanetPos] = []
    for p in natal["planets"]:
        lon2 = norm360(p.lon + arc)
        s2, d2 = sign_of(lon2)
        directed_planets.append(PlanetPos(
            name=p.name,
            lon=lon2,
            sign=s2,
            deg_in_sign=d2,
            lon_speed=0.0,
            retro=False
        ))

    # rotate cusps/angles similarly
    houses_nat = natal["houses"]
    cusps_sid = [0.0] + [norm360(houses_nat["cusps_sid"][h] + arc) for h in range(1, 13)]
    cusp_info = {}
    for h in range(1, 13):
        s, d = sign_of(cusps_sid[h])
        cusp_info[h] = {"lon": cusps_sid[h], "sign": s, "deg": d}

    asc_sid = norm360(houses_nat["asc_sid"] + arc)
    mc_sid = norm360(houses_nat["mc_sid"] + arc)
    asc_sign, asc_deg = sign_of(asc_sid)
    mc_sign, mc_deg = sign_of(mc_sid)

    houses_dir = {
        "ayanamsa": houses_nat["ayanamsa"],
        "cusps_sid": cusps_sid,
        "cusp_info": cusp_info,
        "asc_sid": asc_sid, "asc_sign": asc_sign, "asc_deg": asc_deg,
        "mc_sid": mc_sid, "mc_sign": mc_sign, "mc_deg": mc_deg,
    }

    dt_local = _target_date(birth_local, target_year, month, day, hh, mm, ss)
    return {"planets": directed_planets, "houses": houses_dir, "dt_local": dt_local, "dt_utc": dt_local.astimezone(ZoneInfo("UTC")), "sid_mode": sid_mode}