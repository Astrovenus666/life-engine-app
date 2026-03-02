from __future__ import annotations

import re
import calendar
from dataclasses import dataclass
from datetime import datetime, date, time, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

import pycountry
import geonamescache
from timezonefinder import TimezoneFinder

from geopy.geocoders import Nominatim

import plotly.graph_objects as go

import astro_calc
from astro_calc import (
    compute_birth_chart,
    compute_progressed_chart,
    compute_transit_chart,
    compute_solar_arc_chart,
    SIGNS,
    SIGN_GLYPH,
)

# =========================================================
# Streamlit config
# =========================================================
st.set_page_config(page_title="Life Path Graph", layout="wide")
st.title("Life Path Graph")

# =========================================================
# Helpers
# =========================================================
def norm_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def safe_tz(tzname: str) -> str:
    try:
        ZoneInfo(tzname)
        return tzname
    except Exception:
        return "UTC"

def parse_time_text(s: str) -> time | None:
    """
    Accepts:
      HH:MM
      HH:MM:SS
      HHMM
      HHMMSS
    """
    s = (s or "").strip()
    if not s:
        return None

    if ":" in s:
        parts = s.split(":")
        if len(parts) == 2:
            hh, mm = parts
            ss = "0"
        elif len(parts) == 3:
            hh, mm, ss = parts
        else:
            return None
        if not (hh.isdigit() and mm.isdigit() and ss.isdigit()):
            return None
        h, m, sec = int(hh), int(mm), int(ss)
    else:
        if not s.isdigit():
            return None
        if len(s) == 4:
            h, m, sec = int(s[:2]), int(s[2:4]), 0
        elif len(s) == 6:
            h, m, sec = int(s[:2]), int(s[2:4]), int(s[4:6])
        else:
            return None

    if not (0 <= h <= 23 and 0 <= m <= 59 and 0 <= sec <= 59):
        return None
    return time(h, m, sec)

def fmt_dmy(dt: datetime | date) -> str:
    if isinstance(dt, datetime):
        return dt.strftime("%d/%m/%Y")
    return dt.strftime("%d/%m/%Y")

def fmt_dmy_dash(dt: datetime | date) -> str:
    return dt.strftime("%d-%b-%Y") if isinstance(dt, (datetime, date)) else str(dt)

@dataclass
class Place:
    country_name: str
    country_code: str
    city_name: str
    admin1_code: str | None
    lat: float
    lon: float
    timezone: str

def label_place(p: Place) -> str:
    parts = [p.city_name]
    if p.admin1_code:
        parts.append(p.admin1_code)
    parts.append(p.country_name)
    return ", ".join(parts)

ROMAN = ["", "I","II","III","IV","V","VI","VII","VIII","IX","X","XI","XII"]

def planet_abbr(name: str) -> str:
    return {
        "Sun": "Su", "Moon": "Mo", "Mars": "Ma", "Mercury": "Me",
        "Jupiter": "Ju", "Venus": "Ve", "Saturn": "Sa", "Uranus": "Ur",
        "Rahu": "Ra", "Ketu": "Ke"
    }.get(name, name[:2])

def fmt_deg(deg: float) -> str:
    d = int(deg)
    m_float = (deg - d) * 60
    m = int(m_float)
    s = int(round((m_float - m) * 60))
    if s == 60:
        s = 0
        m += 1
    if m == 60:
        m = 0
        d += 1
    return f"{d:02d}:{m:02d}:{s:02d}"

def sign_index(sign: str) -> int:
    return SIGNS.index(sign)

def house_for_sign(asc_sign: str, sign: str) -> int:
    a = sign_index(asc_sign)
    s = sign_index(sign)
    return ((s - a) % 12) + 1

def sign_for_house(asc_sign: str, house_num: int) -> str:
    a = sign_index(asc_sign)
    return SIGNS[(a + (house_num - 1)) % 12]

SIGN_LORD = {
    "Aries": "Mars",
    "Taurus": "Venus",
    "Gemini": "Mercury",
    "Cancer": "Moon",
    "Leo": "Sun",
    "Virgo": "Mercury",
    "Libra": "Venus",
    "Scorpio": "Mars",
    "Sagittarius": "Jupiter",
    "Capricorn": "Saturn",
    "Aquarius": "Saturn",
    "Pisces": "Jupiter",
}

MOVABLE = {"Aries", "Cancer", "Libra", "Capricorn"}
FIXED = {"Taurus", "Leo", "Scorpio", "Aquarius"}

def badhaka_house_for_lagna(lagna_sign: str) -> int:
    if lagna_sign in MOVABLE:
        return 11
    if lagna_sign in FIXED:
        return 9
    return 7

def cusp_str(cusp_info: dict, house_num: int) -> str:
    ci = cusp_info[house_num]
    return f"{fmt_deg(ci['deg'])}{SIGN_GLYPH.get(ci['sign'], '')}"

def to_naive_ts(x):
    t = pd.to_datetime(x, errors="coerce")
    if pd.isna(t):
        return t
    if getattr(t, "tz", None) is not None:
        t = t.tz_localize(None)
    return t

# =========================================================
# South Indian grid
# =========================================================
SOUTH_GRID = [
    ["Pisces","Aries","Taurus","Gemini"],
    ["Aquarius","", "", "Cancer"],
    ["Capricorn","", "", "Leo"],
    ["Sagittarius","Scorpio","Libra","Virgo"],
]

# =========================================================
# Place data (cached)
# =========================================================

@st.cache_resource
def get_geocoder():
    # Nominatim (OpenStreetMap) fallback for towns/villages not in geonamescache
    return Nominatim(user_agent="life_engine_app", timeout=10)

@st.cache_resource
def load_place_data():
    gc = geonamescache.GeonamesCache()
    tf = TimezoneFinder(in_memory=True)

    countries = [(c.name, c.alpha_2) for c in pycountry.countries]
    countries = sorted(countries, key=lambda x: x[0].lower())
    country_name_to_code = {n: cc for n, cc in countries}

    cities = gc.get_cities()
    rows = []
    for _id, c in cities.items():
        rows.append({
            "geonameid": int(_id),
            "name": c.get("name", ""),
            "countrycode": c.get("countrycode", ""),
            "admin1code": c.get("admin1code", "") or "",
            "lat": float(c.get("latitude", 0.0)),
            "lon": float(c.get("longitude", 0.0)),
            "population": int(c.get("population", 0) or 0),
        })
    df = pd.DataFrame(rows)
    df["name_l"] = df["name"].astype(str).str.lower()
    df["countrycode"] = df["countrycode"].astype(str)
    return countries, country_name_to_code, df, tf

def find_city_matches_stable(df: pd.DataFrame, country_code: str, query: str, limit: int = 50) -> pd.DataFrame:
    """
    STABLE + SIMPLE:
      - Does NOT live-update on every keystroke (we call it only when user presses Search)
      - Uses forgiving "contains" match on cleaned query (but not token splitting)
    """
    sub = df[df["countrycode"] == country_code]
    q = norm_spaces(query).lower()
    q = re.sub(r"[^a-z0-9\s]", " ", q)
    q = norm_spaces(q)

    if not q:
        return sub.nlargest(limit, "population")

    m = sub[sub["name_l"].str.contains(re.escape(q), na=False)]
    if m.empty:
        # fallback: try first word only
        first = q.split(" ")[0]
        if first:
            m = sub[sub["name_l"].str.contains(re.escape(first), na=False)]
    return m.nlargest(limit, "population")

def resolve_place(country_name: str, country_code: str, row: pd.Series, tf: TimezoneFinder) -> Place:
    lat = float(row["lat"])
    lon = float(row["lon"])
    tz = tf.timezone_at(lat=lat, lng=lon) or "UTC"
    tz = safe_tz(tz)
    admin1 = str(row.get("admin1code", "") or "")
    return Place(country_name, country_code, str(row["name"]), admin1 if admin1 else None, lat, lon, tz)

# =========================================================
# BCP helpers
# =========================================================
def bcp_house_from_age(age: int) -> int:
    r = age % 12
    return 12 if r == 0 else r

def bcp_age_for_house(current_age: int, current_house: int, target_house: int) -> int:
    diff = (current_house - target_house) % 12
    return current_age - diff

# =========================================================
# Panchang helpers (compact)
# =========================================================
TITHI_NAMES = [
    "Pratipada","Dvitiya","Tritiya","Chaturthi","Panchami","Shashthi","Saptami","Ashtami","Navami","Dashami",
    "Ekadashi","Dvadashi","Trayodashi","Chaturdashi","Purnima",
    "Pratipada","Dvitiya","Tritiya","Chaturthi","Panchami","Shashthi","Saptami","Ashtami","Navami","Dashami",
    "Ekadashi","Dvadashi","Trayodashi","Chaturdashi","Amavasya"
]
YOGA_NAMES = [
    "Vishkambha","Priti","Ayushman","Saubhagya","Shobhana","Atiganda","Sukarman","Dhriti","Shoola","Ganda",
    "Vriddhi","Dhruva","Vyaghata","Harshana","Vajra","Siddhi","Vyatipata","Variyana","Parigha","Shiva",
    "Siddha","Sadhya","Shubha","Shukla","Brahma","Indra","Vaidhriti"
]
KARANA_SEQ = ["Bava","Balava","Kaulava","Taitila","Garaja","Vanija","Vishti"]
NAK_SIZE = (13 + 20/60)
PADA_SPAN = NAK_SIZE / 4.0

def compute_pada_from_moon_lon(moon_lon_sid: float) -> int:
    within = moon_lon_sid % NAK_SIZE
    pada = int(within // PADA_SPAN) + 1
    return min(4, max(1, pada))

def compute_tithi_yoga_karana(sun_lon: float, moon_lon: float):
    diff = (moon_lon - sun_lon) % 360.0
    tithi_index = int(diff // 12.0)
    tithi_no = tithi_index + 1
    paksha = "Shukla" if tithi_no <= 15 else "Krishna"
    tithi_name = TITHI_NAMES[tithi_index]

    yoga_index = int(((sun_lon + moon_lon) % 360.0) // NAK_SIZE)
    yoga_name = YOGA_NAMES[yoga_index]

    half_idx = int(diff // 6.0)
    if half_idx == 0:
        karana = "Kimstughna"
    elif half_idx == 57:
        karana = "Shakuni"
    elif half_idx == 58:
        karana = "Chatushpada"
    elif half_idx == 59:
        karana = "Naga"
    else:
        karana = KARANA_SEQ[(half_idx - 1) % 7]

    return {"paksha": paksha, "tithi": tithi_name, "yoga": yoga_name, "karana": karana}

def render_panchang_panel_compact(effective_mode: str, info: dict):
    if effective_mode == "Dark":
        bg = "rgba(255,255,255,0.06)"
        border = "rgba(255,255,255,0.28)"
        text = "#ffffff"
        label = "rgba(255,255,255,0.70)"
    else:
        bg = "rgba(255,255,255,0.92)"
        border = "rgba(0,0,0,0.18)"
        text = "#111111"
        label = "rgba(0,0,0,0.60)"

    html = f"""
    <div style="border:1px solid {border}; background:{bg}; border-radius:12px; padding:8px 10px; color:{text};
                font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto;">
      <div style="display:grid; grid-template-columns: repeat(3, minmax(0, 1fr));
                  gap:8px 12px; font-size:11px; line-height:1.1;">
        <div><div style="color:{label};font-weight:800;">Day</div><div style="font-weight:900;">{info['weekday']}</div></div>
        <div><div style="color:{label};font-weight:800;">Rashi</div><div style="font-weight:900;">{info['rashi']}</div></div>
        <div><div style="color:{label};font-weight:800;">Nak</div><div style="font-weight:900;">{info['nak_pada']}</div></div>

        <div><div style="color:{label};font-weight:800;">Tithi</div><div style="font-weight:900;">{info['tithi']}</div></div>
        <div><div style="color:{label};font-weight:800;">Yoga</div><div style="font-weight:900;">{info['yoga']}</div></div>
        <div><div style="color:{label};font-weight:800;">Karana</div><div style="font-weight:900;">{info['karana']}</div></div>
      </div>
    </div>
    """
    components.html(html, height=86, scrolling=False)

# =========================================================
# South Indian chart renderer
# =========================================================
def south_chart_html(chart_title: str, planets, houses, center_lines: list[str], effective_mode: str, size_mode: str) -> str:
    asc_sign = houses["asc_sign"]
    cusp_info = houses["cusp_info"]
    activation_house = houses.get("bcp_house")
    bcp_age = houses.get("bcp_age")

    sign_map = {s: [] for s in SIGNS}
    for p in planets:
        ab = planet_abbr(p.name)
        tag = f"[{ab}]" if getattr(p, "retro", False) else ab
        sign_map[p.sign].append((p.name, f"{tag} {fmt_deg(p.deg_in_sign)}"))

    if effective_mode == "Dark":
        cell_bg = "rgba(255,255,255,0.06)"
        blank_bg = "rgba(255,255,255,0.03)"
        border = "rgba(255,255,255,0.40)"
        border_blank = "rgba(255,255,255,0.22)"
        text_color = "#ffffff"
        house_color = "#7CFF7C"
        house_badge_bg = "rgba(0,0,0,0.25)"
        fixed_white = "#ffffff"
    else:
        cell_bg = "#f6e3e3"
        blank_bg = "#f0d7d7"
        border = "#7a1f1f"
        border_blank = "#9b4a4a"
        text_color = "#111111"
        house_color = "#1aa31a"
        house_badge_bg = "rgba(255,255,255,0.60)"
        fixed_white = "#000000"

    planet_colors = {
        "Mercury": "#1aa31a",
        "Mars":    "#d00000",
        "Sun":     "#ff7a00",
        "Jupiter": "#c9a400",
        "Uranus":  "#00838f",
        "Venus":   fixed_white,
        "Saturn":  fixed_white,
        "Moon":    fixed_white,
        "Rahu":    fixed_white,
        "Ketu":    fixed_white,
    }

    if size_mode == "full":
        td_h, pad = 150, 10
        items_fs, cusp_fs, hnum_fs = 15, 13, 16
        title_fs, center_fs = 18, 14
    else:
        td_h, pad = 145, 9
        items_fs, cusp_fs, hnum_fs = 14.5, 12.8, 15.5
        title_fs, center_fs = 17, 13.5

    def colored_span(pname: str, text: str) -> str:
        col = planet_colors.get(pname, text_color)
        return f"<span style='color:{col}; font-weight:800;'>{text}</span>"

    def cell_content(sign: str) -> str:
        h = house_for_sign(asc_sign, sign)
        lagna = " (L)" if h == 1 else ""
        cusp = cusp_str(cusp_info, h)

        activated = (activation_house == h)
        cls = "hnum activated" if activated else "hnum"

        age_tag = ""
        if isinstance(bcp_age, int) and isinstance(activation_house, int):
            age_here = bcp_age_for_house(bcp_age, activation_house, h)
            age_tag = f" <span class='age'>[{age_here}]</span>"

        header = f"""
        <div class="hrow">
          <span class="{cls}">{ROMAN[h]}{lagna}{age_tag}</span>
          <span class="cusp">{cusp}</span>
        </div>
        """
        items = sign_map[sign]
        body = "<br>".join(colored_span(pn, txt) for pn, txt in items) if items else "&nbsp;"
        return header + f"<div class='items'>{body}</div>"

    center_html = "<br>".join([f"<div class='cline'>{ln}</div>" for ln in center_lines])

    style = f"""
    <style>
      .title {{
        font-size: {title_fs}px; font-weight: 800; margin: 6px 0 10px 0; color: {text_color};
        font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto;
      }}
      table.south {{
        width: 100%;
        border-collapse: collapse;
        table-layout: fixed;
        color: {text_color};
        font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto;
      }}
      table.south td {{
        border: 2px solid {border};
        height: {td_h}px;
        vertical-align: top;
        padding: {pad}px;
        background: {cell_bg};
      }}
      table.south td.blank {{
        border: 2px solid {border_blank};
        background: {blank_bg};
      }}
      .hrow {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 8px;
        gap: 8px;
      }}
      .hnum {{
        font-size: {hnum_fs}px;
        font-weight: 900;
        color: {house_color};
        padding: 2px 8px;
        border-radius: 10px;
        background: {house_badge_bg};
        white-space: nowrap;
      }}
      .age {{
        font-size: 0.92em;
        font-weight: 900;
        opacity: 0.95;
        margin-left: 4px;
      }}
      .hnum.activated {{
        background: linear-gradient(45deg, #f7ff00, #00ff88);
        color: #000 !important;
        box-shadow: 0 0 10px rgba(0,255,136,0.85), 0 0 20px rgba(247,255,0,0.65);
      }}
      .cusp {{
        font-size: {cusp_fs}px;
        font-weight: 800;
        opacity: 0.95;
      }}
      .items {{
        font-size: {items_fs}px;
        line-height: 1.22;
      }}
      .centerbox {{
        height: 100%;
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        text-align: center;
        padding: 8px;
        gap: 6px;
        font-weight: 650;
      }}
      .cline {{
        font-size: {center_fs}px;
        font-weight: 650;
      }}
    </style>
    """

    html = [style, f"<div class='title'>{chart_title}</div>", "<table class='south'>"]
    for r in range(4):
        html.append("<tr>")
        for c in range(4):
            sign = SOUTH_GRID[r][c]
            if r == 1 and c == 1:
                html.append(f"<td class='blank' rowspan='2' colspan='2'><div class='centerbox'>{center_html}</div></td>")
                continue
            if (r == 1 and c == 2) or (r == 2 and c == 1) or (r == 2 and c == 2):
                continue
            if sign == "":
                html.append("<td class='blank'></td>")
            else:
                html.append(f"<td>{cell_content(sign)}</td>")
        html.append("</tr>")
    html.append("</table>")
    return "".join(html)

def render_south_chart(chart_title: str, planets, houses, center_lines: list[str], effective_mode: str, size_mode: str):
    html = south_chart_html(chart_title, planets, houses, center_lines, effective_mode, size_mode=size_mode)
    height = 760 if size_mode == "full" else 740
    components.html(html, height=height, scrolling=False)

# =========================================================
# Vimshottari selector (CLICK TO SELECT + Clear)
# =========================================================
def dasha_selector_click(md_periods):
    st.subheader("Vimshottari (click to select)")

    for k in ("sel_md", "sel_ad", "sel_pd", "selected_period"):
        if k not in st.session_state:
            st.session_state[k] = None

    def clear_all():
        st.session_state["sel_md"] = None
        st.session_state["sel_ad"] = None
        st.session_state["sel_pd"] = None
        st.session_state["selected_period"] = None

    if st.button("Clear selection", use_container_width=True):
        clear_all()

    def label(p):
        return f"{p.lord} {p.level} — {fmt_dmy_dash(p.start)} to {fmt_dmy_dash(p.end)}"

    md_labels = [label(p) for p in md_periods]
    md_idx = st.selectbox("MD (pick, then click Select MD)", range(len(md_periods)), format_func=lambda i: md_labels[i])

    if st.button("Select MD", use_container_width=True):
        st.session_state["sel_md"] = md_periods[md_idx]
        st.session_state["sel_ad"] = None
        st.session_state["sel_pd"] = None
        st.session_state["selected_period"] = {
            "level": "MD",
            "lord": st.session_state["sel_md"].lord,
            "start": st.session_state["sel_md"].start.isoformat(),
            "end": st.session_state["sel_md"].end.isoformat(),
            "label": label(st.session_state["sel_md"]),
        }

    md = st.session_state["sel_md"]
    if md is None:
        st.info("No MD selected yet.")
        return

    ads = astro_calc.build_subperiods(md, "AD")
    ad_labels = [label(p) for p in ads]
    ad_idx = st.selectbox("AD (pick, then click Select AD)", range(len(ads)), format_func=lambda i: ad_labels[i])

    if st.button("Select AD", use_container_width=True):
        st.session_state["sel_ad"] = ads[ad_idx]
        st.session_state["sel_pd"] = None
        st.session_state["selected_period"] = {
            "level": "AD",
            "lord": st.session_state["sel_ad"].lord,
            "start": st.session_state["sel_ad"].start.isoformat(),
            "end": st.session_state["sel_ad"].end.isoformat(),
            "label": label(st.session_state["sel_ad"]),
        }

    ad = st.session_state["sel_ad"]
    if ad is None:
        st.caption("Select AD to enable PD.")
        return

    pds = astro_calc.build_subperiods(ad, "PD")
    pd_labels = [label(p) for p in pds]
    pd_idx = st.selectbox("PD (pick, then click Select PD)", range(len(pds)), format_func=lambda i: pd_labels[i])

    if st.button("Select PD", use_container_width=True):
        st.session_state["sel_pd"] = pds[pd_idx]
        st.session_state["selected_period"] = {
            "level": "PD",
            "lord": st.session_state["sel_pd"].lord,
            "start": st.session_state["sel_pd"].start.isoformat(),
            "end": st.session_state["sel_pd"].end.isoformat(),
            "label": label(st.session_state["sel_pd"]),
        }

    sel = st.session_state.get("selected_period")
    if sel:
        st.success(f"Selected: **{sel['label']}**")

# =========================================================
# Aspect helpers + engines
# =========================================================
ORB_DEG = 1.0
ASPECTS = [
    (0.0,   "CONJ"),
    (60.0,  "SEXT"),
    (90.0,  "SQUARE"),
    (120.0, "TRINE"),
    (180.0, "OPP"),
]

BENEFIC = {"Jupiter", "Venus", "Mercury", "Moon"}
MALEFIC = {"Saturn", "Sun", "Mars", "Uranus", "Rahu", "Ketu"}

def norm360(x: float) -> float:
    x %= 360.0
    return x if x >= 0 else x + 360.0

def ang_diff(a: float, b: float) -> float:
    d = abs(norm360(a) - norm360(b))
    return d if d <= 180.0 else 360.0 - d

def tri_weight(delta: float, orb: float = ORB_DEG) -> float:
    ad = abs(delta)
    if ad > orb:
        return 0.0
    return 1.0 - (ad / orb)

def classify_aspect(p: float, n: float) -> tuple[str, float] | None:
    sep = ang_diff(p, n)
    best_name = None
    best_delta = 999.0
    for deg, name in ASPECTS:
        d = sep - deg
        if abs(d) < abs(best_delta):
            best_delta = d
            best_name = name
    if best_name is None or abs(best_delta) > ORB_DEG:
        return None
    return best_name, best_delta

def add_months_local(dt: datetime, months: int) -> datetime:
    y = dt.year + (dt.month - 1 + months) // 12
    m = (dt.month - 1 + months) % 12 + 1
    last_day = calendar.monthrange(y, m)[1]
    d = min(dt.day, last_day)
    return dt.replace(year=y, month=m, day=d)

def find_current_md_ad(md_periods: list[astro_calc.DashaPeriod], dt_local: datetime):
    md_hit = None
    for md in md_periods:
        if md.start <= dt_local < md.end:
            md_hit = md
            break
    if md_hit is None:
        return None, None, None, None
    ads = astro_calc.build_subperiods(md_hit, "AD")
    ad_hit = None
    for ad in ads:
        if ad.start <= dt_local < ad.end:
            ad_hit = ad
            break
    return md_hit.lord, (ad_hit.lord if ad_hit else None), md_hit, ad_hit

def moon_waxing(sun_lon: float, moon_lon: float) -> bool:
    diff = (moon_lon - sun_lon) % 360.0
    return diff < 180.0

def house_lords_for_lagna(lagna_sign: str) -> dict[int, str]:
    lords = {}
    for h in range(1, 13):
        s = sign_for_house(lagna_sign, h)
        lords[h] = SIGN_LORD[s]
    return lords

def planet_house_wholesign(lagna_sign: str, planet_sign: str) -> int:
    return house_for_sign(lagna_sign, planet_sign)

def saade_sati_active(natal_moon_sign: str | None, saturn_sign: str | None) -> bool:
    if not natal_moon_sign or not saturn_sign:
        return False
    mi = sign_index(natal_moon_sign)
    prev_sign = SIGNS[(mi - 1) % 12]
    next_sign = SIGNS[(mi + 1) % 12]
    return saturn_sign in {prev_sign, natal_moon_sign, next_sign}

@st.cache_data(show_spinner=False)
def generate_general_life_df(
    birth_local: datetime,
    lat: float,
    lon: float,
    sid_mode_key: str,
    progression_type: str,
    max_years: int = 120,
    step_months: int = 3,
) -> pd.DataFrame:
    birth = compute_birth_chart(birth_local, lat, lon, sid_mode=sid_mode_key)

    natal_planets = birth["planets"]
    natal_lon = {p.name: float(p.lon) for p in natal_planets}
    md_periods = birth["md_periods"]

    natal_mc_sid = float(birth["houses"]["mc_sid"])

    rows = []
    total_months = max_years * 12

    for m in range(0, total_months + 1, step_months):
        dt_local = add_months_local(birth_local, m)
        age_years = m / 12.0

        if progression_type.startswith("Secondary"):
            prog_local = birth_local + timedelta(days=age_years)
            prog_utc = prog_local.astimezone(ZoneInfo("UTC"))
            prog_planets = astro_calc.calc_sidereal_planets(prog_utc, sid_mode=sid_mode_key)
            prog_houses = astro_calc.calc_houses(prog_utc, lat, lon, sid_mode=sid_mode_key)
            mc_sid = float(prog_houses["mc_sid"])
            prog_lon = {p.name: float(p.lon) for p in prog_planets}
        else:
            arc = float(age_years) * 1.0
            mc_sid = norm360(natal_mc_sid + arc)
            prog_lon = {nm: norm360(lon0 + arc) for nm, lon0 in natal_lon.items()}

        score = 0.0

        for tname, w_base in [("Sun", 1.40), ("Saturn", 1.70), ("Uranus", 1.50)]:
            if tname not in natal_lon:
                continue
            res = classify_aspect(mc_sid, natal_lon[tname])
            if not res:
                continue
            asp, delta = res
            w = tri_weight(delta) * w_base
            if asp in ("TRINE", "SEXT", "CONJ"):
                score += w
            elif asp in ("SQUARE", "OPP"):
                score -= w

        for p_name, p_lon in prog_lon.items():
            for n_name, n_lon in natal_lon.items():
                res = classify_aspect(float(p_lon), float(n_lon))
                if not res:
                    continue
                asp, delta = res
                w = tri_weight(delta)
                if asp in ("TRINE", "SEXT"):
                    score += 1.0 * w
                elif asp in ("SQUARE", "OPP"):
                    score -= 1.0 * w
                elif asp == "CONJ":
                    score += 0.25 * w

        md, ad, _, _ = find_current_md_ad(md_periods, dt_local)

        rows.append({
            "date_local": dt_local.replace(tzinfo=None),
            "age_years": float(age_years),
            "score_total": float(score),
            "md": md,
            "ad": ad,
        })

    return pd.DataFrame(rows).sort_values("date_local").reset_index(drop=True)

@st.cache_data(show_spinner=False)
def generate_career_df(
    birth_local: datetime,
    lat: float,
    lon: float,
    sid_mode_key: str,
    progression_type: str,
    max_years: int = 120,
    step_months: int = 3,
) -> pd.DataFrame:
    birth = compute_birth_chart(birth_local, lat, lon, sid_mode=sid_mode_key)

    natal_planets = birth["planets"]
    natal_lon = {p.name: float(p.lon) for p in natal_planets}
    natal_sign = {p.name: p.sign for p in natal_planets}
    md_periods = birth["md_periods"]

    lagna_sign = birth["houses"]["asc_sign"]
    lords = house_lords_for_lagna(lagna_sign)

    trine_lords = {lords[1], lords[5], lords[9]}
    dusthana_lords = {lords[6], lords[8], lords[12]}
    third_lord = lords[3]
    badhaka_house = badhaka_house_for_lagna(lagna_sign)
    badhaka_lord = lords[badhaka_house]

    natal_mc_sid = float(birth["houses"]["mc_sid"])

    natal_moon_sign = natal_sign.get("Moon")
    if not natal_moon_sign:
        moon_lon = natal_lon.get("Moon")
        if moon_lon is not None:
            natal_moon_sign = SIGNS[int((moon_lon % 360.0) // 30)]

    def is_problem_lord(pl: str) -> bool:
        return (pl in dusthana_lords) or (pl == badhaka_lord) or (pl == third_lord)

    rows = []
    total_months = max_years * 12

    for m in range(0, total_months + 1, step_months):
        dt_local = add_months_local(birth_local, m)
        age_years = m / 12.0

        if progression_type.startswith("Secondary"):
            prog_local = birth_local + timedelta(days=age_years)
            prog_utc = prog_local.astimezone(ZoneInfo("UTC"))
            prog_planets = astro_calc.calc_sidereal_planets(prog_utc, sid_mode=sid_mode_key)
            prog_houses = astro_calc.calc_houses(prog_utc, lat, lon, sid_mode=sid_mode_key)
            mc_sid = float(prog_houses["mc_sid"])
            prog_lon = {p.name: float(p.lon) for p in prog_planets}
            prog_sign = {p.name: p.sign for p in prog_planets}
            sun_p = prog_lon.get("Sun")
            moon_p = prog_lon.get("Moon")
        else:
            arc = float(age_years) * 1.0
            mc_sid = norm360(natal_mc_sid + arc)
            prog_lon = {nm: norm360(lon0 + arc) for nm, lon0 in natal_lon.items()}
            prog_sign = {nm: SIGNS[int(norm360(lv) // 30.0)] for nm, lv in prog_lon.items()}
            sun_p = prog_lon.get("Sun")
            moon_p = prog_lon.get("Moon")

        md, ad, md_obj, ad_obj = find_current_md_ad(md_periods, dt_local)

        tzname = st.session_state.get("tzname", "UTC")
        dt_utc = dt_local.replace(tzinfo=ZoneInfo(tzname)).astimezone(ZoneInfo("UTC"))
        trans_planets = astro_calc.calc_sidereal_planets(dt_utc, sid_mode=sid_mode_key)
        trans_sat = next((p for p in trans_planets if p.name == "Saturn"), None)
        trans_sat_sign = trans_sat.sign if trans_sat else None

        score = 0.0

        if md:
            if md in trine_lords:
                score += 2.2
            if md in dusthana_lords:
                score -= 2.4
            if md == badhaka_lord:
                score -= 2.6
            if md == third_lord:
                score -= 1.8
            if md in {"Rahu", "Ketu", "Saturn"}:
                score -= 1.0

        if ad:
            if ad in trine_lords:
                score += 1.6
            if ad in dusthana_lords:
                score -= 1.8
            if ad == badhaka_lord:
                score -= 2.0
            if ad == third_lord:
                score -= 1.2
            if ad in {"Rahu", "Ketu", "Saturn"}:
                score -= 0.7

        if md_obj and (md_obj.end - dt_local).days <= 365:
            score -= 1.4 if md in {"Saturn", "Rahu", "Ketu"} else 0.6
        if ad_obj:
            ad_days_left = (ad_obj.end - dt_local).days
            if ad_days_left <= 92:
                score -= 1.0 if ad in {"Saturn", "Rahu", "Ketu"} else 0.5
            if md_obj and (md_obj.end - dt_local).days <= 365 and ad_days_left <= 92:
                score -= 0.9

        for h in [2, 5, 7, 10, 11, 1, 9]:
            lord = lords[h]
            lon_l = natal_lon.get(lord)
            if lon_l is None:
                continue
            res = classify_aspect(mc_sid, lon_l)
            if not res:
                continue
            asp, delta = res
            w = tri_weight(delta) * 1.6
            if asp in ("TRINE", "SEXT", "CONJ"):
                if not is_problem_lord(lord):
                    score += w
            elif asp in ("SQUARE", "OPP"):
                score -= 0.6 * w

        for lord in {lords[6], lords[8], lords[12], badhaka_lord, third_lord}:
            lon_l = natal_lon.get(lord)
            if lon_l is None:
                continue
            res = classify_aspect(mc_sid, lon_l)
            if not res:
                continue
            asp, delta = res
            w = tri_weight(delta) * 2.0
            if asp in ("SQUARE", "OPP", "CONJ"):
                score -= w

        waning = False
        if (sun_p is not None) and (moon_p is not None):
            try:
                waning = not moon_waxing(float(sun_p), float(moon_p))
            except Exception:
                waning = False

        waning_hits = 0

        for p_name, p_lon in prog_lon.items():
            for n_name, n_lon in natal_lon.items():
                res = classify_aspect(float(p_lon), float(n_lon))
                if not res:
                    continue
                asp, delta = res
                w = tri_weight(delta)
                if w <= 0:
                    continue

                p_prob = is_problem_lord(p_name)
                n_prob = is_problem_lord(n_name)

                if asp in ("TRINE", "SEXT"):
                    if (p_name in BENEFIC) and not p_prob:
                        score += 0.55 * w
                    elif p_name in MALEFIC:
                        score -= 0.25 * w
                    else:
                        score += 0.15 * w
                elif asp in ("SQUARE", "OPP", "CONJ"):
                    if p_name in MALEFIC:
                        score -= 0.75 * w
                    else:
                        score -= 0.30 * w
                    if p_prob or n_prob:
                        score -= 0.35 * w
                    if waning and (p_name == "Moon" or n_name == "Moon"):
                        waning_hits += 1

        if waning_hits > 0:
            score -= min(1.2, 0.35 * waning_hits)

        dust_cnt = 0
        for _, sg in prog_sign.items():
            h = planet_house_wholesign(lagna_sign, sg)
            if h in {6, 8, 12}:
                dust_cnt += 1
        if dust_cnt >= 3:
            score -= 1.8
            if (md in {"Saturn"} and ad in {"Saturn"}) or (md in {"Rahu"} and ad in {"Rahu"}) or (md in {"Ketu"} and ad in {"Ketu"}):
                score -= 0.9

        if saade_sati_active(natal_moon_sign, trans_sat_sign):
            score -= 1.2

        rows.append({
            "date_local": dt_local.replace(tzinfo=None),
            "age_years": float(age_years),
            "score_total": float(score),
            "md": md,
            "ad": ad,
        })

    return pd.DataFrame(rows).sort_values("date_local").reset_index(drop=True)

# =========================================================
# Baseline scaling helpers
# =========================================================
def auto_baseline_year(dates: pd.Series, values: pd.Series) -> int | None:
    d = pd.to_datetime(dates, errors="coerce")
    v = pd.to_numeric(values, errors="coerce")
    ok = d.notna() & v.notna()
    if ok.sum() == 0:
        return None
    tmp = pd.DataFrame({"d": d[ok], "v": v[ok]})
    yearly = tmp.set_index("d")["v"].resample("YS").mean().dropna()
    if yearly.empty:
        return None
    med = float(yearly.median())
    yr = int((yearly - med).abs().idxmin().year)
    return yr

def score_to_0_1000(series: pd.Series, dates: pd.Series, baseline_year: int | None) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    d = pd.to_datetime(dates, errors="coerce")
    ok = s.notna() & d.notna()
    if ok.sum() == 0:
        return pd.Series([np.nan] * len(series), index=series.index)

    if baseline_year is None:
        baseline = float(s[ok].median())
    else:
        base_mask = ok & (d.dt.year == int(baseline_year))
        baseline = float(s[base_mask].mean()) if base_mask.sum() else float(s[ok].median())

    centered = s - baseline
    max_abs = float(np.nanmax(np.abs(centered[ok])))
    if not np.isfinite(max_abs) or max_abs == 0:
        return pd.Series([500.0] * len(series), index=series.index)

    scaled = 500.0 + (centered / max_abs) * 500.0
    return scaled.clip(0, 1000)

# =========================================================
# Recompute
# =========================================================
def recompute_all():
    if not st.session_state.get("has_run"):
        return
    birth_local = st.session_state.get("birth_local")
    lat = st.session_state.get("lat")
    lon = st.session_state.get("lon")
    sid_mode_key = st.session_state.get("sid_mode_key")
    progression_type = st.session_state.get("progression_type")
    if birth_local is None or lat is None or lon is None or sid_mode_key is None or progression_type is None:
        return

    birth = compute_birth_chart(birth_local, lat, lon, sid_mode=sid_mode_key)
    st.session_state["birth"] = birth

    year = int(st.session_state.get("year", datetime.now().year))
    st.session_state["tran"] = compute_transit_chart(birth_local, lat, lon, year, sid_mode=sid_mode_key)

    if progression_type.startswith("Secondary"):
        st.session_state["prog"] = compute_progressed_chart(birth_local, lat, lon, year, sid_mode=sid_mode_key)
    else:
        st.session_state["prog"] = compute_solar_arc_chart(birth_local, lat, lon, year, sid_mode=sid_mode_key)

    st.session_state["life_df"] = generate_general_life_df(
        birth_local=birth_local,
        lat=lat,
        lon=lon,
        sid_mode_key=sid_mode_key,
        progression_type=progression_type,
        max_years=120,
        step_months=3,
    )
    st.session_state["career_df"] = generate_career_df(
        birth_local=birth_local,
        lat=lat,
        lon=lon,
        sid_mode_key=sid_mode_key,
        progression_type=progression_type,
        max_years=120,
        step_months=3,
    )

def recompute_year_only():
    if not st.session_state.get("has_run"):
        return
    birth_local = st.session_state.get("birth_local")
    lat = st.session_state.get("lat")
    lon = st.session_state.get("lon")
    sid_mode_key = st.session_state.get("sid_mode_key")
    progression_type = st.session_state.get("progression_type")
    if birth_local is None or lat is None or lon is None or sid_mode_key is None or progression_type is None:
        return

    year = int(st.session_state.get("year", datetime.now().year))
    st.session_state["tran"] = compute_transit_chart(birth_local, lat, lon, year, sid_mode=sid_mode_key)
    if progression_type.startswith("Secondary"):
        st.session_state["prog"] = compute_progressed_chart(birth_local, lat, lon, year, sid_mode=sid_mode_key)
    else:
        st.session_state["prog"] = compute_solar_arc_chart(birth_local, lat, lon, year, sid_mode=sid_mode_key)

# =========================================================
# UI Inputs
# =========================================================
countries, country_name_to_code, city_df, tf = load_place_data()

# defaults
default_name = st.session_state.get("name", "")
default_dob = st.session_state.get("dob", date(2000, 1, 1))
default_tob_text = st.session_state.get("tob_text", "")
default_country = st.session_state.get("country", "India")

# IMPORTANT: We keep a *draft* city input (user types) and only apply it when they click Search.
draft_city = st.session_state.get("draft_city", st.session_state.get("city_query", "Hyderabad"))
applied_city = st.session_state.get("city_query", "Hyderabad")

row1 = st.columns([1.2, 1.2, 1.4, 2.0, 2.2, 1.0], vertical_alignment="bottom")
with row1[0]:
    name = st.text_input("Name", value=default_name, placeholder="Enter name…")
with row1[1]:
    dob = st.date_input(
    "DOB",
    value=default_dob,
    min_value=date(1800, 1, 1),
    max_value=date(2100, 12, 31),
    format="DD/MM/YYYY",
)
with row1[2]:
    tob_text = st.text_input("TOB", value=default_tob_text, placeholder="HH:MM or HH:MM:SS")
with row1[3]:
    country_names = [x[0] for x in countries]
    idx = country_names.index(default_country) if default_country in country_names else 0
    country = st.selectbox("Country", country_names, index=idx)
with row1[4]:
    draft_city = st.text_input(
        "City search (type, then click Search)",
        value=draft_city,
        placeholder="Type city… (e.g., Hyderabad)",
        key="draft_city",
    )
with row1[5]:
    ui_mode = st.selectbox("UI", ["Auto", "Light", "Dark"], index=0)

row_search = st.columns([1.0, 5.0], vertical_alignment="bottom")
with row_search[0]:
    do_search = st.button("Search city", use_container_width=True)
with row_search[1]:
    st.caption(f"Applied city filter: **{applied_city}** (updates only when you click **Search city**)")

# Apply city filter only on button press (NO interruptions while typing)
if do_search:
    st.session_state["city_query"] = st.session_state.get("draft_city", "").strip()
    applied_city = st.session_state["city_query"]

_base = (st.get_option("theme.base") or "light").lower()
effective_mode = ("Dark" if _base == "dark" else "Light") if ui_mode == "Auto" else ui_mode

tob_val = parse_time_text(tob_text)
if tob_val is None:
    st.warning("Enter TOB in HH:MM or HH:MM:SS (example: 22:01 or 22:01:05).")
    st.stop()

row2 = st.columns([1.6, 1.6, 1.6, 1.2], vertical_alignment="bottom")
with row2[0]:
    sid_mode = st.selectbox(
        "Ayanamsa",
        ["KRISHNAMURTI (KP)", "LAHIRI"],
        index=0,
        key="sid_mode_ui",
        on_change=recompute_all,
    )
with row2[1]:
    st.selectbox(
        "Progression Type",
        ["Secondary (Day-for-Year)", "Solar Arc"],
        key="progression_type",
        on_change=recompute_all,
    )
with row2[2]:
    st.number_input(
        "Year",
        min_value=1900,
        max_value=2200,
        value=int(st.session_state.get("year", datetime.now().year)),
        step=1,
        key="year",
        on_change=recompute_year_only,
    )
with row2[3]:
    compute_now = st.button("Compute", type="primary", use_container_width=True)

sid_mode_key = "KRISHNAMURTI" if "KRISHNAMURTI" in sid_mode else "LAHIRI"

# City matches based on APPLIED filter only (stable)
country_code = country_name_to_code[country]
matches = find_city_matches_stable(city_df, country_code, st.session_state.get("city_query", "Hyderabad"), limit=60)

if matches.empty:
    # Fallback: Nominatim (OpenStreetMap) for towns/villages not present in geonamescache
    q_city = str(st.session_state.get("city_query", "") or "").strip()
    if not q_city:
        st.warning("No matches. Type a place name and click Search city.")
        st.stop()

    geocoder = get_geocoder()
    try:
        # Use country context to improve accuracy
        q = f"{q_city}, {country}"
        hits = geocoder.geocode(q, exactly_one=False, limit=10, addressdetails=True)
    except Exception:
        hits = None

    if not hits:
        st.warning("No matches found (including fallback search). Try adding state/district (e.g., Karimnagar, Telangana).")
        st.stop()

    nom_rows = []
    for h in hits:
        try:
            lat_h = float(getattr(h, "latitude", None))
            lon_h = float(getattr(h, "longitude", None))
        except Exception:
            continue
        disp = getattr(h, "address", None) or getattr(h, "raw", {}).get("display_name", "") or str(h)
        nom_rows.append({"label": disp, "lat": lat_h, "lon": lon_h})

    if not nom_rows:
        st.warning("Fallback search returned results but coordinates were unavailable.")
        st.stop()

    nom_labels = [r["label"] for r in nom_rows]
    pick = st.selectbox("Pick place (fallback search)", nom_labels, index=0)
    chosen = nom_rows[nom_labels.index(pick)]

    # Build a Place directly
    tz = tf.timezone_at(lat=chosen["lat"], lng=chosen["lon"]) or "UTC"
    tz = safe_tz(tz)

    place = Place(
        country_name=country,
        country_code=country_name_to_code.get(country, ""),
        city_name=pick.split(",")[0].strip() or q_city,
        admin1_code=None,
        lat=float(chosen["lat"]),
        lon=float(chosen["lon"]),
        timezone=tz,
    )

    st.caption(
        f"Resolved: **{label_place(place)}** | TZ: **{place.timezone}** | "
        f"Lat: **{place.lat:.6f}** | Lon: **{place.lon:.6f}** | "
        f"DOB: **{fmt_dmy(dob)}** | TOB: **{tob_val.strftime('%H:%M:%S')}**"
    )

    birth_local = datetime.combine(dob, tob_val).replace(tzinfo=ZoneInfo(place.timezone))

    # Persist inputs
    st.session_state["name"] = name
    st.session_state["dob"] = dob
    st.session_state["tob_text"] = tob_text
    st.session_state["country"] = country
    st.session_state["lat"] = place.lat
    st.session_state["lon"] = place.lon
    st.session_state["sid_mode_key"] = sid_mode_key
    st.session_state["tzname"] = place.timezone
    st.session_state["birth_local"] = birth_local
else:

    # Stable selection by geonameid (no jumping)
    if "selected_geonameid" not in st.session_state:
        st.session_state["selected_geonameid"] = int(matches.iloc[0]["geonameid"])

    labels = []
    id_list = []
    for _, r in matches.iterrows():
        pop = int(r.get("population", 0) or 0)
        admin = r.get("admin1code", "") or ""
        labels.append(f"{r['name']} | {admin} | pop:{pop:,}")
        id_list.append(int(r["geonameid"]))

    # keep current id if still in list, else choose first
    if st.session_state["selected_geonameid"] in id_list:
        default_idx = id_list.index(st.session_state["selected_geonameid"])
    else:
        default_idx = 0
        st.session_state["selected_geonameid"] = id_list[0]

    sel_label = st.selectbox("Pick place", labels, index=default_idx)
    sel_idx = labels.index(sel_label)
    st.session_state["selected_geonameid"] = id_list[sel_idx]
    sel_row = matches.iloc[sel_idx]

    place = resolve_place(country, country_code, sel_row, tf)

    st.caption(
        f"Resolved: **{label_place(place)}** | TZ: **{place.timezone}** | "
        f"Lat: **{place.lat:.6f}** | Lon: **{place.lon:.6f}** | "
        f"DOB: **{fmt_dmy(dob)}** | TOB: **{tob_val.strftime('%H:%M:%S')}**"
    )

    birth_local = datetime.combine(dob, tob_val).replace(tzinfo=ZoneInfo(place.timezone))

    # Persist inputs
    st.session_state["name"] = name
    st.session_state["dob"] = dob
    st.session_state["tob_text"] = tob_text
    st.session_state["country"] = country
    st.session_state["lat"] = place.lat
    st.session_state["lon"] = place.lon
    st.session_state["sid_mode_key"] = sid_mode_key
    st.session_state["tzname"] = place.timezone
    st.session_state["birth_local"] = birth_local
# =========================================================
# Compute
# =========================================================
if compute_now:
    with st.spinner("Computing charts + life engines…"):
        st.session_state["has_run"] = True
        recompute_all()

birth = st.session_state.get("birth")
prog = st.session_state.get("prog")
tran = st.session_state.get("tran")
life_df = st.session_state.get("life_df")
career_df = st.session_state.get("career_df")

if not st.session_state.get("has_run") or not birth or not prog or not tran or life_df is None or career_df is None:
    st.info("Enter birth details and click **Compute**.")
    st.stop()

# Sidebar dasha
with st.sidebar:
    dasha_selector_click(birth["md_periods"])
    sel = st.session_state.get("selected_period")
    if sel:
        st.caption("Selected window:")
        st.write(sel["label"])
    else:
        st.caption("Selected window: none")

st.divider()

# =========================================================
# Panchang (compact)
# =========================================================
topL, topR = st.columns([1.25, 1.75], vertical_alignment="top")

with topL:
    sun = next(p for p in birth["planets"] if p.name == "Sun")
    moon = next(p for p in birth["planets"] if p.name == "Moon")
    weekday = birth_local.strftime("%A")
    rashi = moon.sign
    nak_name = birth["nakshatra"]["name"]
    pada = compute_pada_from_moon_lon(moon.lon)
    panch = compute_tithi_yoga_karana(sun.lon, moon.lon)
    nak_pada = f"{nak_name}-{pada}"

    render_panchang_panel_compact(effective_mode, {
        "weekday": weekday,
        "rashi": rashi,
        "nak_pada": nak_pada,
        "tithi": f"{panch['paksha']} {panch['tithi']}",
        "yoga": panch["yoga"],
        "karana": panch["karana"],
    })

with topR:
    st.caption(
        f"Ayanamsa: **{sid_mode_key}**  |  Progression: **{st.session_state['progression_type']}**  |  "
        f"Year: **{int(st.session_state['year'])}**"
    )

# =========================================================
# Birth + Progressed charts
# =========================================================
c1, c2 = st.columns(2, vertical_alignment="top")

asc_sign = birth["houses"]["asc_sign"]
center_birth = [
    f"<b>{name or '—'}</b>",
    f"{fmt_dmy(dob)} {tob_val.strftime('%H:%M:%S')}",
    f"{label_place(place)}",
    f"Lagna: <b>{asc_sign}</b>",
    f"Nak: <b>{nak_pada}</b>",
]

with c1:
    render_south_chart("Birth Chart (South Indian)", birth["planets"], birth["houses"], center_birth, effective_mode, size_mode="half")

with c2:
    center_prog = [
        f"<b>{name or '—'}</b>",
        f"{st.session_state['progression_type']}",
        f"Year: <b>{int(st.session_state['year'])}</b>",
        f"{fmt_dmy(prog['dt_local'])} {prog['dt_local']:%H:%M:%S}",
    ]
    render_south_chart(
        f"Progressed Chart ({int(st.session_state['year'])})",
        prog["planets"], prog["houses"], center_prog,
        effective_mode, size_mode="half"
    )

# =========================================================
# Transit chart + BCP
# =========================================================
st.markdown("---")

age = int(st.session_state["year"]) - birth["birth_local"].year
active_house = bcp_house_from_age(age)
tran["houses"]["bcp_house"] = active_house
tran["houses"]["bcp_age"] = age

st.caption(f"BCP Activation: Age **{age}** → Activated House **{active_house}**")

center_tr = [
    f"<b>{name or '—'}</b>",
    f"Transit year <b>{int(st.session_state['year'])}</b>",
    f"{fmt_dmy(tran['dt_local'])} {tran['dt_local']:%H:%M:%S}",
]
render_south_chart(
    f"Transit Chart ({int(st.session_state['year'])})",
    tran["planets"], tran["houses"], center_tr,
    effective_mode, size_mode="full"
)

# =========================================================
# GRAPHS (Life + Career separate)
# =========================================================
st.divider()
st.subheader("Graphs & Timeline")

CHART_BG = "#0b0f17"
GRID_COL = "rgba(255,255,255,0.28)"
ZERO_COL = "rgba(255,255,255,0.55)"
FONT_COL = "rgba(255,255,255,0.92)"

ctl1, ctl2, ctl3 = st.columns([1.2, 1.2, 2.0], vertical_alignment="bottom")
with ctl1:
    highlight_mode = st.radio("Selection behavior", ["Highlight only", "Zoom to selection"], horizontal=True, key="sel_mode_graph")
with ctl2:
    baseline_mode = st.selectbox("Baseline", ["Auto (recommended)", "Manual year"], index=0, key="base_mode_graph")
with ctl3:
    manual_year = st.number_input("Manual baseline year", min_value=1900, max_value=2200, value=2016, step=1, key="manual_base_year_graph")

sel = st.session_state.get("selected_period")

def plot_engine(df: pd.DataFrame, title: str, key_prefix: str):
    yearly = (
        df.set_index("date_local")["score_total"]
        .resample("YS")
        .mean()
        .dropna()
        .reset_index()
    )
    yearly.columns = ["date_local", "score_total"]

    if baseline_mode.startswith("Auto"):
        by = auto_baseline_year(yearly["date_local"], yearly["score_total"])
    else:
        by = int(manual_year)

    yearly["life_0_1000"] = score_to_0_1000(yearly["score_total"], yearly["date_local"], baseline_year=by)

    fig_full = go.Figure()
    fig_full.add_trace(go.Scatter(
        x=yearly["date_local"],
        y=yearly["life_0_1000"],
        mode="lines+markers",
        line=dict(color="white", width=2),
        marker=dict(color="white", size=6),
    ))

    if sel:
        s0 = to_naive_ts(sel.get("start"))
        s1 = to_naive_ts(sel.get("end"))
        if pd.notna(s0) and pd.notna(s1):
            fig_full.add_vrect(x0=s0, x1=s1, opacity=0.10, line_width=0)
            if highlight_mode == "Zoom to selection":
                fig_full.update_xaxes(range=[s0, s1])

    fig_full.update_layout(
        height=520,
        margin=dict(l=20, r=20, t=10, b=20),
        hovermode="x unified",
        showlegend=False,
        paper_bgcolor=CHART_BG,
        plot_bgcolor=CHART_BG,
        font=dict(color=FONT_COL),
        title=dict(text=title, font=dict(color=FONT_COL, size=16)),
    )
    fig_full.update_xaxes(
        dtick="M12",
        tickformat="%Y",
        showgrid=True,
        gridcolor=GRID_COL,
        zeroline=False,
        linecolor=GRID_COL,
        tickfont=dict(color=FONT_COL),
    )
    fig_full.update_yaxes(
        range=[0, 1000],
        dtick=100,
        showgrid=True,
        gridcolor=GRID_COL,
        zeroline=True,
        zerolinecolor=ZERO_COL,
        zerolinewidth=1,
        tickfont=dict(color=FONT_COL),
    )

    st.plotly_chart(fig_full, use_container_width=True, key=f"{key_prefix}_full")

    st.markdown("### Detail inside selected window (3 months)")

    if not sel:
        st.info("Select MD (and optionally AD/PD) from the left, then the detail view will appear here.")
        return

    s0 = to_naive_ts(sel.get("start"))
    s1 = to_naive_ts(sel.get("end"))
    window_df = df[(df["date_local"] >= s0) & (df["date_local"] <= s1)].copy()
    if window_df.empty:
        st.warning("No data points in this selected window.")
        return

    detail = (
        window_df.set_index("date_local")["score_total"]
        .resample("3MS")
        .mean()
        .dropna()
        .reset_index()
    )
    detail.columns = ["date_local", "score_total"]
    detail["life_0_1000"] = score_to_0_1000(detail["score_total"], detail["date_local"], baseline_year=by)

    fig_det = go.Figure()
    fig_det.add_trace(go.Scatter(
        x=detail["date_local"],
        y=detail["life_0_1000"],
        mode="lines+markers",
        line=dict(color="white", width=2),
        marker=dict(color="white", size=7),
    ))
    fig_det.add_vrect(x0=s0, x1=s1, opacity=0.10, line_width=0)

    fig_det.update_layout(
        height=480,
        margin=dict(l=20, r=20, t=10, b=20),
        hovermode="x unified",
        showlegend=False,
        paper_bgcolor=CHART_BG,
        plot_bgcolor=CHART_BG,
        font=dict(color=FONT_COL),
    )
    fig_det.update_xaxes(
        showgrid=True,
        gridcolor=GRID_COL,
        zeroline=False,
        linecolor=GRID_COL,
        tickfont=dict(color=FONT_COL),
    )
    fig_det.update_yaxes(
        range=[0, 1000],
        dtick=100,
        showgrid=True,
        gridcolor=GRID_COL,
        zeroline=True,
        zerolinecolor=ZERO_COL,
        zerolinewidth=1,
        tickfont=dict(color=FONT_COL),
    )

    st.plotly_chart(fig_det, use_container_width=True, key=f"{key_prefix}_detail")

tab1, tab2 = st.tabs(["Life (General)", "Career / Profession"])
with tab1:
    plot_engine(life_df, "Life Path (General)", "life")
with tab2:
    plot_engine(career_df, "Career / Profession Engine", "career")

st.caption("City search is now stable: it updates only when you click **Search city** (no interruptions).")