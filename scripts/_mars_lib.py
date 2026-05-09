"""Phase 4 helpers: palette transformer + Mars overlays + render utility.

Kept separate from `17_mars_generate.py` so it can be imported from the
embedding script (`18_mars_embed.py`) and figure script (`19_mars_render.py`)
as well, without re-running generation.
"""
from __future__ import annotations

import io
import re
import sys
from pathlib import Path

import cairosvg
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts._earth_templates import CATEGORY_TEMPLATES  # noqa: E402

OUT_DIR = ROOT / "out" / "mars"
PNG_OUT_DIR = OUT_DIR / "flags"
SVG_OUT_DIR = OUT_DIR / "svg"


# ─────────────────────────── Mars palette ─────────────────────────────────
#
# Mars-specific replacement colors.  Each "earth color family" maps to a
# Mars hex.  We never replace pure black (it remains a valid contrast).
# A few of the Mars colors are deliberately analogue-but-shifted; the goal
# is the flags read as Mars at a glance even when the structural template
# is unmistakably Nordic / pan-Arab / Latin / etc.
#
#   blue   → rust-orange (no blue on a barely-terraformed Mars sky)
#   white  → pale-cream  (Mars's pale dusty haze)
#   green  → sage        (the only plausible terraformed flora green)
#   red    → iron-red    (Mars regolith)
#   yellow → sulfur      (Tharsis sulfate evaporite)
#   accent → cyan-ice    (subsurface H2O ice, used as the Mars-specific accent)

MARS_HEX = {
    "iron_red":     "#9C2A1A",
    "rust_orange":  "#C9663C",
    "pale_cream":   "#F0E4C8",
    "sage":         "#7B8D6A",
    "sulfur":       "#D9B85B",
    "dark_basalt":  "#1A0E0A",
    "ice_cyan":     "#A8C8D4",
    "iron_dark":    "#5A1810",
    "polar_white":  "#FAF1DC",
}

# Ordered list of "Earth color family anchors" → Mars hex.  We classify each
# hex string in the SVG by nearest anchor and substitute.
_EARTH_ANCHORS: list[tuple[tuple[int, int, int], str]] = [
    # blue family
    ((  1, 33, 105), "rust_orange"),  # #012169 union jack blue
    ((  0, 40, 104), "rust_orange"),  # #002868 stars-and-stripes blue
    (( 30, 75, 175), "rust_orange"),  # mid blue
    ((100,160,210), "ice_cyan"),      # light blue → ice
    # red family
    ((207, 20, 43), "iron_red"),       # #cf142b pan-arab red
    ((191, 10, 48), "iron_red"),       # #bf0a30 stars-and-stripes red
    ((200,  0,  0), "iron_red"),       # generic red
    ((140, 20, 25), "iron_dark"),
    # green family
    ((  0,122, 61), "sage"),           # #007a3d pan-arab green
    ((  7,137, 48), "sage"),           # #078930 pan-african green
    (( 30,130, 60), "sage"),
    # yellow family
    ((255,215,  0), "sulfur"),         # #ffd700 communist gold star
    ((252,221,  9), "sulfur"),         # #fcdd09 pan-african yellow
    ((245,200, 40), "sulfur"),
    # white
    ((255,255,255), "pale_cream"),
    ((247,244,236), "pale_cream"),
    # black
    ((  0,  0,  0), "dark_basalt"),
    (( 20, 20, 20), "dark_basalt"),
]

# A few accent hexes per region — at least one Mars-specific accent in each
# generated flag.  Keys are region_id; the accent gets injected into the
# overlay SVG fragment via the `{accent}` placeholder.
MARS_ACCENTS = {
    "mars-tharsis":      MARS_HEX["sulfur"],
    "mars-olympus":      MARS_HEX["dark_basalt"],
    "mars-hellas":       MARS_HEX["ice_cyan"],
    "mars-argyre":       MARS_HEX["polar_white"],
    "mars-marineris":    MARS_HEX["iron_dark"],
    "mars-vastitas":     MARS_HEX["ice_cyan"],
    "mars-australe":     MARS_HEX["polar_white"],
    "mars-arabia":       MARS_HEX["sulfur"],
    "mars-sirenum":      MARS_HEX["iron_dark"],
    "mars-cydonia":      MARS_HEX["pale_cream"],
    "mars-elysium":      MARS_HEX["sulfur"],
    "mars-amazonis":     MARS_HEX["sage"],
    "mars-utopia":       MARS_HEX["sulfur"],
    "mars-acidalia":     MARS_HEX["ice_cyan"],
    "mars-chryse":       MARS_HEX["sulfur"],
    "mars-eridania":     MARS_HEX["ice_cyan"],
    "mars-aeolis":       MARS_HEX["pale_cream"],
    "mars-mangala":      MARS_HEX["sulfur"],
    "mars-noachis":      MARS_HEX["iron_dark"],
    "mars-solis":        MARS_HEX["sulfur"],
    "mars-margaritifer": MARS_HEX["pale_cream"],
    "mars-phlegra":      MARS_HEX["dark_basalt"],
    "mars-thaumasia":    MARS_HEX["iron_dark"],
    "mars-tempe":        MARS_HEX["sage"],
    "mars-sabaea":       MARS_HEX["sulfur"],
}

# For backwards-name reference from 17_mars_generate.py
EARTH_TO_MARS_HEX = MARS_HEX


# ─────────────────────────── palette transform ─────────────────────────────

_HEX_RE = re.compile(r"#([0-9a-fA-F]{6})")
_HEX3_RE = re.compile(r"#([0-9a-fA-F]{3})\b")


def _hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def _classify_to_mars(rgb: tuple[int, int, int]) -> str:
    r, g, b = rgb
    best_name = "pale_cream"
    best_d = float("inf")
    for anchor, name in _EARTH_ANCHORS:
        d = (r - anchor[0]) ** 2 + (g - anchor[1]) ** 2 + (b - anchor[2]) ** 2
        if d < best_d:
            best_d = d
            best_name = name
    return MARS_HEX[best_name]


def apply_mars_palette(svg: str, region_id: str = "") -> str:
    """Substitute every #rrggbb in the SVG with its nearest Mars equivalent."""
    cache: dict[str, str] = {}

    def _sub(match: re.Match) -> str:
        original = match.group(0)
        if original in cache:
            return cache[original]
        rgb = _hex_to_rgb(original)
        mars = _classify_to_mars(rgb)
        cache[original] = mars
        return mars

    # Expand 3-char hex first to avoid double-matching.
    def _expand3(m: re.Match) -> str:
        h = m.group(1)
        return "#" + "".join(c * 2 for c in h)

    svg = _HEX3_RE.sub(_expand3, svg)
    return _HEX_RE.sub(_sub, svg)


# ─────────────────────────── Mars overlays ─────────────────────────────────
#
# Each overlay is an SVG fragment that gets inserted just before </svg>.
# Coordinates assume the 480x320 canvas.  Overlays are designed to occupy
# ≤25% of canvas area.  `{accent}` is substituted with the region's accent
# hex from MARS_ACCENTS.

def _olympus_silhouette(x: int = 360, y: int = 220, accent: str = "#1A0E0A") -> str:
    # Shield-shaped volcanic dome with summit caldera.
    return (
        f'<path d="M {x-60},{y} '
        f'C {x-50},{y-40} {x-25},{y-65} {x},{y-70} '
        f'C {x+25},{y-65} {x+50},{y-40} {x+60},{y} Z" '
        f'fill="{accent}" opacity="0.92"/>'
        f'<ellipse cx="{x}" cy="{y-62}" rx="14" ry="5" fill="{MARS_HEX["iron_dark"]}" opacity="0.9"/>'
    )


def _polar_cap(x: int = 240, y: int = 80, accent: str = MARS_HEX["polar_white"]) -> str:
    # Crescent-cap silhouette over a horizon.
    return (
        f'<path d="M {x-80},{y+30} '
        f'A 90 60 0 0 1 {x+80},{y+30} '
        f'L {x+60},{y+50} L {x-60},{y+50} Z" '
        f'fill="{accent}" opacity="0.95"/>'
    )


def _phobos_deimos(x: int = 360, y: int = 70, accent: str = MARS_HEX["pale_cream"]) -> str:
    return (
        f'<circle cx="{x-12}" cy="{y}" r="14" fill="{accent}" opacity="0.95"/>'
        f'<circle cx="{x+22}" cy="{y+12}" r="8"  fill="{MARS_HEX["sulfur"]}" opacity="0.9"/>'
    )


def _dust_storm(x: int = 360, y: int = 240, accent: str = MARS_HEX["pale_cream"]) -> str:
    # Concentric arcs forming a hurricane spiral.
    return (
        f'<g stroke="{accent}" stroke-width="3" fill="none" opacity="0.92">'
        f'<path d="M {x},{y} m -38,0 a 38 38 0 1 0 76 0"/>'
        f'<path d="M {x},{y} m -24,0 a 24 24 0 1 0 48 0"/>'
        f'<path d="M {x},{y} m -10,0 a 10 10 0 1 0 20 0"/>'
        f'</g>'
    )


def _canyon_line(accent: str = MARS_HEX["iron_dark"]) -> str:
    # Long horizontal canyon scar with two branching tributaries.
    return (
        f'<g fill="{accent}" opacity="0.88">'
        f'<path d="M 30,200 L 450,200 L 445,210 L 35,210 Z"/>'
        f'<path d="M 110,200 L 150,170 L 155,176 L 115,206 Z"/>'
        f'<path d="M 320,200 L 360,232 L 354,236 L 314,206 Z"/>'
        f'</g>'
    )


def _terraforming_chamber(x: int = 240, y: int = 230, accent: str = MARS_HEX["sage"]) -> str:
    # Geodesic dome silhouette with vertical strut.
    return (
        f'<g fill="{accent}" opacity="0.92">'
        f'<path d="M {x-50},{y} '
        f'A 50 50 0 0 1 {x+50},{y} L {x+50},{y+5} L {x-50},{y+5} Z"/>'
        f'<rect x="{x-2}" y="{y-50}" width="4" height="50"/>'
        f'</g>'
        f'<line x1="{x-50}" y1="{y}" x2="{x+50}" y2="{y}" stroke="{MARS_HEX["dark_basalt"]}" stroke-width="2"/>'
        f'<line x1="{x-30}" y1="{y-40}" x2="{x+30}" y2="{y-40}" stroke="{MARS_HEX["dark_basalt"]}" stroke-width="1.5" opacity="0.6"/>'
    )


def _crater(x: int = 240, y: int = 160, r: int = 55, accent: str = MARS_HEX["pale_cream"]) -> str:
    # Single bowl-shaped crater (impact basin).
    return (
        f'<circle cx="{x}" cy="{y}" r="{r}" fill="{accent}" opacity="0.94"/>'
        f'<circle cx="{x}" cy="{y}" r="{int(r*0.72)}" fill="{MARS_HEX["iron_dark"]}" opacity="0.55"/>'
    )


def _face_silhouette(x: int = 360, y: int = 90, accent: str = MARS_HEX["pale_cream"]) -> str:
    # Stylized "Face on Mars" outline.
    return (
        f'<g fill="{accent}" opacity="0.93">'
        f'<ellipse cx="{x}" cy="{y}" rx="32" ry="42"/>'
        f'</g>'
        f'<g fill="{MARS_HEX["dark_basalt"]}" opacity="0.85">'
        f'<circle cx="{x-9}" cy="{y-6}" r="3"/>'
        f'<circle cx="{x+9}" cy="{y-6}" r="3"/>'
        f'<rect x="{x-7}" y="{y+12}" width="14" height="3"/>'
        f'</g>'
    )


def _sun_disc(x: int = 240, y: int = 160, accent: str = MARS_HEX["sulfur"]) -> str:
    # Sunburst with rays.
    rays = ""
    import math as _m
    for i in range(8):
        a = i * (2 * _m.pi / 8)
        x1 = x + 38 * _m.cos(a); y1 = y + 38 * _m.sin(a)
        x2 = x + 60 * _m.cos(a); y2 = y + 60 * _m.sin(a)
        rays += f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="{accent}" stroke-width="5"/>'
    return (
        f'<circle cx="{x}" cy="{y}" r="34" fill="{accent}" opacity="0.95"/>'
        f'{rays}'
    )


def _lander(x: int = 240, y: int = 235, accent: str = MARS_HEX["pale_cream"]) -> str:
    # Stylized Viking-style lander silhouette.
    return (
        f'<g fill="{accent}" opacity="0.95">'
        f'<polygon points="{x-30},{y} {x+30},{y} {x+22},{y-18} {x-22},{y-18}"/>'
        f'<rect x="{x-4}" y="{y-32}" width="8" height="14"/>'
        f'<line x1="{x-30}" y1="{y}" x2="{x-46}" y2="{y+18}" stroke="{accent}" stroke-width="3"/>'
        f'<line x1="{x+30}" y1="{y}" x2="{x+46}" y2="{y+18}" stroke="{accent}" stroke-width="3"/>'
        f'</g>'
    )


def _rover_tracks(accent: str = MARS_HEX["iron_dark"]) -> str:
    # Pair of parallel curving tire tracks.
    return (
        f'<g stroke="{accent}" stroke-width="3" fill="none" opacity="0.85" stroke-dasharray="8 6">'
        f'<path d="M 50,260 C 180,210 320,300 440,240"/>'
        f'<path d="M 50,275 C 180,225 320,315 440,255"/>'
        f'</g>'
    )


def _moon_crescent(x: int = 360, y: int = 80, accent: str = MARS_HEX["pale_cream"]) -> str:
    return (
        f'<circle cx="{x}" cy="{y}" r="28" fill="{accent}" opacity="0.95"/>'
        f'<circle cx="{x+10}" cy="{y-2}" r="24" fill="{MARS_HEX["iron_red"]}" opacity="1"/>'
    )


def _mountain_belt(accent: str = MARS_HEX["dark_basalt"]) -> str:
    # Diagonal jagged ridge crossing flag.
    return (
        f'<polygon points="0,260 60,200 110,240 170,180 220,230 280,170 340,220 400,160 480,210 480,320 0,320" '
        f'fill="{accent}" opacity="0.9"/>'
    )


def _pearl(x: int = 240, y: int = 160, accent: str = MARS_HEX["pale_cream"]) -> str:
    return (
        f'<circle cx="{x}" cy="{y}" r="36" fill="{accent}" opacity="0.97"/>'
        f'<circle cx="{x-8}" cy="{y-10}" r="10" fill="#ffffff" opacity="0.55"/>'
    )


def _sea_wave(accent: str = MARS_HEX["ice_cyan"]) -> str:
    # Wavy band along the bottom — the lost sea.
    return (
        f'<path d="M 0,250 Q 60,230 120,250 T 240,250 T 360,250 T 480,250 L 480,320 L 0,320 Z" '
        f'fill="{accent}" opacity="0.9"/>'
        f'<path d="M 0,270 Q 60,255 120,270 T 240,270 T 360,270 T 480,270" '
        f'stroke="{MARS_HEX["pale_cream"]}" stroke-width="2" fill="none" opacity="0.6"/>'
    )


def _ancient_sigil(x: int = 240, y: int = 160, accent: str = MARS_HEX["iron_dark"]) -> str:
    # Stylized concentric-ring sigil for the most ancient terrain.
    return (
        f'<g fill="none" stroke="{accent}" stroke-width="3" opacity="0.9">'
        f'<circle cx="{x}" cy="{y}" r="40"/>'
        f'<circle cx="{x}" cy="{y}" r="28"/>'
        f'<circle cx="{x}" cy="{y}" r="16"/>'
        f'</g>'
        f'<circle cx="{x}" cy="{y}" r="6" fill="{accent}"/>'
    )


def _mars_globe(x: int = 360, y: int = 80, accent: str = MARS_HEX["pale_cream"]) -> str:
    # Small Mars globe with a faint canyon scar.
    return (
        f'<circle cx="{x}" cy="{y}" r="28" fill="{MARS_HEX["iron_red"]}" stroke="{accent}" stroke-width="2"/>'
        f'<path d="M {x-22},{y+4} Q {x},{y-2} {x+22},{y+4}" stroke="{MARS_HEX["iron_dark"]}" stroke-width="2" fill="none"/>'
        f'<circle cx="{x-6}" cy="{y-12}" r="3" fill="{MARS_HEX["polar_white"]}"/>'
    )


# Map region_id → overlay SVG fragment.  The actual fragments are computed
# fresh per call so that we can swap accents per region in a follow-up.
def _overlay_for(region_id: str) -> str:
    accent = MARS_ACCENTS.get(region_id, MARS_HEX["pale_cream"])
    if region_id == "mars-tharsis":
        return _olympus_silhouette(x=380, y=220, accent=MARS_HEX["dark_basalt"])
    if region_id == "mars-olympus":
        return _olympus_silhouette(x=240, y=230, accent=MARS_HEX["dark_basalt"])
    if region_id == "mars-hellas":
        return _crater(x=240, y=160, r=70, accent=MARS_HEX["pale_cream"])
    if region_id == "mars-argyre":
        return _crater(x=240, y=160, r=58, accent=MARS_HEX["polar_white"])
    if region_id == "mars-marineris":
        return _canyon_line()
    if region_id == "mars-vastitas":
        return _polar_cap(x=240, y=70)
    if region_id == "mars-australe":
        return _polar_cap(x=240, y=70)
    if region_id == "mars-arabia":
        return _phobos_deimos(x=370, y=70)
    if region_id == "mars-sirenum":
        return _mountain_belt()
    if region_id == "mars-cydonia":
        return _face_silhouette(x=240, y=160)
    if region_id == "mars-elysium":
        return _olympus_silhouette(x=380, y=240, accent=MARS_HEX["iron_dark"])
    if region_id == "mars-amazonis":
        return _dust_storm(x=360, y=80, accent=MARS_HEX["pale_cream"])
    if region_id == "mars-utopia":
        return _lander(x=240, y=240, accent=MARS_HEX["pale_cream"])
    if region_id == "mars-acidalia":
        return _phobos_deimos(x=370, y=70)
    if region_id == "mars-chryse":
        return _lander(x=240, y=240)
    if region_id == "mars-eridania":
        return _sea_wave()
    if region_id == "mars-aeolis":
        return _rover_tracks()
    if region_id == "mars-mangala":
        return _terraforming_chamber(x=240, y=240, accent=MARS_HEX["sage"])
    if region_id == "mars-noachis":
        return _ancient_sigil(x=240, y=160)
    if region_id == "mars-solis":
        return _sun_disc(x=240, y=160)
    if region_id == "mars-margaritifer":
        return _pearl(x=240, y=160)
    if region_id == "mars-phlegra":
        return _mountain_belt()
    if region_id == "mars-thaumasia":
        return _olympus_silhouette(x=380, y=230, accent=MARS_HEX["iron_dark"])
    if region_id == "mars-tempe":
        return _terraforming_chamber(x=240, y=240, accent=MARS_HEX["sage"])
    if region_id == "mars-sabaea":
        return _moon_crescent(x=370, y=70)
    return _mars_globe()


class _OverlayLookup:
    """Dict-like wrapper so callers can do MARS_OVERLAYS[region_id]."""
    def get(self, key, default=None):
        try:
            return _overlay_for(key)
        except Exception:
            return default

    def __getitem__(self, key):
        return _overlay_for(key)


MARS_OVERLAYS = _OverlayLookup()


def inject_overlay(svg: str, overlay: str) -> str:
    """Insert overlay fragment immediately before `</svg>`."""
    return svg.replace("</svg>", overlay + "</svg>", 1)


# ─────────────────────────── render utility ────────────────────────────────

def render_svg_to_png(svg: str, output_width: int = 480, height: int = 320) -> Image.Image:
    png = cairosvg.svg2png(bytestring=svg.encode(), output_width=output_width)
    img = Image.open(io.BytesIO(png)).convert("RGBA")
    if img.size != (output_width, height):
        img = img.resize((output_width, height), Image.LANCZOS)
    # Composite onto a transparent canvas (templates already use full canvas).
    return img
