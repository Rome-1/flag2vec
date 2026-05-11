"""SVG templates for the 15 vex categories.

Exact functional copies of the templates in `scripts/11_average_flags.py`,
factored into an importable module so Phase 4 (`17_mars_generate.py`) can
reuse them without import-by-numeric-filename gymnastics.

If you change the templates here, mirror the change in 11_average_flags.py
(or refactor that script to import from here).
"""
from __future__ import annotations


PALETTE_BUCKETS = {
    "red":         (200,  30,  35),
    "dark_red":    (140,  20,  25),
    "blue":        ( 40,  70, 170),
    "dark_blue":   ( 15,  35, 100),
    "light_blue":  ( 90, 150, 220),
    "green":       ( 30, 130,  60),
    "dark_green":  ( 20,  85,  35),
    "yellow":      (245, 200,  40),
    "orange":      (240, 130,  40),
    "white":       (245, 245, 245),
    "black":       ( 20,  20,  20),
    "brown":       (120,  70,  35),
}


def hex_color(name: str) -> str:
    r, g, b = PALETTE_BUCKETS[name]
    return f"#{r:02x}{g:02x}{b:02x}"


def _svg_wrap(inner: str, w: int = 480, h: int = 320) -> str:
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="0 0 {w} {h}" width="{w}" height="{h}">{inner}</svg>'
    )


def template_horizontal_tricolor(palette: list[str]) -> str:
    cs = (palette + ["red", "white", "blue"])[:3]
    h = 320 // 3
    inner = (
        f'<rect x="0" y="0" width="480" height="{h}" fill="{hex_color(cs[0])}"/>'
        f'<rect x="0" y="{h}" width="480" height="{h}" fill="{hex_color(cs[1])}"/>'
        f'<rect x="0" y="{2*h}" width="480" height="{320-2*h}" fill="{hex_color(cs[2])}"/>'
    )
    return _svg_wrap(inner)


def template_vertical_tricolor(palette: list[str]) -> str:
    cs = (palette + ["blue", "white", "red"])[:3]
    w = 480 // 3
    inner = (
        f'<rect x="0" y="0" width="{w}" height="320" fill="{hex_color(cs[0])}"/>'
        f'<rect x="{w}" y="0" width="{w}" height="320" fill="{hex_color(cs[1])}"/>'
        f'<rect x="{2*w}" y="0" width="{480-2*w}" height="320" fill="{hex_color(cs[2])}"/>'
    )
    return _svg_wrap(inner)


def template_nordic_cross(palette: list[str]) -> str:
    field = palette[0] if palette else "blue"
    cross = palette[1] if len(palette) > 1 else "white"
    inner = (
        f'<rect width="480" height="320" fill="{hex_color(field)}"/>'
        f'<rect x="0" y="130" width="480" height="60" fill="{hex_color(cross)}"/>'
        f'<rect x="140" y="0" width="60" height="320" fill="{hex_color(cross)}"/>'
    )
    return _svg_wrap(inner)


def template_pan_arab(palette: list[str]) -> str:
    inner = (
        '<rect x="0" y="0" width="480" height="106" fill="#cf142b"/>'
        '<rect x="0" y="106" width="480" height="108" fill="#ffffff"/>'
        '<rect x="0" y="214" width="480" height="106" fill="#000000"/>'
        '<polygon points="0,0 0,320 200,160" fill="#007a3d"/>'
    )
    return _svg_wrap(inner)


def template_british_ensign(palette: list[str]) -> str:
    field = palette[0] if palette and palette[0] in ("dark_blue", "blue", "red") else "dark_blue"
    inner = (
        f'<rect width="480" height="320" fill="{hex_color(field)}"/>'
        '<rect x="0" y="0" width="240" height="160" fill="#012169"/>'
        '<rect x="100" y="0" width="40" height="160" fill="#ffffff"/>'
        '<rect x="0" y="60" width="240" height="40" fill="#ffffff"/>'
        '<rect x="110" y="0" width="20" height="160" fill="#c8102e"/>'
        '<rect x="0" y="70" width="240" height="20" fill="#c8102e"/>'
        '<line x1="0" y1="0" x2="240" y2="160" stroke="#ffffff" stroke-width="20"/>'
        '<line x1="240" y1="0" x2="0" y2="160" stroke="#ffffff" stroke-width="20"/>'
        '<line x1="0" y1="0" x2="240" y2="160" stroke="#c8102e" stroke-width="8"/>'
        '<line x1="240" y1="0" x2="0" y2="160" stroke="#c8102e" stroke-width="8"/>'
    )
    return _svg_wrap(inner)


def template_communist_red(palette: list[str]) -> str:
    inner = (
        '<rect width="480" height="320" fill="#cf142b"/>'
        '<polygon points="100,40 115,90 165,90 125,120 140,170 100,140 60,170 75,120 35,90 85,90"'
        ' fill="#ffd700"/>'
    )
    return _svg_wrap(inner)


def template_star_crescent(palette: list[str]) -> str:
    field = palette[0] if palette else "green"
    if field not in ("red", "green", "dark_blue", "blue"):
        field = "green"
    inner = (
        f'<rect width="480" height="320" fill="{hex_color(field)}"/>'
        '<circle cx="240" cy="160" r="80" fill="#ffffff"/>'
        f'<circle cx="260" cy="160" r="68" fill="{hex_color(field)}"/>'
        '<polygon points="290,135 300,160 327,160 305,177 313,205 290,188 267,205 275,177 253,160 280,160"'
        ' fill="#ffffff"/>'
    )
    return _svg_wrap(inner)


def template_solid_emblem(palette: list[str]) -> str:
    field = palette[0] if palette else "red"
    inner = (
        f'<rect width="480" height="320" fill="{hex_color(field)}"/>'
        '<circle cx="240" cy="160" r="70" fill="#ffffff"/>'
    )
    return _svg_wrap(inner)


def template_pan_african(palette: list[str]) -> str:
    inner = (
        '<rect x="0" y="0" width="480" height="106" fill="#078930"/>'
        '<rect x="0" y="106" width="480" height="108" fill="#fcdd09"/>'
        '<rect x="0" y="214" width="480" height="106" fill="#da121a"/>'
        '<polygon points="240,120 255,160 295,160 263,185 275,225 240,200 205,225 217,185 185,160 225,160"'
        ' fill="#000000"/>'
    )
    return _svg_wrap(inner)


def template_stars_stripes(palette: list[str]) -> str:
    stripes = "".join(
        f'<rect x="0" y="{i*22}" width="480" height="22" fill="{"#bf0a30" if i%2==0 else "#ffffff"}"/>'
        for i in range(15)
    )
    inner = (
        stripes
        + '<rect x="0" y="0" width="190" height="155" fill="#002868"/>'
    )
    return _svg_wrap(inner)


def template_default_horizontal(palette: list[str]) -> str:
    return template_horizontal_tricolor(palette)


CATEGORY_TEMPLATES = {
    "horizontal_tricolor": template_horizontal_tricolor,
    "vertical_tricolor":   template_vertical_tricolor,
    "nordic_cross":        template_nordic_cross,
    "pan_arab":            template_pan_arab,
    "pan_african":         template_pan_african,
    "british_ensign":      template_british_ensign,
    "communist_red":       template_communist_red,
    "star_crescent":       template_star_crescent,
    "solid_emblem":        template_solid_emblem,
    "stars_stripes":       template_stars_stripes,
    "pan_slavic":          lambda p: template_horizontal_tricolor(["white", "blue", "red"]),
    "latin_charge":        lambda p: template_horizontal_tricolor(p or ["blue", "white", "blue"]),
    "saltire":             lambda p: template_horizontal_tricolor(p or ["green", "red", "white"]),
    "heraldic":            lambda p: template_solid_emblem(p or ["red"]),
    "unique":              template_default_horizontal,
}
