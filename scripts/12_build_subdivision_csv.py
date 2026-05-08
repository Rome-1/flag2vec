"""Generate data/subdivision_flags.csv from vendored ISO 3166-2 SVGs.

Default category by country, with manual overrides for known distinctive flags.
"""
from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SVG_DIR = ROOT / "data" / "raw_svg_subdivisions"
OUT_CSV = ROOT / "data" / "subdivision_flags.csv"


# Country defaults — most subdivisions in a country share a convention.
COUNTRY_DEFAULT = {
    "US": "us_state_seal",     # blue/white field + state seal
    "JP": "japanese_geometric", # single geometric symbol on solid field
    "CH": "swiss_canton",       # heraldic miniature
    "DE": "heraldic",
    "AU": "british_ensign",
    "CA": "british_ensign",     # most are British-ensign style
    "BR": "unique",
    "GB": "heraldic",
}

# Manual overrides per ISO 3166-2 subcode. Anything not listed uses COUNTRY_DEFAULT.
OVERRIDES = {
    # US — distinctive state flags that escape the seal convention.
    "US-HI": "british_ensign",   # Union Jack canton
    "US-TX": "unique",
    "US-CA": "unique",           # bear flag
    "US-NM": "unique",           # red sun on yellow
    "US-AZ": "unique",           # rays + star
    "US-AK": "unique",           # big dipper
    "US-MD": "unique",           # Calvert/Crossland heraldic
    "US-TN": "unique",
    "US-CO": "unique",
    "US-SC": "unique",
    "US-OH": "unique",           # only non-rectangular US state flag
    "US-DC": "unique",           # bars and stars
    "US-MS": "unique",           # 2020 magnolia design
    "US-PR": "horizontal_tricolor",  # 5 stripes + triangle
    "US-MN": "unique",            # 2024 redesign

    # CA — provincial overrides
    "CA-QC": "unique",            # fleur-de-lis cross
    "CA-NL": "unique",            # Christopher Pratt design
    "CA-NS": "saltire",           # saltire
    "CA-NB": "british_ensign",    # ish — provincial banner
    "CA-PE": "british_ensign",    # provincial banner
    "CA-NU": "unique",
    "CA-NT": "unique",
    "CA-YT": "unique",

    # GB constituent
    "GB-ENG": "saltire",          # Cross of St George (technically a cross, not saltire — but our nordic_cross category is "Nordic"; treat as "heraldic" instead)
    "GB-SCT": "saltire",          # Cross of St Andrew (saltire)
    "GB-WLS": "unique",            # red dragon
    "GB-NIR": "heraldic",          # Ulster banner

    # Nordic-cross-bearing autonomous regions
    "AX-self": "nordic_cross",
    "FO-self": "nordic_cross",
    "GL-self": "solid_emblem",     # white field with red+white disc — emblem flag

    # German Länder with horizontal tricolors (no arms or simple)
    "DE-BE": "horizontal_tricolor",  # red/white/red
    "DE-BB": "horizontal_tricolor",  # red/white/red w/ eagle
    "DE-HB": "horizontal_tricolor",  # red/white horizontal stripes (Speckflagge)
    "DE-HH": "horizontal_tricolor",  # red/white/red
    "DE-RP": "horizontal_tricolor",  # similar
    "DE-SL": "horizontal_tricolor",
    "DE-TH": "horizontal_tricolor",
    "DE-MV": "horizontal_tricolor",
    "DE-SN": "horizontal_tricolor",
    "DE-ST": "horizontal_tricolor",

    # AU territories that aren't strict ensigns
    "AU-NT": "unique",            # ochre + black
    "AU-ACT": "unique",
    "AU-CT": "unique",            # synonym; safety

    # BR — these are visually distinctive
    "BR-AC": "unique",
    "BR-AL": "unique",
    "BR-AM": "unique",
    "BR-AP": "unique",
    "BR-BA": "horizontal_tricolor",   # red/white/blue with canton
    "BR-CE": "unique",
    "BR-DF": "unique",
    "BR-ES": "horizontal_tricolor",
    "BR-GO": "stars_stripes",          # blue/white horizontal w/ canton + stars
    "BR-MA": "stars_stripes",
    "BR-MG": "saltire",
    "BR-MS": "unique",
    "BR-MT": "unique",
    "BR-PA": "horizontal_tricolor",
    "BR-PB": "unique",
    "BR-PE": "unique",
    "BR-PI": "stars_stripes",
    "BR-PR": "horizontal_tricolor",
    "BR-RJ": "horizontal_tricolor",
    "BR-RN": "unique",
    "BR-RO": "horizontal_tricolor",
    "BR-RR": "horizontal_tricolor",
    "BR-RS": "horizontal_tricolor",
    "BR-SC": "horizontal_tricolor",
    "BR-SE": "stars_stripes",
    "BR-SP": "stars_stripes",
    "BR-TO": "horizontal_tricolor",
}


# Friendly names for ISO 3166-2 codes — shortcut: just use the code as name.
def nice_name(code: str) -> str:
    if code.endswith("-self"):
        c = code.split("-")[0]
        return {"AX": "Åland Islands", "FO": "Faroe Islands", "GL": "Greenland"}.get(c, c)
    return code


def main():
    rows = []
    for path in sorted(SVG_DIR.glob("*.svg")):
        code = path.stem
        country = code.split("-")[0]
        cat = OVERRIDES.get(code, COUNTRY_DEFAULT.get(country, "unique"))
        rows.append({
            "iso2": code,
            "name": nice_name(code),
            "vex_category": cat,
            "parent": country,
            "kind": "subdivision",
        })

    with OUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["iso2", "name", "vex_category",
                                                "parent", "kind"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows -> {OUT_CSV}")
    # Distribution
    from collections import Counter
    cat_counts = Counter(r["vex_category"] for r in rows)
    print("category distribution:")
    for c, n in cat_counts.most_common():
        print(f"  {c}: {n}")


if __name__ == "__main__":
    main()
