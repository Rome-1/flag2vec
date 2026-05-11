"""Phase 4 — Mars-terraformed flag generation.

For each Martian region in `data/mars_regions.csv`:

  1. Pick the inherited Earth template from `scripts/11_average_flags.py`'s
     CATEGORY_TEMPLATES (e.g., template_pan_arab for Arabia Terra).
  2. Render the Earth template SVG, then transform its palette via
     MARS_PALETTE: blue→rust-orange, white→pale-cream, green→sage,
     red→deeper-iron-red, yellow→sulfur-yellow.  Each flag also receives
     at least one Mars-specific accent color.
  3. Composite a Mars-specific overlay symbol per region (Olympus Mons
     silhouette, polar cap, Phobos+Deimos pair, dust-storm spiral,
     Valles Marineris canyon line, etc.).  Overlay kept ≤25% of canvas.
  4. Render to `out/mars/flags/<region_id>.png` (480x320 PNG).
"""
from __future__ import annotations

import io
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cairosvg
import pandas as pd
from PIL import Image

# Re-use the Earth template machinery.
from scripts._mars_lib import (  # noqa: E402  (script-relative import)
    CATEGORY_TEMPLATES,
    EARTH_TO_MARS_HEX,
    MARS_ACCENTS,
    MARS_OVERLAYS,
    PNG_OUT_DIR,
    SVG_OUT_DIR,
    apply_mars_palette,
    inject_overlay,
    render_svg_to_png,
)

ROOT = Path(__file__).resolve().parent.parent
MARS_CSV = ROOT / "data" / "mars_regions.csv"


def generate_one(row: dict) -> tuple[str, Image.Image]:
    region_id = row["region_id"]
    tradition = row["inherited_tradition"]
    feature = row["feature_type"]

    template_fn = CATEGORY_TEMPLATES.get(tradition)
    if template_fn is None:
        raise KeyError(
            f"unknown inherited tradition {tradition!r} for {region_id}"
        )

    # The Earth templates take a `palette` of color-name strings; pass the
    # tradition's canonical palette so we get the most distinctive form.
    earth_svg = template_fn([])
    mars_svg = apply_mars_palette(earth_svg, region_id=region_id)
    overlay = MARS_OVERLAYS.get(region_id) or MARS_OVERLAYS["_default"]
    final_svg = inject_overlay(mars_svg, overlay)
    img = render_svg_to_png(final_svg, output_width=480, height=320)
    return region_id, img


def main() -> None:
    PNG_OUT_DIR.mkdir(parents=True, exist_ok=True)
    SVG_OUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(MARS_CSV)
    print(f"generating {len(df)} Mars flags...")
    for _, row in df.iterrows():
        region_id, img = generate_one(row.to_dict())
        out_png = PNG_OUT_DIR / f"{region_id}.png"
        img.save(out_png, optimize=True)
        print(f"  wrote {out_png}")
    print("done.")


if __name__ == "__main__":
    main()
