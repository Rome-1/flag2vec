"""Rasterize sovereign flag SVGs to uniform 480x320 transparent PNGs."""
from __future__ import annotations

import csv
import io
from pathlib import Path

import cairosvg
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
SVG_DIR = ROOT / "data" / "raw_svg"
OUT_DIR = ROOT / "data" / "png"
CSV_PATH = ROOT / "data" / "sovereign_flags.csv"

CANVAS = (480, 320)  # 3:2 (width:height) == 2:3 (height:width), the most common flag ratio


def rasterize(iso2: str) -> Image.Image:
    svg_path = SVG_DIR / f"{iso2}.svg"
    if not svg_path.exists():
        raise FileNotFoundError(svg_path)
    png_bytes = cairosvg.svg2png(
        url=str(svg_path),
        output_width=CANVAS[0],
    )
    img = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
    if img.height > CANVAS[1]:
        new_w = int(round(img.width * CANVAS[1] / img.height))
        img = img.resize((new_w, CANVAS[1]), Image.LANCZOS)
    canvas = Image.new("RGBA", CANVAS, (0, 0, 0, 0))
    x = (CANVAS[0] - img.width) // 2
    y = (CANVAS[1] - img.height) // 2
    canvas.paste(img, (x, y), img)
    return canvas


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with CSV_PATH.open() as f:
        rows = list(csv.DictReader(f))
    failed = []
    for row in rows:
        iso2 = row["iso2"]
        try:
            img = rasterize(iso2)
            img.save(OUT_DIR / f"{iso2}.png", optimize=True)
        except Exception as e:
            failed.append((iso2, str(e)))
    print(f"rasterized {len(rows) - len(failed)}/{len(rows)} flags -> {OUT_DIR}")
    if failed:
        print("FAILURES:")
        for iso2, err in failed:
            print(f"  {iso2}: {err}")


if __name__ == "__main__":
    main()
