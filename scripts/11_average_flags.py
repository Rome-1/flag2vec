"""Average flag analysis — multiple approaches.

The naive pixel-mean of a category produces a blurry image that is OOD for
DINOv2. We try four approaches to a more meaningful "average":

  1. pixel mean         — naive average of pixel values
  2. centroid neighbor  — closest REAL flag to the category's mean DINOv2 vector
  3. K-nearest mosaic   — top 5 flags closest to the centroid (small grid)
  4. procedural         — rule-based SVG composition using observed marginals

Then compares each to the UN flag in pixel + DINOv2 spaces.
"""
from __future__ import annotations

import io
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cairosvg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import timm
from PIL import Image, ImageOps
from sklearn.preprocessing import normalize
from timm.data import create_transform, resolve_data_config

from flag2vec.style import (
    BG, CATEGORY_COLORS, CSV_PATH, EMB_DIR, NICE_NAME, OUT_DIR, PNG_DIR,
    SUBTLE, TEXT, configure_typography, load_thumb,
)

OUT_AVG = OUT_DIR / "averages"
OUT_ANALYSIS = OUT_DIR / "analysis"

CANVAS = (480, 320)


# ─────────────────────────── DINOv2 helpers ────────────────────────────────

def _dinov2_model():
    model = timm.create_model("vit_small_patch14_dinov2.lvd142m",
                              pretrained=True, num_classes=0).eval()
    cfg = resolve_data_config({}, model=model)
    cfg["input_size"] = (3, 518, 518)
    transform = create_transform(**cfg, is_training=False)
    return model, transform


def _embed_image(model, transform, img: Image.Image) -> np.ndarray:
    img = img.convert("RGBA")
    side = 518
    scale = min(side / img.width, side / img.height)
    new_w = max(1, int(round(img.width * scale)))
    new_h = max(1, int(round(img.height * scale)))
    resized = img.resize((new_w, new_h), Image.LANCZOS)
    canvas = Image.new("RGB", (side, side), (255, 255, 255))
    canvas.paste(resized, ((side - new_w) // 2, (side - new_h) // 2), resized)
    t = transform(canvas).unsqueeze(0)
    with torch.inference_mode():
        feats = model.forward_features(t)
        if isinstance(feats, dict):
            cls = feats.get("x_norm_clstoken", feats.get("cls_token"))
        else:
            cls = feats[:, 0] if feats.ndim == 3 else feats
    return cls.squeeze(0).cpu().numpy().astype(np.float32)


# ─────────────────────────── data loading ──────────────────────────────────

def load_data():
    X = np.load(EMB_DIR / "dinov2_vits14.npy")
    iso2 = (EMB_DIR / "iso2_order.txt").read_text().strip().splitlines()
    meta = pd.read_csv(CSV_PATH).set_index("iso2").loc[iso2].reset_index()
    Xn = normalize(X, norm="l2")
    return X, Xn, iso2, meta


def load_png(iso2: str) -> Image.Image:
    return Image.open(PNG_DIR / f"{iso2}.png").convert("RGBA")


# ─────────────────────────── approach 1: pixel mean ────────────────────────

def pixel_mean(iso2_list: list[str]) -> Image.Image:
    arrs = []
    for iso2 in iso2_list:
        img = load_png(iso2)
        rgb = Image.new("RGB", img.size, (255, 255, 255))
        rgb.paste(img, mask=img.split()[-1])
        arrs.append(np.asarray(rgb, dtype=np.float32))
    mean = np.mean(arrs, axis=0).clip(0, 255).astype(np.uint8)
    return Image.fromarray(mean)


# ─────────────────────────── approach 4: procedural ────────────────────────
# Color extraction: dominant colors per flag

PALETTE_BUCKETS = {
    # name        : RGB anchor                        (used for nearest-anchor labeling)
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


def dominant_colors(iso2: str, n: int = 5, min_frac: float = 0.04) -> list[tuple[tuple[int,int,int], float]]:
    """Return [(rgb, frac), ...] for colors covering ≥min_frac of pixels."""
    img = load_png(iso2)
    rgb = np.asarray(img.convert("RGB")).reshape(-1, 3)
    alpha = np.asarray(img.split()[-1]).ravel()
    rgb = rgb[alpha > 200]
    # quantize to 16-step cube and count
    q = (rgb // 16) * 16 + 8
    keys = q[:, 0].astype(int) * 65536 + q[:, 1].astype(int) * 256 + q[:, 2].astype(int)
    counts = Counter(keys.tolist())
    total = sum(counts.values())
    out = []
    for k, c in counts.most_common(n):
        frac = c / total
        if frac < min_frac:
            break
        r = (k >> 16) & 0xff; g = (k >> 8) & 0xff; b = k & 0xff
        out.append(((r, g, b), frac))
    return out


def color_to_anchor(rgb: tuple[int,int,int]) -> str:
    r, g, b = rgb
    best = None; best_d = float("inf")
    for name, anchor in PALETTE_BUCKETS.items():
        d = (r-anchor[0])**2 + (g-anchor[1])**2 + (b-anchor[2])**2
        if d < best_d:
            best_d = d; best = name
    return best


def category_color_distribution(iso2_list: list[str]) -> list[tuple[str, float]]:
    """Aggregate dominant colors across a category, return sorted by total weight."""
    weights = Counter()
    for iso2 in iso2_list:
        for rgb, frac in dominant_colors(iso2, n=4):
            weights[color_to_anchor(rgb)] += frac
    total = sum(weights.values()) or 1
    return [(k, v / total) for k, v in weights.most_common()]


def hex_color(name: str) -> str:
    r, g, b = PALETTE_BUCKETS[name]
    return f"#{r:02x}{g:02x}{b:02x}"


# Templates — each is a function(palette: list[str]) -> svg string
def _svg_wrap(inner: str, w: int = 480, h: int = 320) -> str:
    return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}">{inner}</svg>'


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
    # field is dominant non-cross color; cross uses the second dominant
    field = palette[0] if palette else "blue"
    cross = palette[1] if len(palette) > 1 else "white"
    inner = (
        f'<rect width="480" height="320" fill="{hex_color(field)}"/>'
        # horizontal arm
        f'<rect x="0" y="130" width="480" height="60" fill="{hex_color(cross)}"/>'
        # vertical arm offset to hoist (Nordic cross convention)
        f'<rect x="140" y="0" width="60" height="320" fill="{hex_color(cross)}"/>'
    )
    return _svg_wrap(inner)


def template_pan_arab(palette: list[str]) -> str:
    # red / white / black horizontal stripes, with green triangle on hoist
    inner = (
        f'<rect x="0" y="0" width="480" height="106" fill="#cf142b"/>'
        f'<rect x="0" y="106" width="480" height="108" fill="#ffffff"/>'
        f'<rect x="0" y="214" width="480" height="106" fill="#000000"/>'
        f'<polygon points="0,0 0,320 200,160" fill="#007a3d"/>'
    )
    return _svg_wrap(inner)


def template_british_ensign(palette: list[str]) -> str:
    # blue field with Union Jack canton
    field = palette[0] if palette and palette[0] in ("dark_blue", "blue", "red") else "dark_blue"
    inner = (
        f'<rect width="480" height="320" fill="{hex_color(field)}"/>'
        # canton (Union Jack simplified)
        f'<rect x="0" y="0" width="240" height="160" fill="#012169"/>'
        f'<rect x="100" y="0" width="40" height="160" fill="#ffffff"/>'
        f'<rect x="0" y="60" width="240" height="40" fill="#ffffff"/>'
        f'<rect x="110" y="0" width="20" height="160" fill="#c8102e"/>'
        f'<rect x="0" y="70" width="240" height="20" fill="#c8102e"/>'
        f'<line x1="0" y1="0" x2="240" y2="160" stroke="#ffffff" stroke-width="20"/>'
        f'<line x1="240" y1="0" x2="0" y2="160" stroke="#ffffff" stroke-width="20"/>'
        f'<line x1="0" y1="0" x2="240" y2="160" stroke="#c8102e" stroke-width="8"/>'
        f'<line x1="240" y1="0" x2="0" y2="160" stroke="#c8102e" stroke-width="8"/>'
    )
    return _svg_wrap(inner)


def template_communist_red(palette: list[str]) -> str:
    # red field with yellow star top-left
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
    # solid field with white disc emblem
    field = palette[0] if palette else "red"
    inner = (
        f'<rect width="480" height="320" fill="{hex_color(field)}"/>'
        '<circle cx="240" cy="160" r="70" fill="#ffffff"/>'
    )
    return _svg_wrap(inner)


def template_pan_african(palette: list[str]) -> str:
    # green / yellow / red horizontal (pan-African) with central black star
    inner = (
        '<rect x="0" y="0" width="480" height="106" fill="#078930"/>'
        '<rect x="0" y="106" width="480" height="108" fill="#fcdd09"/>'
        '<rect x="0" y="214" width="480" height="106" fill="#da121a"/>'
        '<polygon points="240,120 255,160 295,160 263,185 275,225 240,200 205,225 217,185 185,160 225,160"'
        ' fill="#000000"/>'
    )
    return _svg_wrap(inner)


def template_stars_stripes(palette: list[str]) -> str:
    # red and white horizontal stripes with blue canton
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
    # The diffuse / heterogeneous categories fall back to a horizontal tri:
    "pan_slavic":          lambda p: template_horizontal_tricolor(["white", "blue", "red"]),
    "latin_charge":        lambda p: template_horizontal_tricolor(p),
    "saltire":             lambda p: template_horizontal_tricolor(p),
    "heraldic":            lambda p: template_solid_emblem(p),
    "unique":              template_default_horizontal,
}


def render_svg(svg: str) -> Image.Image:
    png = cairosvg.svg2png(bytestring=svg.encode(), output_width=CANVAS[0])
    img = Image.open(io.BytesIO(png)).convert("RGB")
    if img.size != CANVAS:
        img = img.resize(CANVAS, Image.LANCZOS)
    return img


# ─────────────────────────── distance metrics ──────────────────────────────

def img_to_array_rgb(img: Image.Image) -> np.ndarray:
    if img.mode != "RGB":
        bg = Image.new("RGB", img.size, (255, 255, 255))
        if img.mode == "RGBA":
            bg.paste(img, mask=img.split()[-1])
        else:
            bg.paste(img.convert("RGBA"))
        img = bg
    if img.size != CANVAS:
        img = img.resize(CANVAS, Image.LANCZOS)
    return np.asarray(img, dtype=np.float32) / 255.0


def pixel_distance(a: Image.Image, b: Image.Image) -> float:
    A = img_to_array_rgb(a); B = img_to_array_rgb(b)
    return float(np.sqrt(np.mean((A - B) ** 2)))


# ─────────────────────────── main analysis ─────────────────────────────────

def main():
    configure_typography()
    OUT_AVG.mkdir(parents=True, exist_ok=True)
    OUT_ANALYSIS.mkdir(parents=True, exist_ok=True)

    X, Xn, iso2, meta = load_data()
    print(f"loaded {X.shape}")

    # Embed UN flag
    print("embedding UN flag...")
    model, transform = _dinov2_model()
    un_img = load_png("un")
    un_emb = _embed_image(model, transform, un_img)
    un_emb_n = un_emb / (np.linalg.norm(un_emb) + 1e-9)
    print(f"UN embedding: {un_emb.shape}")

    cats = sorted(meta["vex_category"].unique(),
                  key=lambda c: -(meta["vex_category"] == c).sum())

    rows = []  # [{category, n, pixel_avg, centroid, mosaic, procedural, dist_*}]

    # global "all" row
    all_iso = meta["iso2"].tolist()
    pixel_avg_global = pixel_mean(all_iso)
    pixel_avg_global.save(OUT_AVG / "global_pixel_mean.png")
    rows.append({
        "category": "all",
        "n": len(all_iso),
        "pixel_avg": pixel_avg_global,
    })

    # per-category
    for cat in cats:
        sub = meta[meta["vex_category"] == cat]
        if len(sub) < 2:
            continue
        cat_iso = sub["iso2"].tolist()

        # 1. pixel mean
        pix = pixel_mean(cat_iso)
        pix.save(OUT_AVG / f"{cat}_pixel_mean.png")

        # 2. centroid neighbor (closest real flag to mean DINOv2 vec)
        idx = meta["vex_category"] == cat
        cent = Xn[idx].mean(axis=0)
        cent /= np.linalg.norm(cent) + 1e-9
        d_to_cent = 1 - Xn[idx] @ cent
        i_cent = np.argmin(d_to_cent)
        cent_iso = sub.iloc[i_cent]["iso2"]

        # 3. K-nearest mosaic (closest 5 real flags to centroid)
        k = min(5, idx.sum())
        order = np.argsort(d_to_cent)[:k]
        nearest_isos = sub.iloc[order]["iso2"].tolist()

        # 4. procedural
        palette = [name for name, _ in category_color_distribution(cat_iso)][:5]
        template = CATEGORY_TEMPLATES.get(cat, template_default_horizontal)
        proc = render_svg(template(palette))
        proc.save(OUT_AVG / f"{cat}_procedural.png")

        # Compute distances
        pix_to_un_pix = pixel_distance(pix, un_img)
        proc_to_un_pix = pixel_distance(proc, un_img)
        cent_to_un_pix = pixel_distance(load_png(cent_iso), un_img)
        # DINOv2 distances
        cent_emb_n = Xn[idx][i_cent]
        cent_to_un_emb = float(1 - np.dot(cent_emb_n, un_emb_n))
        # avg DINOv2 of category (vector-mean) to UN
        avg_to_un_emb = float(1 - np.dot(cent, un_emb_n))
        # procedural in DINOv2
        proc_emb = _embed_image(model, transform, proc)
        proc_emb_n = proc_emb / (np.linalg.norm(proc_emb) + 1e-9)
        proc_to_un_emb = float(1 - np.dot(proc_emb_n, un_emb_n))
        # pixel-mean in DINOv2 (the OOD image)
        pix_emb = _embed_image(model, transform, pix)
        pix_emb_n = pix_emb / (np.linalg.norm(pix_emb) + 1e-9)
        pix_to_un_emb = float(1 - np.dot(pix_emb_n, un_emb_n))

        rows.append({
            "category": cat,
            "n": int(idx.sum()),
            "centroid_iso": cent_iso,
            "centroid_name": sub.iloc[i_cent]["name"],
            "nearest_isos": nearest_isos,
            "palette": palette,
            "pixel_avg_path": str(OUT_AVG / f"{cat}_pixel_mean.png"),
            "procedural_path": str(OUT_AVG / f"{cat}_procedural.png"),
            "pix_to_un_pix": pix_to_un_pix,
            "proc_to_un_pix": proc_to_un_pix,
            "cent_to_un_pix": cent_to_un_pix,
            "avg_emb_to_un": avg_to_un_emb,
            "cent_emb_to_un": cent_to_un_emb,
            "proc_emb_to_un": proc_to_un_emb,
            "pix_emb_to_un": pix_to_un_emb,
        })

    # global procedural — most common pattern overall
    print("global procedural...")
    overall_palette = [name for name, _ in category_color_distribution(all_iso)][:5]
    # Most common pattern is horizontal tricolor (n=26) — use that.
    proc_global = render_svg(template_horizontal_tricolor(overall_palette))
    proc_global.save(OUT_AVG / "global_procedural.png")
    pix_emb_g = _embed_image(model, transform, pixel_avg_global)
    proc_emb_g = _embed_image(model, transform, proc_global)
    rows[0].update({
        "palette": overall_palette,
        "procedural_path": str(OUT_AVG / "global_procedural.png"),
        "pix_to_un_pix": pixel_distance(pixel_avg_global, un_img),
        "proc_to_un_pix": pixel_distance(proc_global, un_img),
        "pix_emb_to_un": float(1 - np.dot(pix_emb_g / (np.linalg.norm(pix_emb_g)+1e-9), un_emb_n)),
        "proc_emb_to_un": float(1 - np.dot(proc_emb_g / (np.linalg.norm(proc_emb_g)+1e-9), un_emb_n)),
        "avg_emb_to_un": float(1 - np.dot(Xn.mean(axis=0) / (np.linalg.norm(Xn.mean(axis=0))+1e-9), un_emb_n)),
    })

    df = pd.DataFrame(rows)
    df.to_csv(OUT_AVG / "summary.csv", index=False)
    print(df[["category", "n", "pix_emb_to_un", "proc_emb_to_un", "cent_emb_to_un", "avg_emb_to_un"]].to_string())

    # ───── Figure 1: per-category 4-up grid (pixel | centroid | mosaic | procedural)
    cat_rows = [r for r in rows if r["category"] != "all"]
    n_rows = len(cat_rows)
    fig = plt.figure(figsize=(20, 2.6 * n_rows + 1.6))
    fig.patch.set_facecolor(BG)
    fig.suptitle("average flag per category — four ways",
                 fontsize=24, fontweight="medium", color=TEXT,
                 x=0.025, y=0.985, ha="left")
    fig.text(0.025, 0.965,
             "Comparing approaches to a category 'average'.  "
             "Pixel mean is OOD (blurry).  Embedding centroid is the closest real flag.  "
             "Mosaic shows the top-5 closest real flags.  Procedural is rule-based composition "
             "from observed marginals.  Distances are cosine in DINOv2 space, vs the UN flag.",
             fontsize=11, color=SUBTLE, ha="left")

    cols = 4
    for i, r in enumerate(cat_rows):
        cat = r["category"]
        nice = NICE_NAME.get(cat, cat)
        color = CATEGORY_COLORS.get(cat, "#888888")

        def add_cell(col, img_or_isos, title, dist_label):
            ax = fig.add_subplot(n_rows, cols, i * cols + col + 1)
            ax.set_facecolor(BG); ax.axis("off")
            if isinstance(img_or_isos, list):
                # mosaic
                mosaic = []
                for iso in img_or_isos:
                    mosaic.append(load_thumb(iso, height_px=80, border_px=1,
                                             border_color="#CCCCCC"))
                # tile horizontally
                widths = [a.shape[1] for a in mosaic]
                total_w = sum(widths) + 4 * (len(mosaic) - 1)
                tile = np.ones((mosaic[0].shape[0], total_w, 4), dtype=np.uint8) * 247
                tile[..., 3] = 255
                x = 0
                for a in mosaic:
                    tile[:, x:x + a.shape[1]] = a
                    x += a.shape[1] + 4
                ax.imshow(tile)
            else:
                arr = np.asarray(img_or_isos.convert("RGBA"))
                ax.imshow(arr)
            ax.set_title(title, fontsize=10, color=TEXT, loc="left", pad=3)
            ax.text(0.0, -0.06, dist_label, transform=ax.transAxes,
                    fontsize=8.5, color=SUBTLE, ha="left", va="top")
            if col == 0:
                ax.text(-0.04, 0.5,
                        f"{nice}\nn={r['n']}",
                        transform=ax.transAxes, fontsize=11, fontweight="medium",
                        color=color, ha="right", va="center", linespacing=1.4)

        add_cell(0, Image.open(r["pixel_avg_path"]), "pixel mean",
                 f"DINOv2→UN  {r['pix_emb_to_un']:.2f}")
        add_cell(1, load_png(r["centroid_iso"]), f"centroid: {r['centroid_name']}",
                 f"DINOv2→UN  {r['cent_emb_to_un']:.2f}")
        add_cell(2, r["nearest_isos"], "5-nearest real flags",
                 "")
        add_cell(3, Image.open(r["procedural_path"]),
                 f"procedural ({', '.join(r['palette'][:4])})",
                 f"DINOv2→UN  {r['proc_emb_to_un']:.2f}")

    plt.subplots_adjust(left=0.06, right=0.99, top=0.95, bottom=0.01,
                        hspace=0.45, wspace=0.05)
    out = OUT_ANALYSIS / "average_flag_per_category.png"
    fig.savefig(out, dpi=160, facecolor=BG)
    plt.close(fig)
    print(f"wrote {out}")

    # ───── Figure 2: distance to UN bar chart (sorted)
    cdf = pd.DataFrame(cat_rows)
    cdf = cdf.sort_values("cent_emb_to_un", ascending=True).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(11, 9))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(colors=SUBTLE, length=0)

    y = np.arange(len(cdf))
    h = 0.18
    cent_d = cdf["cent_emb_to_un"].to_numpy()
    proc_d = cdf["proc_emb_to_un"].to_numpy()
    pix_d  = cdf["pix_emb_to_un"].to_numpy()
    avg_d  = cdf["avg_emb_to_un"].to_numpy()

    ax.barh(y - 1.5*h, cent_d, h, color=CATEGORY_COLORS["pan_arab"],
            label="centroid (real flag)", edgecolor=BG, linewidth=0.5)
    ax.barh(y - 0.5*h, proc_d, h, color=CATEGORY_COLORS["nordic_cross"],
            label="procedural composition", edgecolor=BG, linewidth=0.5)
    ax.barh(y + 0.5*h, pix_d, h, color=CATEGORY_COLORS["star_crescent"],
            label="pixel mean (OOD)", edgecolor=BG, linewidth=0.5)
    ax.barh(y + 1.5*h, avg_d, h, color=SUBTLE, alpha=0.6,
            label="latent mean vector", edgecolor=BG, linewidth=0.5)

    ax.set_yticks(y)
    ax.set_yticklabels([f"{NICE_NAME.get(c, c)}  (n={int(n)})"
                        for c, n in zip(cdf["category"], cdf["n"])],
                       fontsize=10)
    ax.set_xlabel("cosine distance to UN flag in DINOv2 space",
                  color=SUBTLE, fontsize=10.5)
    ax.legend(frameon=False, fontsize=10, labelcolor=TEXT, loc="lower right")

    fig.suptitle("category averages → UN flag",
                 fontsize=22, fontweight="medium", color=TEXT,
                 x=0.04, y=0.97, ha="left")
    fig.text(0.04, 0.92,
             "Four ways to compute a category's 'average flag', then measure distance to "
             "the United Nations flag in DINOv2 space.  Sorted by centroid (closest real "
             "flag) distance.  The UN flag is essentially solid_emblem — light blue field "
             "with white emblem — so we expect that family to dominate the top.",
             fontsize=11, color=SUBTLE, ha="left")
    plt.subplots_adjust(left=0.20, right=0.95, top=0.85, bottom=0.07)
    out = OUT_ANALYSIS / "average_distance_to_un.png"
    fig.savefig(out, dpi=200, facecolor=BG)
    plt.close(fig)
    print(f"wrote {out}")

    # ───── Figure 3: the procedural "average sovereign flag" gallery
    # Show all 15 procedural average flags + global, big, with UN at top.
    n_panels = 1 + len(cat_rows)
    cols = 4
    rows_n = int(np.ceil(n_panels / cols))
    fig = plt.figure(figsize=(3.6 * cols, 2.8 * rows_n + 1.4))
    fig.patch.set_facecolor(BG)
    fig.suptitle("procedural 'average flag' per category",
                 fontsize=22, fontweight="medium", color=TEXT,
                 x=0.04, y=0.965, ha="left")
    fig.text(0.04, 0.93,
             "Each flag generated by a rule-based template (field division + canonical colors) "
             "rather than averaging pixels — keeping the result on the manifold of real flag designs.",
             fontsize=11, color=SUBTLE, ha="left")
    panels = [(rows[0], "all sovereign", "global_procedural.png")] + [
        (r, NICE_NAME.get(r["category"], r["category"]), f"{r['category']}_procedural.png")
        for r in cat_rows
    ]
    for k, (r, label, path) in enumerate(panels):
        ax = fig.add_subplot(rows_n, cols, k + 1)
        ax.set_facecolor(BG); ax.axis("off")
        img = Image.open(OUT_AVG / path)
        ax.imshow(np.asarray(img))
        ax.set_title(label, fontsize=12, color=TEXT, fontweight="medium",
                     loc="left", pad=4)
        if "proc_emb_to_un" in r:
            ax.text(0.0, -0.05,
                    f"DINOv2 distance to UN flag: {r['proc_emb_to_un']:.2f}  "
                    f"·  palette: {', '.join(r.get('palette', [])[:3])}",
                    transform=ax.transAxes, fontsize=8.5, color=SUBTLE,
                    ha="left", va="top")

    plt.subplots_adjust(left=0.025, right=0.975, top=0.88, bottom=0.03,
                        hspace=0.55, wspace=0.1)
    out = OUT_ANALYSIS / "procedural_average_flags.png"
    fig.savefig(out, dpi=180, facecolor=BG)
    plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
