"""Render the hero figure: 3-panel PCA / t-SNE / PHATE with flag-as-mark."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image, ImageOps
from scipy.spatial import ConvexHull

ROOT = Path(__file__).resolve().parent.parent
PNG_DIR = ROOT / "data" / "png"
PROJ_DIR = ROOT / "data" / "projections"
OUT_DIR = ROOT / "out"

# Anthropic-ish palette: warm off-white background, restrained accent colors.
BG = "#F7F4EC"
TEXT = "#1A1A1A"
SUBTLE = "#6B6B6B"
GRID = "#E8E2D2"

# Category colors — chosen to be distinguishable on the warm BG and tied
# loosely to the flag tradition's own dominant hue when possible.
CATEGORY_COLORS = {
    "nordic_cross":       "#2E5DA5",
    "pan_african":        "#7E5A1F",
    "pan_arab":           "#A02C2C",
    "pan_slavic":         "#6E5C9C",
    "british_ensign":     "#1F3F6E",
    "vertical_tricolor":  "#3D7C5A",
    "horizontal_tricolor":"#9C8A5C",
    "star_crescent":      "#1F7A6B",
    "communist_red":      "#B23A3A",
    "latin_charge":       "#C0892F",
    "saltire":            "#5C7C46",
    "solid_emblem":       "#8C5A2E",
    "stars_stripes":      "#2C4F8E",
    "heraldic":           "#6E4A3A",
    "unique":             "#A0A0A0",
}

# Categories worth highlighting in legend & via colored borders.
HIGHLIGHT_ORDER = [
    "nordic_cross",
    "british_ensign",
    "pan_arab",
    "star_crescent",
    "pan_african",
    "vertical_tricolor",
    "communist_red",
    "stars_stripes",
    "latin_charge",
    "pan_slavic",
]

NICE_NAME = {
    "nordic_cross":       "Nordic cross",
    "pan_african":        "Pan-African",
    "pan_arab":           "Pan-Arab",
    "pan_slavic":         "Pan-Slavic",
    "british_ensign":     "British ensign",
    "vertical_tricolor":  "Vertical tricolor",
    "horizontal_tricolor":"Horizontal tricolor",
    "star_crescent":      "Star & crescent",
    "communist_red":      "Communist red",
    "latin_charge":       "Latin (charge)",
    "saltire":            "Saltire",
    "solid_emblem":       "Solid + emblem",
    "stars_stripes":      "Stars & stripes",
    "heraldic":           "Heraldic",
    "unique":             "Unique",
}

PANELS = [
    ("PCA",   "pca_x",   "pca_y",   "21% + 11% var"),
    ("t-SNE", "tsne_x",  "tsne_y",  "perplexity 20, cosine"),
    ("PHATE", "phate_x", "phate_y", "knn 10, decay 20"),
]


def load_thumb(iso2: str, height_px: int, border_color: str | None,
               border_px: int = 2) -> np.ndarray:
    img = Image.open(PNG_DIR / f"{iso2}.png").convert("RGBA")
    bbox = img.getbbox()
    if bbox is not None:
        img = img.crop(bbox)
    scale = height_px / img.height
    new_w = max(1, int(round(img.width * scale)))
    new_h = max(1, int(round(img.height * scale)))
    img = img.resize((new_w, new_h), Image.LANCZOS)
    if border_color:
        img = ImageOps.expand(img, border=border_px, fill=border_color)
    return np.asarray(img)


def add_soft_hull(ax, pts: np.ndarray, color: str,
                  alpha_fill=0.07, alpha_edge=0.25, expand=0.04, axis_span=None):
    if len(pts) < 3:
        return
    try:
        hull = ConvexHull(pts)
    except Exception:
        return
    poly = pts[hull.vertices]
    centroid = poly.mean(axis=0)
    poly = centroid + (poly - centroid) * (1.0 + expand)
    poly = np.vstack([poly, poly[:1]])
    ax.fill(poly[:, 0], poly[:, 1], color=color, alpha=alpha_fill,
            zorder=0, linewidth=0)
    ax.plot(poly[:, 0], poly[:, 1], color=color, alpha=alpha_edge,
            linewidth=0.7, zorder=0)


def category_compactness(pts: np.ndarray, all_pts: np.ndarray) -> float:
    """Mean pairwise distance within / mean pairwise distance across — lower is tighter."""
    if len(pts) < 2:
        return 0.0
    inner = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
    inner_mean = inner[np.triu_indices(len(pts), k=1)].mean()
    outer = np.linalg.norm(all_pts[:, None, :] - all_pts[None, :, :], axis=-1)
    outer_mean = outer[np.triu_indices(len(all_pts), k=1)].mean()
    return float(inner_mean / outer_mean) if outer_mean > 0 else 0.0


def plot_panel(ax, df: pd.DataFrame, xcol: str, ycol: str, title: str,
               subtitle: str, thumb_height: int = 26):
    ax.set_facecolor(BG)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])

    all_pts = df[[xcol, ycol]].to_numpy()
    # Soft hull only for the *tightest* highlighted clusters — those that
    # visually cluster in this projection. Keeps the figure honest: hulls
    # are evidence of clustering, not decoration.
    for cat in HIGHLIGHT_ORDER:
        sub = df[df["vex_category"] == cat]
        if len(sub) < 3:
            continue
        pts = sub[[xcol, ycol]].to_numpy()
        compact = category_compactness(pts, all_pts)
        if compact < 0.55:
            add_soft_hull(ax, pts, CATEGORY_COLORS[cat])

    xs = all_pts[:, 0]
    ys = all_pts[:, 1]
    pad_x = (xs.max() - xs.min()) * 0.07
    pad_y = (ys.max() - ys.min()) * 0.07
    ax.set_xlim(xs.min() - pad_x, xs.max() + pad_x)
    ax.set_ylim(ys.min() - pad_y, ys.max() + pad_y)

    for _, row in df.iterrows():
        cat = row["vex_category"]
        border = CATEGORY_COLORS.get(cat) if cat in HIGHLIGHT_ORDER else None
        thumb = load_thumb(row["iso2"], thumb_height, border)
        oi = OffsetImage(thumb, zoom=1.0, interpolation="lanczos")
        ab = AnnotationBbox(
            oi, (row[xcol], row[ycol]),
            frameon=False, pad=0, box_alignment=(0.5, 0.5), zorder=3,
        )
        ax.add_artist(ab)

    ax.text(0.0, 1.02, title, transform=ax.transAxes,
            fontsize=15, color=TEXT, fontweight="medium", ha="left", va="bottom")
    ax.text(1.0, 1.02, subtitle, transform=ax.transAxes,
            fontsize=9.5, color=SUBTLE, ha="right", va="bottom")


def build_legend(fig, df: pd.DataFrame):
    handles = []
    labels = []
    for cat in HIGHLIGHT_ORDER:
        n = int((df["vex_category"] == cat).sum())
        if n == 0:
            continue
        color = CATEGORY_COLORS[cat]
        # Border-style handle to match what's drawn on the flags.
        patch = plt.Rectangle((0, 0), 1, 1, facecolor="none",
                              edgecolor=color, linewidth=1.6)
        handles.append(patch)
        labels.append(f"{NICE_NAME[cat]}  ({n})")
    fig.legend(
        handles, labels,
        loc="lower center", ncol=5,
        frameon=False, fontsize=10.5, labelcolor=TEXT,
        bbox_to_anchor=(0.5, 0.018),
        handlelength=1.4, handleheight=0.95, columnspacing=2.4,
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(PROJ_DIR / "projections.parquet")
    print(f"plotting {len(df)} flags")

    fig, axes = plt.subplots(1, 3, figsize=(26, 11))
    fig.patch.set_facecolor(BG)

    for ax, (title, xcol, ycol, sub) in zip(axes, PANELS):
        plot_panel(ax, df, xcol, ycol, title, sub)

    fig.suptitle(
        "Latent flags",
        fontsize=26, fontweight="medium", color=TEXT,
        x=0.025, y=0.965, ha="left",
    )
    fig.text(
        0.025, 0.928,
        "DINOv2 visual embeddings of 197 sovereign flags, projected to 2D. "
        "Borders mark hand-curated vexillological categories — "
        "soft hulls appear only where a category's flags genuinely cluster in the projection.",
        fontsize=12, color=SUBTLE, ha="left",
    )

    build_legend(fig, df)
    plt.subplots_adjust(left=0.015, right=0.985, top=0.86,
                        bottom=0.10, wspace=0.04)

    out_png = OUT_DIR / "latent_flags.png"
    out_pdf = OUT_DIR / "latent_flags.pdf"
    fig.savefig(out_png, dpi=240, facecolor=BG)
    fig.savefig(out_pdf, facecolor=BG)
    print(f"wrote {out_png}")
    print(f"wrote {out_pdf}")


if __name__ == "__main__":
    main()
