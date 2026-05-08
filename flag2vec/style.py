"""Shared style + helpers for flag2vec figures."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageOps
from scipy.spatial import ConvexHull

ROOT = Path(__file__).resolve().parent.parent
PNG_DIR = ROOT / "data" / "png"
PROJ_DIR = ROOT / "data" / "projections"
EMB_DIR = ROOT / "data" / "embeddings"
CSV_PATH = ROOT / "data" / "sovereign_flags.csv"
REGION_CSV = ROOT / "data" / "regions.csv"
OUT_DIR = ROOT / "out"

# Anthropic-ish palette: warm off-white background, restrained accents.
BG = "#F7F4EC"
TEXT = "#1A1A1A"
SUBTLE = "#6B6B6B"
GRID = "#E8E2D2"
FADE = "#C9C4B8"

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
    "saltire":             "Saltire",
    "solid_emblem":       "Solid + emblem",
    "stars_stripes":      "Stars & stripes",
    "heraldic":           "Heraldic",
    "unique":             "Unique",
}

REGION_COLORS = {
    "Africa":   "#C77A3E",
    "Americas": "#3E7CB1",
    "Asia":     "#9C5C9C",
    "Europe":   "#3E8C73",
    "Oceania":  "#C04D6E",
}

PANELS = [
    ("PCA",   "pca_x",   "pca_y",   "21% + 11% var"),
    ("t-SNE", "tsne_x",  "tsne_y",  "perplexity 20, cosine"),
    ("PHATE", "phate_x", "phate_y", "knn 10, decay 20"),
]


def configure_typography():
    plt.rcParams["font.family"] = [
        "Inter", "IBM Plex Sans", "Helvetica Neue", "Helvetica",
        "Arial", "DejaVu Sans",
    ]
    plt.rcParams["axes.titleweight"] = "medium"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42


def load_thumb(iso2: str, height_px: int, border_color: str | None = None,
               border_px: int = 3, faded: bool = False,
               grayscale: bool = False) -> np.ndarray:
    img = Image.open(PNG_DIR / f"{iso2}.png").convert("RGBA")
    bbox = img.getbbox()
    if bbox is not None:
        img = img.crop(bbox)
    scale = height_px / img.height
    new_w = max(1, int(round(img.width * scale)))
    new_h = max(1, int(round(img.height * scale)))
    img = img.resize((new_w, new_h), Image.LANCZOS)
    if grayscale:
        gray = ImageOps.grayscale(img)
        img = Image.merge("RGBA", (gray, gray, gray, img.split()[-1]))
    if faded:
        alpha = img.split()[-1]
        alpha = alpha.point(lambda v: int(v * 0.28))
        img.putalpha(alpha)
    if border_color:
        img = ImageOps.expand(img, border=border_px, fill=border_color)
    return np.asarray(img)


def soft_hull(ax, pts: np.ndarray, color: str,
              alpha_fill=0.08, alpha_edge=0.30, expand=0.04):
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
    if len(pts) < 2:
        return 0.0
    inner = np.linalg.norm(pts[:, None, :] - pts[None, :, :], axis=-1)
    inner_mean = inner[np.triu_indices(len(pts), k=1)].mean()
    outer = np.linalg.norm(all_pts[:, None, :] - all_pts[None, :, :], axis=-1)
    outer_mean = outer[np.triu_indices(len(all_pts), k=1)].mean()
    return float(inner_mean / outer_mean) if outer_mean > 0 else 0.0


def clean_axes(ax, bg=BG):
    ax.set_facecolor(bg)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])


def set_axis_limits(ax, xs: np.ndarray, ys: np.ndarray, pad_frac: float = 0.07):
    pad_x = (xs.max() - xs.min()) * pad_frac
    pad_y = (ys.max() - ys.min()) * pad_frac
    ax.set_xlim(xs.min() - pad_x, xs.max() + pad_x)
    ax.set_ylim(ys.min() - pad_y, ys.max() + pad_y)
