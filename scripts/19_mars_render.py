"""Phase 4 — Mars figures.

Reads:
  data/mars_regions.csv
  data/mars_distance_table.csv
  data/projections/projections_all_phase4.parquet
  out/mars/flags/<region_id>.png
  data/png/<iso2>.png  (Earth flags)

Writes:
  out/mars/mars_map.png         — Mars-disc layout, flags placed at lat/long
  out/mars/joint_embedding.png  — 3-panel PCA/t-SNE/PHATE with Mars highlighted
  out/mars/inheritance_check.png — per-Mars-flag strip vs nearest Earth tradition centroid
  out/mars/inheritance_hit_rate.png — bar chart of nearest-tradition matches
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.patches import Circle, Rectangle, FancyBboxPatch
from PIL import Image, ImageOps

from flag2vec.style import (
    BG, CATEGORY_COLORS, NICE_NAME, OUT_DIR, PNG_DIR, SUBTLE, TEXT,
    clean_axes, configure_typography, set_axis_limits,
)

ROOT = Path(__file__).resolve().parent.parent
MARS_OUT = OUT_DIR / "mars"
MARS_FLAGS = MARS_OUT / "flags"
PROJ_PATH = ROOT / "data" / "projections" / "projections_all_phase4.parquet"
MARS_CSV = ROOT / "data" / "mars_regions.csv"
HIT_CSV = ROOT / "data" / "mars_distance_table.csv"

# Mars-specific aesthetic accents.
MARS_BG = "#F4E5D2"   # warmer cream than the Earth Anthropic BG
MARS_DISC = "#C26E45" # rust-orange disc fill
MARS_GLOW = "#E03A1F" # bright accent for "this is a Mars flag" border


# ─────────────────────────── helpers ──────────────────────────────────────

def _load_thumb_arr(path: Path, height_px: int,
                    border_color: str | None = None,
                    border_px: int = 3,
                    faded: bool = False, grayscale: bool = False) -> np.ndarray:
    img = Image.open(path).convert("RGBA")
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
        alpha = img.split()[-1].point(lambda v: int(v * 0.28))
        img.putalpha(alpha)
    if border_color:
        img = ImageOps.expand(img, border=border_px, fill=border_color)
    return np.asarray(img)


def _earth_thumb(iso2: str, height_px: int, border_color: str | None = None,
                 border_px: int = 3, faded: bool = False,
                 grayscale: bool = False) -> np.ndarray:
    return _load_thumb_arr(
        PNG_DIR / f"{iso2}.png", height_px, border_color, border_px,
        faded, grayscale,
    )


def _mars_thumb(region_id: str, height_px: int,
                border_color: str | None = MARS_GLOW, border_px: int = 4,
                faded: bool = False, grayscale: bool = False) -> np.ndarray:
    return _load_thumb_arr(
        MARS_FLAGS / f"{region_id}.png", height_px, border_color, border_px,
        faded, grayscale,
    )


def _flag_thumb_for_row(row: pd.Series, height_px: int,
                        border_color: str | None,
                        border_px: int,
                        faded: bool = False,
                        grayscale: bool = False) -> np.ndarray:
    if row["is_mars"]:
        return _mars_thumb(row["id"], height_px, border_color or MARS_GLOW,
                           border_px=max(border_px, 4),
                           faded=faded, grayscale=grayscale)
    else:
        return _earth_thumb(row["id"], height_px, border_color, border_px,
                            faded=faded, grayscale=grayscale)


# ─────────────────────────── 1. Mars map ───────────────────────────────────

def render_mars_map() -> None:
    df = pd.read_csv(MARS_CSV)

    fig, ax = plt.subplots(figsize=(18, 11))
    fig.patch.set_facecolor(MARS_BG)
    ax.set_facecolor(MARS_BG)
    clean_axes(ax, bg=MARS_BG)

    # Plate Carrée projection: x = longitude (-180..180), y = latitude (-90..90).
    # Faint topographic backdrop: Tharsis bulge + Hellas as soft circles.
    ax.set_xlim(-185, 185)
    ax.set_ylim(-95, 95)
    # Equator + prime meridian guide lines
    ax.axhline(0, color="#B05030", alpha=0.18, linewidth=1)
    ax.axvline(0, color="#B05030", alpha=0.18, linewidth=1)
    # Polar caps (very faint cream)
    ax.add_patch(Rectangle((-180,  68), 360, 22, color="#F0E0CC", alpha=0.55, linewidth=0))
    ax.add_patch(Rectangle((-180, -90), 360, 22, color="#F0E0CC", alpha=0.55, linewidth=0))
    # Tharsis bulge (rough silhouette)
    ax.add_patch(Circle((-110, 0),  35, color="#A8553A", alpha=0.18, linewidth=0))
    ax.add_patch(Circle(( 70, -42), 22, color="#7C3D2A", alpha=0.22, linewidth=0))  # Hellas
    # Frame
    ax.add_patch(Rectangle((-180, -90), 360, 180, fill=False,
                           edgecolor="#7C3D2A", alpha=0.45, linewidth=1.3))

    # Place each Mars flag at its lat/long
    for _, row in df.iterrows():
        lon = float(row["longitude"])
        lat = float(row["latitude"])
        thumb = _mars_thumb(row["region_id"], height_px=58,
                            border_color=MARS_GLOW, border_px=3)
        ax.add_artist(AnnotationBbox(
            OffsetImage(thumb, zoom=1.0, interpolation="lanczos"),
            (lon, lat),
            frameon=False, pad=0, box_alignment=(0.5, 0.5),
            zorder=4,
        ))
        ax.text(lon, lat - 12, row["name"], fontsize=8.5,
                ha="center", va="top", color="#3A1A0E",
                alpha=0.92, fontweight="medium")

    ax.set_xlabel("areocentric longitude (°E)", fontsize=10, color="#5A2C1A")
    ax.set_ylabel("areocentric latitude (°N)",  fontsize=10, color="#5A2C1A")
    ax.tick_params(colors="#7A3D26", length=0)
    ax.set_xticks([-180, -90, 0, 90, 180])
    ax.set_yticks([-90, -45, 0, 45, 90])
    for spine in ax.spines.values():
        spine.set_visible(False)

    fig.suptitle("flag2vec — Phase 4: Mars",
                 fontsize=30, fontweight="medium", color="#3A1A0E",
                 x=0.04, y=0.97, ha="left")
    fig.text(0.04, 0.93,
             "Year ~2300.  Twenty-five freshly-minted Mars regional governments commission their first flags.  "
             "Each region inherits an Earth flag tradition by climate / terrain / settlement-order analogy, "
             "then layers a Mars-specific motif on top.",
             fontsize=12, color="#5A2C1A", ha="left")
    fig.text(0.04, 0.025,
             "backdrop: faint Tharsis bulge (left of centre) and Hellas Basin (lower right) over a Plate Carrée projection.  "
             "polar caps shown as cream bands.",
             fontsize=9, color="#7C3D2A", ha="left", alpha=0.85)

    plt.subplots_adjust(left=0.04, right=0.97, top=0.88, bottom=0.07)
    out = MARS_OUT / "mars_map.png"
    fig.savefig(out, dpi=160, facecolor=MARS_BG)
    plt.close(fig)
    print(f"wrote {out}")


# ─────────────────────────── 2. joint embedding ────────────────────────────

def render_joint_embedding() -> None:
    df = pd.read_parquet(PROJ_PATH)
    n_earth = (df["kind"] != "mars").sum()
    n_mars = (df["kind"] == "mars").sum()

    PANELS = [
        ("PCA",   "pca_x",   "pca_y",   "global axes"),
        ("t-SNE", "tsne_x",  "tsne_y",  "perplexity 30, cosine"),
        ("PHATE", "phate_x", "phate_y", "knn 15, decay 20"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(34, 14))
    fig.patch.set_facecolor(BG)

    earth_df = df[~df["is_mars"]].copy()
    mars_df = df[df["is_mars"]].copy()

    for ax, (title, xcol, ycol, sub) in zip(axes, PANELS):
        clean_axes(ax)
        all_pts = df[[xcol, ycol]].to_numpy()
        set_axis_limits(ax, all_pts[:, 0], all_pts[:, 1], pad_frac=0.06)

        # Earth: small thumbs, faded slightly, drawn first.
        for _, row in earth_df.iterrows():
            cat = row["vex_category"]
            border = CATEGORY_COLORS.get(cat, "#888888") if row["kind"] == "sovereign" else None
            h = 22 if row["kind"] == "sovereign" else 16
            try:
                thumb = _earth_thumb(row["id"], h, border, border_px=1,
                                     faded=row["kind"] != "sovereign")
            except FileNotFoundError:
                continue
            ax.add_artist(AnnotationBbox(
                OffsetImage(thumb, zoom=1.0, interpolation="lanczos"),
                (row[xcol], row[ycol]),
                frameon=False, pad=0, box_alignment=(0.5, 0.5),
                zorder=2 if row["kind"] == "sovereign" else 1,
            ))

        # Mars: much bigger, with bright glow border, drawn on top.
        for _, row in mars_df.iterrows():
            thumb = _mars_thumb(row["id"], 56, MARS_GLOW, border_px=5)
            ax.add_artist(AnnotationBbox(
                OffsetImage(thumb, zoom=1.0, interpolation="lanczos"),
                (row[xcol], row[ycol]),
                frameon=False, pad=0, box_alignment=(0.5, 0.5),
                zorder=6,
            ))

        ax.text(0.0, 1.02, title, transform=ax.transAxes,
                fontsize=15, color=TEXT, fontweight="medium", ha="left", va="bottom")
        ax.text(1.0, 1.02, sub, transform=ax.transAxes,
                fontsize=10, color=SUBTLE, ha="right", va="bottom")

    fig.suptitle("flag2vec — Phase 4: Mars in DINOv2 space",
                 fontsize=32, fontweight="medium", color=TEXT,
                 x=0.025, y=0.965, ha="left")
    fig.text(0.025, 0.92,
             f"DINOv2 ViT-S/14 embeddings of {n_earth} Earth sovereign flags jointly with "
             f"{n_mars} procedurally-generated Mars-region flags (positional embeddings interpolated to 224×224 for CPU speed).  "
             "Mars flags drawn larger with a bright-red glow border.  "
             "Watch where Vastitas Borealis (Nordic cross) lands relative to Sweden/Finland, "
             "Arabia Terra (pan-Arab) relative to Iraq/Jordan, and the heraldic-inheritance Mars flags (Argyre, Marineris, Noachis) "
             "drift toward the solid-emblem cluster.",
             fontsize=12, color=SUBTLE, ha="left")

    plt.subplots_adjust(left=0.012, right=0.988, top=0.88,
                        bottom=0.03, wspace=0.04)
    out = MARS_OUT / "joint_embedding.png"
    fig.savefig(out, dpi=170, facecolor=BG)
    plt.close(fig)
    print(f"wrote {out}")


# ─────────────────────────── 3. inheritance check ──────────────────────────

def render_inheritance_check() -> None:
    hit_df = pd.read_csv(HIT_CSV)
    mars_df = pd.read_csv(MARS_CSV).set_index("region_id")
    # Sort by inheritance success (closest match first = lowest distance_inherited).
    hit_df = hit_df.sort_values("distance_inherited", ascending=True).reset_index(drop=True)

    n = len(hit_df)
    row_h = 0.78
    fig, ax = plt.subplots(figsize=(15, n * row_h + 1.7))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.set_xlim(0, 12)
    ax.set_ylim(-0.5, n - 0.5)
    ax.invert_yaxis()
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([]); ax.set_yticks([])

    # Layout columns:
    #   x=0.5–2.5 : Mars flag (left)
    #   x=3.0–6.5 : arrow + distance label
    #   x=7.0–9.0 : nearest Earth tradition centroid flag (right)
    #   x=9.5–11.8: text (Mars name | tradition labels | hit/miss)

    for i, row in hit_df.iterrows():
        y = i

        # Mars flag
        thumb = _mars_thumb(row["region_id"], 58, MARS_GLOW, border_px=3)
        ax.imshow(
            thumb,
            extent=(0.5, 2.5, y - 0.34, y + 0.34),
            aspect="auto", zorder=4,
        )

        # Arrow with distance
        d_inh = row["distance_inherited"]
        d_near = row["distance_nearest"]
        arrow_color = CATEGORY_COLORS.get(row["nearest_tradition"], "#888888")
        ax.annotate(
            "", xy=(6.4, y), xytext=(2.7, y),
            arrowprops=dict(arrowstyle="->", color=arrow_color,
                            lw=1.6, alpha=0.85),
            zorder=3,
        )
        ax.text(4.55, y - 0.2,
                f"d(Mars → inherited)  {d_inh:.3f}",
                fontsize=9, color=TEXT, ha="center", va="bottom")
        ax.text(4.55, y + 0.05,
                f"d(Mars → nearest)    {d_near:.3f}    rank inherited #{int(row['rank_inherited'])}",
                fontsize=9, color=SUBTLE, ha="center", va="bottom")

        # Nearest Earth centroid flag
        nearest_iso = row["nearest_centroid_iso2"]
        try:
            cent_thumb = _earth_thumb(
                nearest_iso, 58,
                CATEGORY_COLORS.get(row["nearest_tradition"], "#888888"),
                border_px=3,
            )
            ax.imshow(
                cent_thumb,
                extent=(7.0, 9.0, y - 0.34, y + 0.34),
                aspect="auto", zorder=4,
            )
        except FileNotFoundError:
            pass

        # Text column
        feature = mars_df.loc[row["region_id"], "feature_type"]
        hit_marker = "✓ hit" if row["hit"] else "✗ miss"
        hit_color = "#3E8C73" if row["hit"] else "#A02C2C"
        ax.text(9.4, y - 0.18,
                f"{row['name']}  ({feature})",
                fontsize=11, color=TEXT, fontweight="medium", ha="left", va="bottom")
        ax.text(9.4, y + 0.05,
                f"inherited: {NICE_NAME.get(row['inherited_tradition'], row['inherited_tradition'])}    "
                f"nearest: {NICE_NAME.get(row['nearest_tradition'], row['nearest_tradition'])}",
                fontsize=9, color=SUBTLE, ha="left", va="bottom")
        ax.text(9.4, y + 0.27, hit_marker,
                fontsize=10, color=hit_color, fontweight="medium", ha="left", va="bottom")

    n_hit = int(hit_df["hit"].sum())
    fig.suptitle("Phase 4 — does each Mars flag land near its inherited Earth tradition?",
                 fontsize=20, fontweight="medium", color=TEXT,
                 x=0.04, y=0.985, ha="left")
    fig.text(0.04, 0.965,
             f"Sorted by distance to inherited tradition centroid (closest match first).  "
             f"Left: Mars flag.  Right: prototypical Earth flag of the Mars flag's nearest tradition.  "
             f"Top-1 hits: {n_hit}/{n} = {n_hit/n:.0%}.  "
             f"Chance ≈ 1/{hit_df['nearest_tradition'].nunique()} = {1/hit_df['nearest_tradition'].nunique():.0%}.",
             fontsize=11, color=SUBTLE, ha="left")

    plt.subplots_adjust(left=0.02, right=0.98, top=0.96, bottom=0.01)
    out = MARS_OUT / "inheritance_check.png"
    fig.savefig(out, dpi=160, facecolor=BG)
    plt.close(fig)
    print(f"wrote {out}")


# ─────────────────────────── 4. hit rate bar ───────────────────────────────

def render_hit_rate_bar() -> None:
    hit_df = pd.read_csv(HIT_CSV)
    chance = 1.0 / hit_df["nearest_tradition"].nunique()
    n_hit = int(hit_df["hit"].sum())
    n = len(hit_df)
    overall = n_hit / n

    # Stable plot order: by inherited tradition then region.
    hit_df = hit_df.sort_values(
        ["inherited_tradition", "region_id"]
    ).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(13, 9))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(colors=SUBTLE, length=0)

    y = np.arange(len(hit_df))
    rank = hit_df["rank_inherited"].clip(upper=14).to_numpy()
    score = 1.0 - (rank - 1) / 14.0  # 1 = top-1 hit, 0 = bottom of 15

    colors = [
        CATEGORY_COLORS.get(t, "#888888")
        for t in hit_df["inherited_tradition"]
    ]
    ax.barh(y, score, color=colors, edgecolor=BG, linewidth=0.6, alpha=0.9)

    # Threshold lines
    ax.axvline(score.mean(), color=TEXT, alpha=0.4, linewidth=1, linestyle="--")
    ax.text(score.mean() + 0.012, len(y) - 0.5,
            f"mean  {score.mean():.2f}",
            color=TEXT, fontsize=9, va="top", alpha=0.65)
    chance_score = 1.0 - (np.ceil(15 * chance) - 1) / 14.0  # nominal "chance" line
    # Chance = 1/15 → in expectation the inherited tradition would rank 8th, so score ≈ 0.5
    ax.axvline(0.5, color=SUBTLE, alpha=0.5, linewidth=1, linestyle=":")
    ax.text(0.5 + 0.012, -0.5, "chance baseline (rank 8 of 15)",
            color=SUBTLE, fontsize=9, va="top", alpha=0.85)

    labels = [
        f"{r['name']}  ←  {NICE_NAME.get(r['inherited_tradition'], r['inherited_tradition'])}    "
        f"(rank {int(r['rank_inherited'])})"
        for _, r in hit_df.iterrows()
    ]
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("inheritance score  ·  1.00 = inherited tradition is the nearest of 15",
                  color=SUBTLE, fontsize=10.5)

    fig.suptitle("Phase 4 — inheritance score per Mars region",
                 fontsize=22, fontweight="medium", color=TEXT,
                 x=0.04, y=0.965, ha="left")
    fig.text(0.04, 0.92,
             f"Top-1 hit rate {n_hit}/{n} = {overall:.0%} (chance ≈ {chance:.0%}, lift ≈ {overall/chance:.1f}×).  "
             f"Bar color = inherited tradition.  Bar length = how high the inherited tradition ranks among "
             f"the 15 Earth tradition centroids in cosine distance to the Mars flag.",
             fontsize=11, color=SUBTLE, ha="left")

    plt.subplots_adjust(left=0.40, right=0.96, top=0.85, bottom=0.07)
    out = MARS_OUT / "inheritance_hit_rate.png"
    fig.savefig(out, dpi=170, facecolor=BG)
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    configure_typography()
    MARS_OUT.mkdir(parents=True, exist_ok=True)
    print("=== mars_map ===")
    render_mars_map()
    print("=== joint_embedding ===")
    render_joint_embedding()
    print("=== inheritance_check ===")
    render_inheritance_check()
    print("=== inheritance_hit_rate ===")
    render_hit_rate_bar()
    print("done.")


if __name__ == "__main__":
    main()
