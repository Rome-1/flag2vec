"""Phase 3 figures: hero, trajectories, trajectory_lengths, per_successor.

Reads:
  data/embeddings/dinov2_phase3.npy
  data/embeddings/iso2_order_phase3.txt
  data/projections/projections_all_phase3.parquet

Writes:
  out/phase3/hero.png
  out/phase3/trajectories.png
  out/phase3/trajectory_lengths.png
  out/phase3/per_successor.png
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image, ImageOps
from sklearn.preprocessing import normalize

from flag2vec.style import (
    BG, CATEGORY_COLORS, NICE_NAME, OUT_DIR, PNG_DIR, SUBTLE, TEXT,
    clean_axes, configure_typography, set_axis_limits,
)

ROOT = Path(__file__).resolve().parent.parent
EMB_DIR = ROOT / "data" / "embeddings"
PROJ_DIR = ROOT / "data" / "projections"
OUT = OUT_DIR / "phase3"

GOLD = "#C08A3E"
SEPIA_TINT = (180, 150, 100)


# ─────────────────────────── thumbnails (sepia for historical) ────────────

def load_thumb_sepia(iso2: str, height_px: int,
                     border_color: str | None = None,
                     border_px: int = 2,
                     dotted: bool = False,
                     faded: bool = False,
                     grayscale: bool = False) -> np.ndarray:
    """Load a flag thumbnail. Historical flags are desaturated to sepia."""
    img = Image.open(PNG_DIR / f"{iso2}.png").convert("RGBA")
    bbox = img.getbbox()
    if bbox is not None:
        img = img.crop(bbox)
    scale = height_px / img.height
    new_w = max(1, int(round(img.width * scale)))
    new_h = max(1, int(round(img.height * scale)))
    img = img.resize((new_w, new_h), Image.LANCZOS)

    is_hist = iso2.startswith("hist-")
    if is_hist:
        # Desaturate to sepia.
        gray = ImageOps.grayscale(img)
        sepia = ImageOps.colorize(gray, black=(60, 40, 20),
                                  white=(245, 230, 205))
        img = Image.merge("RGBA", (*sepia.split(), img.split()[-1]))
    if grayscale and not is_hist:
        gray = ImageOps.grayscale(img)
        img = Image.merge("RGBA", (gray, gray, gray, img.split()[-1]))
    if faded:
        a = img.split()[-1]
        a = a.point(lambda v: int(v * 0.35))
        img.putalpha(a)
    if border_color:
        img = ImageOps.expand(img, border=border_px, fill=border_color)
    return np.asarray(img)


# ─────────────────────────── data helpers ────────────────────────────────

def load_data():
    df = pd.read_parquet(PROJ_DIR / "projections_all_phase3.parquet")
    X = np.load(EMB_DIR / "dinov2_phase3.npy")
    order = (EMB_DIR / "iso2_order_phase3.txt").read_text().strip().splitlines()
    Xn = normalize(X, norm="l2")
    iso2_to_idx = {c: i for i, c in enumerate(order)}
    return df, Xn, iso2_to_idx


def cosine_dist(Xn, i, j) -> float:
    return float(1.0 - Xn[i] @ Xn[j])


# ─────────────────────────── figure 1: hero (3-panel) ─────────────────────

def render_hero(df: pd.DataFrame) -> None:
    PANELS = [
        ("PCA",   "pca_x",   "pca_y",   "global axes"),
        ("t-SNE", "tsne_x",  "tsne_y",  "perplexity 30, cosine"),
        ("PHATE", "phate_x", "phate_y", "knn 15, decay 20"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(34, 14))
    fig.patch.set_facecolor(BG)

    n_sov = (df["kind"] == "sovereign").sum()
    n_sub = (df["kind"] == "subdivision").sum()
    n_hist = (df["kind"] == "historical").sum()

    for ax, (title, xcol, ycol, sub) in zip(axes, PANELS):
        clean_axes(ax)
        all_pts = df[[xcol, ycol]].to_numpy()
        set_axis_limits(ax, all_pts[:, 0], all_pts[:, 1], pad_frac=0.05)

        for _, row in df.iterrows():
            kind = row["kind"]
            cat = row["vex_category"]
            border = CATEGORY_COLORS.get(cat) if cat in CATEGORY_COLORS else "#888888"
            if kind == "sovereign":
                h = 30; bw = 2; b = border; z = 4
            elif kind == "subdivision":
                h = 22; bw = 2; b = border; z = 2
            else:  # historical
                h = 26; bw = 3; b = GOLD; z = 5
            thumb = load_thumb_sepia(row["iso2"], h, b, border_px=bw)
            ax.add_artist(AnnotationBbox(
                OffsetImage(thumb, zoom=1.0, interpolation="lanczos"),
                (row[xcol], row[ycol]),
                frameon=False, pad=0, box_alignment=(0.5, 0.5),
                zorder=z,
            ))

        ax.text(0.0, 1.02, title, transform=ax.transAxes,
                fontsize=15, color=TEXT, fontweight="medium",
                ha="left", va="bottom")
        ax.text(1.0, 1.02, sub, transform=ax.transAxes,
                fontsize=10, color=SUBTLE, ha="right", va="bottom")

    fig.suptitle("flag2vec — Phase 3",
                 fontsize=32, fontweight="medium", color=TEXT,
                 x=0.025, y=0.965, ha="left")
    fig.text(0.025, 0.92,
             f"DINOv2 visual embeddings of {n_sov} sovereigns + {n_sub} subdivisions "
             f"+ {n_hist} historical flags. Historical flags rendered sepia-toned "
             "with a gold border and drawn on top, so you can see where each "
             "predecessor sits relative to its modern successor.",
             fontsize=12, color=SUBTLE, ha="left")
    plt.subplots_adjust(left=0.012, right=0.988, top=0.88,
                        bottom=0.03, wspace=0.04)

    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "hero.png", dpi=180, facecolor=BG)
    plt.close(fig)
    print(f"  wrote {OUT / 'hero.png'}")


# ─────────────────────────── figure 2: trajectories ───────────────────────

def render_trajectories(df: pd.DataFrame, panel: str = "tsne") -> None:
    """One large panel with arrows from each historical flag to its successor."""
    xcol, ycol = f"{panel}_x", f"{panel}_y"
    title = {"tsne": "t-SNE", "phate": "PHATE", "pca": "PCA"}[panel]
    sub_label = {"tsne": "perplexity 30, cosine",
                 "phate": "knn 15, decay 20",
                 "pca": "global axes"}[panel]

    hist_df = df[df["kind"] == "historical"].copy()
    sov_df = df[df["kind"] == "sovereign"].set_index("iso2")

    # Resolve successors: lowercase iso2 in sov_df.
    pairs = []
    for _, row in hist_df.iterrows():
        succ = (row["successor_iso2"] or "").lower()
        if succ in sov_df.index:
            sx, sy = sov_df.loc[succ, xcol], sov_df.loc[succ, ycol]
            pairs.append({
                "iso2": row["iso2"], "name": row["name"],
                "hx": row[xcol], "hy": row[ycol],
                "sx": sx, "sy": sy,
                "successor": succ,
                "era_start": row["era_start"],
                "era_end": row["era_end"],
            })
    pdf = pd.DataFrame(pairs)
    pdf["dx"] = pdf["sx"] - pdf["hx"]
    pdf["dy"] = pdf["sy"] - pdf["hy"]
    pdf["len2d"] = np.hypot(pdf["dx"], pdf["dy"])
    pdf["era_mid"] = (pdf["era_start"].astype(float)
                      + pdf["era_end"].astype(float)) / 2.0

    fig, ax = plt.subplots(figsize=(18, 16))
    fig.patch.set_facecolor(BG)
    clean_axes(ax)
    all_pts = df[[xcol, ycol]].to_numpy()
    set_axis_limits(ax, all_pts[:, 0], all_pts[:, 1], pad_frac=0.06)

    # Background: all sovereigns + subdivisions, faded.
    for _, row in df.iterrows():
        if row["kind"] == "historical":
            continue
        h = 22 if row["kind"] == "sovereign" else 16
        thumb = load_thumb_sepia(row["iso2"], h, None, border_px=0,
                                 faded=True, grayscale=True)
        ax.add_artist(AnnotationBbox(
            OffsetImage(thumb, zoom=1.0, interpolation="lanczos"),
            (row[xcol], row[ycol]),
            frameon=False, pad=0, box_alignment=(0.5, 0.5), zorder=1,
        ))

    # Arrows colored by era midpoint (viridis, 10-year bins).
    if len(pdf) > 0:
        era_min = float(pdf["era_mid"].min())
        era_max = float(pdf["era_mid"].max())
        norm = Normalize(vmin=era_min, vmax=era_max)
        cmap = plt.get_cmap("viridis")

        for _, p in pdf.iterrows():
            color = cmap(norm(p["era_mid"]))
            ax.annotate("",
                xy=(p["sx"], p["sy"]), xytext=(p["hx"], p["hy"]),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=2.0, alpha=0.85,
                                shrinkA=14, shrinkB=14,
                                mutation_scale=18),
                zorder=2,
            )

    # Foreground: full-color successors + sepia historical, drawn on top of arrows.
    successor_codes = set(pdf["successor"].tolist())
    for _, row in df.iterrows():
        is_hist = row["kind"] == "historical"
        is_succ = row["iso2"] in successor_codes
        if not (is_hist or is_succ):
            continue
        if is_hist:
            h = 32; b = GOLD; bw = 3; z = 5
        else:
            cat = row["vex_category"]
            b = CATEGORY_COLORS.get(cat, "#888888")
            h = 30; bw = 2; z = 4
        thumb = load_thumb_sepia(row["iso2"], h, b, border_px=bw)
        ax.add_artist(AnnotationBbox(
            OffsetImage(thumb, zoom=1.0, interpolation="lanczos"),
            (row[xcol], row[ycol]),
            frameon=False, pad=0, box_alignment=(0.5, 0.5), zorder=z,
        ))

    # Annotate longest 5 and shortest 5 trajectories.
    if len(pdf) >= 5:
        longest = pdf.nlargest(5, "len2d")
        shortest = pdf.nsmallest(5, "len2d")
        for _, p in pd.concat([longest, shortest]).iterrows():
            mx = (p["hx"] + p["sx"]) / 2
            my = (p["hy"] + p["sy"]) / 2
            ax.annotate(p["name"], xy=(mx, my),
                        fontsize=8, color=TEXT,
                        ha="center", va="center",
                        bbox=dict(boxstyle="round,pad=0.2",
                                  fc=BG, ec=SUBTLE, lw=0.5,
                                  alpha=0.85),
                        zorder=6)

    fig.suptitle("Historical → modern: trajectories in DINOv2 space",
                 fontsize=26, fontweight="medium", color=TEXT,
                 x=0.025, y=0.965, ha="left")
    fig.text(0.025, 0.93,
             f"{title} projection ({sub_label}). "
             f"Each arrow runs from a historical flag (sepia, gold border) to its "
             "modern successor (full color). Arrow color encodes era midpoint "
             "(viridis: dark = older, light = recent). "
             f"Sovereigns and subdivisions in the background as faded grayscale.",
             fontsize=11, color=SUBTLE, ha="left")
    if len(pdf) > 0:
        # Era colorbar
        sm = ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar_ax = fig.add_axes([0.86, 0.05, 0.01, 0.20])
        cbar = fig.colorbar(sm, cax=cbar_ax)
        cbar.set_label("era midpoint (year)", fontsize=9, color=SUBTLE)
        cbar.ax.tick_params(labelsize=8, colors=SUBTLE)
        cbar.outline.set_visible(False)

    plt.subplots_adjust(left=0.02, right=0.98, top=0.88, bottom=0.03)
    fig.savefig(OUT / "trajectories.png", dpi=180, facecolor=BG)
    plt.close(fig)
    print(f"  wrote {OUT / 'trajectories.png'}")


# ─────────────────────────── figure 3: trajectory_lengths ─────────────────

def render_trajectory_lengths(df: pd.DataFrame, Xn: np.ndarray,
                              iso2_to_idx: dict) -> None:
    """Bar chart: cosine distance between each historical flag and its
    successor in 384-dim DINOv2 space."""
    hist_df = df[df["kind"] == "historical"].copy()
    rows = []
    for _, row in hist_df.iterrows():
        h = row["iso2"]
        s = (row["successor_iso2"] or "").lower()
        if h not in iso2_to_idx or s not in iso2_to_idx:
            continue
        d = cosine_dist(Xn, iso2_to_idx[h], iso2_to_idx[s])
        rows.append({"iso2": h, "name": row["name"],
                     "successor": s, "dist": d})
    rdf = pd.DataFrame(rows).sort_values("dist", ascending=True)

    fig, ax = plt.subplots(figsize=(13, max(7, 0.34 * len(rdf))))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    for s in ax.spines.values():
        s.set_visible(False)

    y = np.arange(len(rdf))
    color_seq = [CATEGORY_COLORS.get(
        df.set_index("iso2").loc[code]["vex_category"], "#777777")
        for code in rdf["iso2"]]
    ax.barh(y, rdf["dist"].values, color=color_seq, alpha=0.85, height=0.7)
    for yi, (name, d, succ) in enumerate(zip(rdf["name"], rdf["dist"],
                                              rdf["successor"])):
        ax.text(d + 0.005, yi, f"→ {succ.upper()}  ({d:.3f})",
                va="center", ha="left", fontsize=8, color=SUBTLE)
    ax.set_yticks(y)
    ax.set_yticklabels(rdf["name"], fontsize=9, color=TEXT)
    ax.set_xlabel("cosine distance (DINOv2, 384-dim)", color=SUBTLE, fontsize=10)
    ax.tick_params(axis="x", colors=SUBTLE, labelsize=9)
    ax.tick_params(axis="y", colors=TEXT, length=0)
    ax.set_xlim(0, max(0.05, rdf["dist"].max() * 1.18))
    ax.grid(axis="x", color="#E8E2D2", linewidth=0.7)
    ax.set_axisbelow(True)

    fig.suptitle("Trajectory length: how visually radical was each transition?",
                 fontsize=20, fontweight="medium", color=TEXT,
                 x=0.025, y=0.96, ha="left")
    fig.text(0.025, 0.92,
             "Cosine distance between each historical flag and its modern successor "
             "in DINOv2 space, sorted ascending. Bars colored by historical-flag "
             "vex category. Long bars = the modern flag looks visually unrelated "
             "to its predecessor; short bars = visual continuity preserved.",
             fontsize=10, color=SUBTLE, ha="left")
    plt.subplots_adjust(left=0.22, right=0.97, top=0.88, bottom=0.07)
    fig.savefig(OUT / "trajectory_lengths.png", dpi=180, facecolor=BG)
    plt.close(fig)
    print(f"  wrote {OUT / 'trajectory_lengths.png'}")


# ─────────────────────────── figure 4: per_successor ──────────────────────

def render_per_successor(df: pd.DataFrame, Xn: np.ndarray,
                         iso2_to_idx: dict) -> None:
    """Small-multiples for modern countries with multiple historical predecessors.
    Each panel shows the modern flag (large, gold border) with all its
    historical predecessors (sepia) to its left, with cosine-distance labels.
    """
    hist_df = df[df["kind"] == "historical"].copy()
    sov_df = df[df["kind"] == "sovereign"].set_index("iso2")

    grouped = hist_df.groupby(hist_df["successor_iso2"].str.lower())
    multi = [(succ, grp) for succ, grp in grouped if len(grp) >= 2]
    multi.sort(key=lambda x: x[0])

    # Also include the most-radical singletons for visual interest (top 4 by distance).
    singletons = []
    for succ, grp in grouped:
        if len(grp) != 1:
            continue
        h = grp.iloc[0]["iso2"]
        s = succ
        if h in iso2_to_idx and s in iso2_to_idx:
            d = cosine_dist(Xn, iso2_to_idx[h], iso2_to_idx[s])
            singletons.append((d, succ, grp))
    singletons.sort(reverse=True)
    multi.extend([(s, g) for _, s, g in singletons[:4]])

    n_panels = len(multi)
    if n_panels == 0:
        print("  no per-successor groups; skipping")
        return
    n_cols = 3
    n_rows = (n_panels + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5.6 * n_cols, 4.0 * n_rows))
    fig.patch.set_facecolor(BG)
    axes = np.atleast_2d(axes)

    for i, (succ, grp) in enumerate(multi):
        ax = axes[i // n_cols][i % n_cols]
        clean_axes(ax)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        if succ not in sov_df.index:
            ax.text(0.5, 0.5, f"successor {succ} missing",
                    ha="center", va="center", color=SUBTLE)
            continue
        sov_name = sov_df.loc[succ, "name"]
        sov_cat = sov_df.loc[succ, "vex_category"]
        sov_border = CATEGORY_COLORS.get(sov_cat, "#888888")

        # Title.
        ax.text(0.0, 1.02, sov_name, transform=ax.transAxes,
                fontsize=14, color=TEXT, fontweight="medium",
                ha="left", va="bottom")
        ax.text(1.0, 1.02, f"{len(grp)} predecessor"
                + ("s" if len(grp) != 1 else ""),
                transform=ax.transAxes, fontsize=9, color=SUBTLE,
                ha="right", va="bottom")

        # Modern flag on the right.
        modern_thumb = load_thumb_sepia(succ, 80, sov_border, border_px=3)
        ax.add_artist(AnnotationBbox(
            OffsetImage(modern_thumb, zoom=1.0),
            (0.82, 0.5), frameon=False, pad=0,
            box_alignment=(0.5, 0.5), zorder=3,
        ))

        # Historical predecessors stacked vertically on the left.
        n = len(grp)
        ys = np.linspace(0.85, 0.15, n) if n > 1 else [0.5]
        for (_, row), yh in zip(grp.iterrows(), ys):
            h_thumb = load_thumb_sepia(row["iso2"], 60, GOLD, border_px=3)
            ax.add_artist(AnnotationBbox(
                OffsetImage(h_thumb, zoom=1.0),
                (0.18, yh), frameon=False, pad=0,
                box_alignment=(0.5, 0.5), zorder=3,
            ))
            d = cosine_dist(Xn, iso2_to_idx[row["iso2"]],
                            iso2_to_idx[succ])
            ax.annotate("", xy=(0.65, 0.5), xytext=(0.30, yh),
                        arrowprops=dict(arrowstyle="-|>",
                                        color="#888", lw=1.2,
                                        alpha=0.7,
                                        shrinkA=4, shrinkB=4),
                        zorder=2)
            mx = (0.30 + 0.65) / 2
            my = (yh + 0.5) / 2
            ax.text(mx, my + 0.025, f"d={d:.2f}",
                    fontsize=8, color=TEXT, ha="center", va="bottom")
            era = f"{int(row['era_start'])}–{int(row['era_end'])}"
            ax.text(0.02, yh - 0.10,
                    f"{row['name']}  ({era})", fontsize=7, color=SUBTLE,
                    ha="left", va="center")

    # Hide unused axes.
    for j in range(n_panels, n_rows * n_cols):
        axes[j // n_cols][j % n_cols].axis("off")

    fig.suptitle("Per-successor: predecessors and the modern flag",
                 fontsize=22, fontweight="medium", color=TEXT,
                 x=0.025, y=0.985, ha="left")
    fig.text(0.025, 0.965,
             "Modern country (right) and all its historical predecessors (left, "
             "sepia) with cosine distance between each pair in DINOv2 space.",
             fontsize=10, color=SUBTLE, ha="left")
    plt.subplots_adjust(left=0.02, right=0.98, top=0.94, bottom=0.02,
                        wspace=0.04, hspace=0.20)
    fig.savefig(OUT / "per_successor.png", dpi=170, facecolor=BG)
    plt.close(fig)
    print(f"  wrote {OUT / 'per_successor.png'}")


def main() -> int:
    configure_typography()
    OUT.mkdir(parents=True, exist_ok=True)

    print("[load]")
    df, Xn, iso2_to_idx = load_data()

    print("[hero]")
    render_hero(df)

    print("[trajectories]")
    # Pick t-SNE; PHATE if t-SNE happens to be too cramped is also valid.
    render_trajectories(df, panel="tsne")

    print("[trajectory_lengths]")
    render_trajectory_lengths(df, Xn, iso2_to_idx)

    print("[per_successor]")
    render_per_successor(df, Xn, iso2_to_idx)

    print("done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
