"""Image-based gallery analyses.

  - centroid_flags.png       most prototypical flag per vex category
  - distant_pairs.png        flags maximally far apart in DINOv2 space
  - cross_neighbors.png      least-distant cross-category neighbor pairs
  - lof_outliers.png         "weirdest" flags by Local Outlier Factor
  - color_count_radius.png   color-count vs distance from global centroid
  - symmetry_scatter.png     flags colored by horizontal/vertical symmetry score
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.preprocessing import normalize

from flag2vec.style import (
    BG, CATEGORY_COLORS, CSV_PATH, EMB_DIR, NICE_NAME, OUT_DIR, PNG_DIR,
    SUBTLE, TEXT, configure_typography, load_thumb,
)

OUT_ANALYSIS = OUT_DIR / "analysis"


def load_data():
    X = np.load(EMB_DIR / "dinov2_vits14.npy")
    iso2 = (EMB_DIR / "iso2_order.txt").read_text().strip().splitlines()
    meta = pd.read_csv(CSV_PATH).set_index("iso2").loc[iso2].reset_index()
    Xn = normalize(X, norm="l2")
    return X, Xn, iso2, meta


def setup_axes(ax, title, subtitle):
    ax.set_facecolor(BG)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(colors=SUBTLE, length=0)
    if title:
        ax.set_title(title, color=TEXT, fontsize=15, fontweight="medium",
                     loc="left", pad=12)
    if subtitle:
        ax.text(1.0, 1.02, subtitle, transform=ax.transAxes,
                fontsize=10, color=SUBTLE, ha="right", va="bottom")


def fig_centroid_flags(Xn: np.ndarray, meta: pd.DataFrame):
    """For each vex category, find the flag closest to the category centroid."""
    cats_by_size = (meta["vex_category"].value_counts()
                    .index.tolist())
    rows = []
    for cat in cats_by_size:
        mask = (meta["vex_category"] == cat).values
        if mask.sum() < 2:
            continue
        centroid = Xn[mask].mean(axis=0)
        centroid /= np.linalg.norm(centroid) + 1e-9
        dists = 1 - Xn[mask] @ centroid
        i = np.argmin(dists)
        sub = meta.loc[mask].iloc[i]
        within = 1 - Xn[mask] @ centroid
        rows.append({
            "category": cat, "n": int(mask.sum()),
            "iso2": sub["iso2"], "name": sub["name"],
            "mean_dist": float(within.mean()),
        })
    grid = pd.DataFrame(rows)

    n = len(grid)
    cols = 5
    rows_n = int(np.ceil(n / cols))
    fig = plt.figure(figsize=(3.4 * cols, 3.4 * rows_n + 1.4))
    fig.patch.set_facecolor(BG)
    fig.suptitle("most prototypical flag per category",
                 fontsize=22, fontweight="medium", color=TEXT,
                 x=0.04, y=0.97, ha="left")
    fig.text(0.04, 0.93,
             "Closest flag (cosine) to its category's mean DINOv2 vector. "
             "Dispersion = mean within-category cosine distance to centroid.",
             fontsize=11, color=SUBTLE, ha="left")

    for i, row in grid.reset_index(drop=True).iterrows():
        ax = fig.add_subplot(rows_n, cols, i + 1)
        ax.set_facecolor(BG)
        ax.axis("off")
        thumb = load_thumb(row["iso2"], height_px=140,
                           border_color=CATEGORY_COLORS.get(row["category"]),
                           border_px=4)
        ax.imshow(thumb)
        ax.set_title(NICE_NAME.get(row["category"], row["category"]),
                     fontsize=12, color=TEXT, fontweight="medium", loc="left",
                     pad=4)
        ax.text(0.0, -0.06,
                f"{row['name']}  ·  n={row['n']}  ·  σ={row['mean_dist']:.2f}",
                transform=ax.transAxes, fontsize=9, color=SUBTLE,
                ha="left", va="top")

    plt.subplots_adjust(left=0.03, right=0.97, top=0.88, bottom=0.04,
                        hspace=0.45, wspace=0.15)
    fig.savefig(OUT_ANALYSIS / "centroid_flags.png", dpi=200, facecolor=BG)
    plt.close(fig)


def fig_distant_pairs(Xn: np.ndarray, meta: pd.DataFrame, top: int = 10):
    """Pairs of flags most distant in DINOv2 space (deduped — each flag at most once)."""
    n = len(Xn)
    sims = Xn @ Xn.T
    np.fill_diagonal(sims, 1.0)
    iu = np.triu_indices(n, k=1)
    pair_sims = sims[iu]
    order = np.argsort(pair_sims)
    pairs = []
    seen = set()
    for k in order:
        i, j = iu[0][k], iu[1][k]
        if i in seen or j in seen:
            continue
        pairs.append((int(i), int(j), float(1 - pair_sims[k])))
        seen.update([i, j])
        if len(pairs) >= top:
            break

    fig, axes = plt.subplots(top, 2, figsize=(8.5, top * 1.5))
    fig.patch.set_facecolor(BG)
    fig.suptitle("most-distant flag pairs in DINOv2 space",
                 fontsize=20, fontweight="medium", color=TEXT,
                 x=0.04, y=0.99, ha="left")
    fig.text(0.04, 0.965,
             "Top-10 pairs by cosine distance — visual extremes of the embedding.",
             fontsize=10.5, color=SUBTLE, ha="left")
    for k, (i, j, d) in enumerate(pairs):
        for col, idx in enumerate([i, j]):
            ax = axes[k, col]
            ax.set_facecolor(BG); ax.axis("off")
            thumb = load_thumb(meta.iloc[idx]["iso2"], height_px=80,
                               border_px=2, border_color="#CCCCCC")
            ax.imshow(thumb)
            label = f"{meta.iloc[idx]['name']}"
            ax.text(0.5, -0.08, label, transform=ax.transAxes, fontsize=9,
                    color=TEXT, ha="center", va="top")
    plt.subplots_adjust(left=0.04, right=0.96, top=0.93, bottom=0.02,
                        hspace=0.6, wspace=0.5)
    # Place distance labels in the gap between paired axes (figure coords).
    for k, (_, _, d) in enumerate(pairs):
        bbox_l = axes[k, 0].get_position()
        bbox_r = axes[k, 1].get_position()
        mid_x = (bbox_l.x1 + bbox_r.x0) / 2
        mid_y = (bbox_l.y0 + bbox_l.y1) / 2
        fig.text(mid_x, mid_y, f"{d:.2f}", fontsize=11, color=SUBTLE,
                 ha="center", va="center")
    fig.savefig(OUT_ANALYSIS / "distant_pairs.png", dpi=200, facecolor=BG)
    plt.close(fig)


def fig_cross_neighbors(Xn: np.ndarray, meta: pd.DataFrame, top: int = 10):
    """Least-distant cross-category pairs — surprising cousins."""
    cats = meta["vex_category"].to_numpy()
    n = len(Xn)
    sims = Xn @ Xn.T
    iu = np.triu_indices(n, k=1)
    pair_sims = sims[iu]
    same = cats[iu[0]] == cats[iu[1]]
    pair_sims_diff = np.where(same, -np.inf, pair_sims)
    order = np.argsort(-pair_sims_diff)
    pairs = []
    seen = set()
    for k in order:
        if pair_sims_diff[k] == -np.inf:
            continue
        i, j = iu[0][k], iu[1][k]
        # avoid the same flag appearing in many pairs
        if i in seen or j in seen:
            continue
        pairs.append((i, j, 1 - pair_sims[k]))
        seen.update([i, j])
        if len(pairs) >= top:
            break

    fig, axes = plt.subplots(top, 2, figsize=(9.0, top * 1.6))
    fig.patch.set_facecolor(BG)
    fig.suptitle("nearest cross-category neighbors",
                 fontsize=20, fontweight="medium", color=TEXT,
                 x=0.04, y=0.99, ha="left")
    fig.text(0.04, 0.965,
             "Closest pairs (cosine) where the two flags belong to different "
             "vexillological categories — visual cousins across symbol families.",
             fontsize=10.5, color=SUBTLE, ha="left")

    for k, (i, j, d) in enumerate(pairs):
        for col, idx in enumerate([i, j]):
            ax = axes[k, col]
            ax.set_facecolor(BG); ax.axis("off")
            cat = cats[idx]
            border = CATEGORY_COLORS.get(cat, "#CCCCCC")
            thumb = load_thumb(meta.iloc[idx]["iso2"], height_px=80,
                               border_px=2, border_color=border)
            ax.imshow(thumb)
            label = (f"{meta.iloc[idx]['name']}\n"
                     f"{NICE_NAME.get(cat, cat)}")
            ax.text(0.5, -0.08, label, transform=ax.transAxes, fontsize=8.5,
                    color=TEXT, ha="center", va="top", linespacing=1.3)
    plt.subplots_adjust(left=0.04, right=0.96, top=0.93, bottom=0.02,
                        hspace=1.05, wspace=0.5)
    for k, (_, _, d) in enumerate(pairs):
        bbox_l = axes[k, 0].get_position()
        bbox_r = axes[k, 1].get_position()
        mid_x = (bbox_l.x1 + bbox_r.x0) / 2
        mid_y = (bbox_l.y0 + bbox_l.y1) / 2
        fig.text(mid_x, mid_y, f"{d:.2f}", fontsize=11, color=SUBTLE,
                 ha="center", va="center")
    fig.savefig(OUT_ANALYSIS / "cross_neighbors.png", dpi=200, facecolor=BG)
    plt.close(fig)


def fig_lof_outliers(Xn: np.ndarray, meta: pd.DataFrame, top: int = 16):
    """Most "unique" flags by Local Outlier Factor."""
    lof = LocalOutlierFactor(n_neighbors=15, metric="cosine")
    lof.fit(Xn)
    scores = -lof.negative_outlier_factor_  # higher = more outlier
    order = np.argsort(-scores)[:top]

    cols = 4
    rows = int(np.ceil(top / cols))
    fig = plt.figure(figsize=(3.2 * cols, 2.6 * rows + 1.4))
    fig.patch.set_facecolor(BG)
    fig.suptitle("most visually unique flags (LOF)",
                 fontsize=22, fontweight="medium", color=TEXT,
                 x=0.04, y=0.97, ha="left")
    fig.text(0.04, 0.93,
             f"Top-{top} flags by Local Outlier Factor (k=15, cosine) on DINOv2 features. "
             "Higher score = farther from typical flag neighborhood.",
             fontsize=11, color=SUBTLE, ha="left")
    for k, idx in enumerate(order):
        ax = fig.add_subplot(rows, cols, k + 1)
        ax.set_facecolor(BG); ax.axis("off")
        cat = meta.iloc[idx]["vex_category"]
        thumb = load_thumb(meta.iloc[idx]["iso2"], height_px=120,
                           border_color=CATEGORY_COLORS.get(cat),
                           border_px=3)
        ax.imshow(thumb)
        ax.set_title(meta.iloc[idx]["name"], fontsize=11, color=TEXT,
                     loc="left", pad=4, fontweight="medium")
        ax.text(0.0, -0.05, f"LOF {scores[idx]:.2f}",
                transform=ax.transAxes, fontsize=9, color=SUBTLE,
                ha="left", va="top")
    plt.subplots_adjust(left=0.03, right=0.97, top=0.88, bottom=0.04,
                        hspace=0.45, wspace=0.2)
    fig.savefig(OUT_ANALYSIS / "lof_outliers.png", dpi=200, facecolor=BG)
    plt.close(fig)


def color_count(iso2: str, threshold: float = 0.01, q: int = 4) -> int:
    """Quantize colors to qxqxq cube and count bins above threshold."""
    img = Image.open(PNG_DIR / f"{iso2}.png").convert("RGBA")
    rgb = np.asarray(img.convert("RGB"))
    alpha = np.asarray(img.split()[-1])
    mask = alpha > 200
    rgb = rgb[mask]
    if len(rgb) == 0:
        return 0
    bins = (rgb // (256 // q)).astype(np.int32)
    keys = bins[:, 0] * q * q + bins[:, 1] * q + bins[:, 2]
    counts = np.bincount(keys, minlength=q * q * q)
    frac = counts / counts.sum()
    return int((frac >= threshold).sum())


def symmetry_score(iso2: str) -> tuple[float, float]:
    """SSIM-like score for horizontal and vertical mirror symmetry."""
    from PIL import ImageOps
    img = Image.open(PNG_DIR / f"{iso2}.png").convert("RGB")
    arr = np.asarray(img, dtype=np.float32)
    arr_h = np.asarray(ImageOps.mirror(img), dtype=np.float32)
    arr_v = np.asarray(ImageOps.flip(img), dtype=np.float32)
    h = 1.0 - np.abs(arr - arr_h).mean() / 255.0
    v = 1.0 - np.abs(arr - arr_v).mean() / 255.0
    return float(h), float(v)


def fig_color_count_radius(Xn: np.ndarray, meta: pd.DataFrame):
    print("  computing color counts ...")
    counts = np.array([color_count(i) for i in meta["iso2"]])
    centroid = Xn.mean(axis=0)
    centroid /= np.linalg.norm(centroid) + 1e-9
    radius = 1 - Xn @ centroid

    fig, ax = plt.subplots(figsize=(11, 7))
    fig.patch.set_facecolor(BG)
    setup_axes(ax,
               "color count vs distance from global centroid",
               "FIAV principle: 'simple flags are good flags' — does DINOv2 push complex flags to the edge?")

    cats = meta["vex_category"].values
    for cat, color in CATEGORY_COLORS.items():
        mask = cats == cat
        if mask.sum() == 0:
            continue
        ax.scatter(counts[mask], radius[mask], s=42, alpha=0.75,
                   color=color, edgecolor=BG, linewidth=0.6, zorder=2,
                   label=NICE_NAME.get(cat, cat) if mask.sum() >= 5 else None)
    # global trend line (with slight jitter on x to combat overlap)
    from numpy.polynomial import Polynomial
    coef = np.polyfit(counts, radius, 1)
    xs = np.linspace(counts.min(), counts.max(), 50)
    ax.plot(xs, np.polyval(coef, xs), color=SUBTLE, linewidth=1.2,
            alpha=0.6, linestyle="--", zorder=1, label=f"linear fit (slope {coef[0]:+.3f})")
    ax.set_xlabel("distinct quantized colors (4-bin per channel, ≥1% pixels)",
                  color=SUBTLE, fontsize=10.5)
    ax.set_ylabel("cosine distance from global centroid", color=SUBTLE, fontsize=10.5)
    ax.legend(frameon=False, fontsize=9, labelcolor=TEXT, loc="upper left",
              ncol=2)
    plt.tight_layout()
    fig.savefig(OUT_ANALYSIS / "color_count_radius.png", dpi=200, facecolor=BG)
    plt.close(fig)


def fig_symmetry_scatter(meta: pd.DataFrame):
    print("  computing symmetry scores ...")
    syms = np.array([symmetry_score(i) for i in meta["iso2"]])
    df = pd.read_parquet(OUT_DIR.parent / "data" / "projections" / "projections.parquet")
    df = df.set_index("iso2").loc[meta["iso2"]].reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(15, 8.5))
    fig.patch.set_facecolor(BG)
    for ax, axis_name, idx, label in [
        (axes[0], "horizontal mirror", 0, "score (1.0 = symmetric)"),
        (axes[1], "vertical mirror",   1, "score (1.0 = symmetric)"),
    ]:
        ax.set_facecolor(BG)
        for spine in ax.spines.values():
            spine.set_visible(False)
        sc = ax.scatter(df["tsne_x"], df["tsne_y"], c=syms[:, idx],
                        cmap="viridis", s=70, edgecolor=BG, linewidth=0.6,
                        vmin=0.55, vmax=1.0)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(axis_name, fontsize=14, color=TEXT, fontweight="medium",
                     loc="left", pad=8)
        cb = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
        cb.outline.set_visible(False)
        cb.ax.tick_params(colors=SUBTLE, length=0)
        cb.set_label(label, color=SUBTLE, fontsize=10)

    fig.suptitle("symmetry vs DINOv2 t-SNE position",
                 fontsize=22, fontweight="medium", color=TEXT,
                 x=0.04, y=0.975, ha="left")
    fig.text(0.04, 0.92,
             "Each point colored by mirror-symmetry score (1 − mean abs pixel diff with mirror image). "
             "Tricolors and crosses score high; canton-based and asymmetric designs score low.",
             fontsize=11, color=SUBTLE, ha="left")
    plt.subplots_adjust(left=0.03, right=0.97, top=0.84, bottom=0.04, wspace=0.06)
    fig.savefig(OUT_ANALYSIS / "symmetry_scatter.png", dpi=200, facecolor=BG)
    plt.close(fig)


def main():
    configure_typography()
    OUT_ANALYSIS.mkdir(parents=True, exist_ok=True)
    X, Xn, iso2, meta = load_data()
    print(f"loaded {X.shape}")

    print("centroid flags ...");      fig_centroid_flags(Xn, meta)
    print("distant pairs ...");       fig_distant_pairs(Xn, meta)
    print("cross-cat neighbors ..."); fig_cross_neighbors(Xn, meta)
    print("LOF outliers ...");        fig_lof_outliers(Xn, meta)
    print("color count vs radius ...");  fig_color_count_radius(Xn, meta)
    print("symmetry scatter ...");    fig_symmetry_scatter(meta)
    print("done")


if __name__ == "__main__":
    main()
