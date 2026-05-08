"""One figure per vex category — highlight that category, fade the rest.

Each figure has a 3-panel layout (PCA / t-SNE / PHATE) so the reader can
immediately see whether the category clusters in some projections and not
others.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.offsetbox import AnnotationBbox, OffsetImage

from flag2vec.style import (
    BG, CATEGORY_COLORS, NICE_NAME, OUT_DIR, PANELS, PROJ_DIR,
    SUBTLE, TEXT, category_compactness, clean_axes, configure_typography,
    load_thumb, set_axis_limits, soft_hull,
)


def render_panel(ax, df: pd.DataFrame, xcol: str, ycol: str, title: str,
                 sub: str, highlight_cat: str, thumb_height: int = 30):
    clean_axes(ax)
    all_pts = df[[xcol, ycol]].to_numpy()

    sub_df = df[df["vex_category"] == highlight_cat]
    color = CATEGORY_COLORS.get(highlight_cat, "#777777")
    if len(sub_df) >= 3:
        pts = sub_df[[xcol, ycol]].to_numpy()
        compact = category_compactness(pts, all_pts)
        soft_hull(ax, pts, color, alpha_fill=0.14, alpha_edge=0.45)
    else:
        compact = float("nan")

    set_axis_limits(ax, all_pts[:, 0], all_pts[:, 1])

    for _, row in df.iterrows():
        is_hl = row["vex_category"] == highlight_cat
        border = color if is_hl else None
        thumb = load_thumb(
            row["iso2"], thumb_height, border, border_px=3,
            faded=not is_hl, grayscale=not is_hl,
        )
        ax.add_artist(AnnotationBbox(
            OffsetImage(thumb, zoom=1.0, interpolation="lanczos"),
            (row[xcol], row[ycol]),
            frameon=False, pad=0, box_alignment=(0.5, 0.5),
            zorder=4 if is_hl else 2,
        ))

    ax.text(0.0, 1.02, title, transform=ax.transAxes,
            fontsize=14, color=TEXT, fontweight="medium",
            ha="left", va="bottom")
    compact_str = f"compactness {compact:.2f}" if compact == compact else "—"
    ax.text(1.0, 1.02, f"{sub}   ·   {compact_str}",
            transform=ax.transAxes,
            fontsize=9.5, color=SUBTLE, ha="right", va="bottom")


def main():
    configure_typography()
    out_dir = OUT_DIR / "categories"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(PROJ_DIR / "projections.parquet")
    cats = sorted(df["vex_category"].unique(),
                  key=lambda c: -(df["vex_category"] == c).sum())

    for cat in cats:
        n = int((df["vex_category"] == cat).sum())
        fig, axes = plt.subplots(1, 3, figsize=(28, 11))
        fig.patch.set_facecolor(BG)
        for ax, (title, xcol, ycol, sub) in zip(axes, PANELS):
            render_panel(ax, df, xcol, ycol, title, sub, cat)

        nice = NICE_NAME.get(cat, cat)
        fig.suptitle(f"{nice}", fontsize=28, fontweight="medium",
                     color=TEXT, x=0.025, y=0.96, ha="left")
        fig.text(0.025, 0.918,
                 f"{n} flags in this category, highlighted across PCA / t-SNE / PHATE.  "
                 "Compactness = mean within-category pairwise distance ÷ mean global pairwise distance "
                 "(lower = tighter cluster).",
                 fontsize=11.5, color=SUBTLE, ha="left")
        plt.subplots_adjust(left=0.012, right=0.988, top=0.86,
                            bottom=0.04, wspace=0.04)
        out_path = out_dir / f"{cat}.png"
        fig.savefig(out_path, dpi=180, facecolor=BG)
        plt.close(fig)
        print(f"wrote {out_path}  ({n} flags)")


if __name__ == "__main__":
    main()
