"""One figure per region — highlight that region, fade the rest."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.offsetbox import AnnotationBbox, OffsetImage

from flag2vec.style import (
    BG, OUT_DIR, PANELS, PROJ_DIR, REGION_COLORS, REGION_CSV,
    SUBTLE, TEXT, category_compactness, clean_axes, configure_typography,
    load_thumb, set_axis_limits, soft_hull,
)


def render_panel(ax, df: pd.DataFrame, xcol: str, ycol: str, title: str,
                 sub: str, region: str, thumb_height: int = 30):
    clean_axes(ax)
    all_pts = df[[xcol, ycol]].to_numpy()

    sub_df = df[df["region"] == region]
    color = REGION_COLORS.get(region, "#777777")
    if len(sub_df) >= 3:
        pts = sub_df[[xcol, ycol]].to_numpy()
        compact = category_compactness(pts, all_pts)
        soft_hull(ax, pts, color, alpha_fill=0.12, alpha_edge=0.40)
    else:
        compact = float("nan")

    set_axis_limits(ax, all_pts[:, 0], all_pts[:, 1])

    for _, row in df.iterrows():
        is_hl = row["region"] == region
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
    out_dir = OUT_DIR / "regions"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(PROJ_DIR / "projections.parquet")
    regions = pd.read_csv(REGION_CSV)
    df = df.merge(regions, on="iso2")

    for region in ["Africa", "Americas", "Asia", "Europe", "Oceania"]:
        n = int((df["region"] == region).sum())
        if n == 0:
            continue
        fig, axes = plt.subplots(1, 3, figsize=(28, 11))
        fig.patch.set_facecolor(BG)
        for ax, (title, xcol, ycol, sub) in zip(axes, PANELS):
            render_panel(ax, df, xcol, ycol, title, sub, region)

        fig.suptitle(region, fontsize=28, fontweight="medium",
                     color=TEXT, x=0.025, y=0.96, ha="left")
        fig.text(0.025, 0.918,
                 f"{n} flags from {region}, highlighted across PCA / t-SNE / PHATE.  "
                 "Compactness = mean within-region pairwise distance ÷ mean global pairwise distance.",
                 fontsize=11.5, color=SUBTLE, ha="left")
        plt.subplots_adjust(left=0.012, right=0.988, top=0.86,
                            bottom=0.04, wspace=0.04)
        out_path = out_dir / f"{region.lower().replace(' ', '_')}.png"
        fig.savefig(out_path, dpi=180, facecolor=BG)
        plt.close(fig)
        print(f"wrote {out_path}  ({n} flags)")


if __name__ == "__main__":
    main()
