"""One large figure per projection (PCA, t-SNE, PHATE)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.offsetbox import AnnotationBbox, OffsetImage

from flag2vec.style import (
    BG, CATEGORY_COLORS, NICE_NAME, OUT_DIR, PANELS, PROJ_DIR,
    SUBTLE, TEXT, category_compactness, clean_axes, configure_typography,
    load_thumb, set_axis_limits, soft_hull,
)

HIGHLIGHT_ORDER = [
    "nordic_cross", "british_ensign", "pan_arab", "star_crescent",
    "pan_african", "vertical_tricolor", "communist_red", "stars_stripes",
    "latin_charge", "pan_slavic",
]


def render_one(df: pd.DataFrame, title: str, xcol: str, ycol: str,
               sub: str, out_path: Path, thumb_height: int = 60,
               compact_threshold: float = 0.65):
    fig, ax = plt.subplots(figsize=(18, 14))
    fig.patch.set_facecolor(BG)
    clean_axes(ax)

    all_pts = df[[xcol, ycol]].to_numpy()
    for cat in HIGHLIGHT_ORDER:
        sub_df = df[df["vex_category"] == cat]
        if len(sub_df) < 3:
            continue
        pts = sub_df[[xcol, ycol]].to_numpy()
        if category_compactness(pts, all_pts) < compact_threshold:
            soft_hull(ax, pts, CATEGORY_COLORS[cat], alpha_fill=0.10, alpha_edge=0.32)

    set_axis_limits(ax, all_pts[:, 0], all_pts[:, 1], pad_frac=0.06)

    for _, row in df.iterrows():
        cat = row["vex_category"]
        border = CATEGORY_COLORS.get(cat) if cat in HIGHLIGHT_ORDER else None
        thumb = load_thumb(row["iso2"], thumb_height, border, border_px=3)
        oi = OffsetImage(thumb, zoom=1.0, interpolation="lanczos")
        ax.add_artist(AnnotationBbox(
            oi, (row[xcol], row[ycol]),
            frameon=False, pad=0, box_alignment=(0.5, 0.5), zorder=3,
        ))

    fig.suptitle(f"flag2vec — {title}", fontsize=28, fontweight="medium",
                 color=TEXT, x=0.04, y=0.965, ha="left")
    fig.text(0.04, 0.928,
             f"DINOv2 embeddings of 197 sovereign flags, projected with {title}.  "
             f"{sub}.  Borders encode hand-curated vexillological categories.",
             fontsize=12, color=SUBTLE, ha="left")

    handles = []
    labels = []
    for cat in HIGHLIGHT_ORDER:
        n = int((df["vex_category"] == cat).sum())
        if n == 0:
            continue
        patch = plt.Rectangle((0, 0), 1, 1, facecolor="none",
                              edgecolor=CATEGORY_COLORS[cat], linewidth=1.6)
        handles.append(patch)
        labels.append(f"{NICE_NAME[cat]}  ({n})")
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False,
               fontsize=10.5, labelcolor=TEXT, bbox_to_anchor=(0.5, 0.018),
               handlelength=1.4, handleheight=0.95, columnspacing=2.4)

    plt.subplots_adjust(left=0.025, right=0.975, top=0.88, bottom=0.07)
    fig.savefig(out_path, dpi=200, facecolor=BG)
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    configure_typography()
    out_dir = OUT_DIR / "projections"
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(PROJ_DIR / "projections.parquet")
    for title, xcol, ycol, sub in PANELS:
        slug = title.lower().replace("-", "")
        render_one(df, title, xcol, ycol, sub, out_dir / f"{slug}.png")


if __name__ == "__main__":
    main()
