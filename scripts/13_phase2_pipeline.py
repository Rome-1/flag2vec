"""Phase 2 pipeline: rasterize subdivisions, embed jointly with sovereigns,
project, render hero + per-country figures.
"""
from __future__ import annotations

import csv
import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cairosvg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import timm
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize
from timm.data import create_transform, resolve_data_config

from flag2vec.style import (
    BG, CATEGORY_COLORS, NICE_NAME, OUT_DIR, PNG_DIR, SUBTLE, TEXT,
    category_compactness, clean_axes, configure_typography, load_thumb,
    set_axis_limits, soft_hull,
)

ROOT = Path(__file__).resolve().parent.parent
SVG_DIR = ROOT / "data" / "raw_svg"
EMB_DIR = ROOT / "data" / "embeddings"
PROJ_DIR = ROOT / "data" / "projections"
SUB_CSV = ROOT / "data" / "subdivision_flags.csv"
SOV_CSV = ROOT / "data" / "sovereign_flags.csv"
ALL_CSV = ROOT / "data" / "all_flags.csv"

CANVAS = (480, 320)
INPUT_SIZE = 518


def build_unified_csv():
    sov = pd.read_csv(SOV_CSV)
    sov["kind"] = "sovereign"
    sov["parent"] = ""
    sub = pd.read_csv(SUB_CSV)
    df = pd.concat([sov[["iso2", "name", "vex_category", "kind", "parent"]],
                    sub[["iso2", "name", "vex_category", "kind", "parent"]]],
                   ignore_index=True)
    df.to_csv(ALL_CSV, index=False)
    print(f"unified CSV: {len(df)} rows -> {ALL_CSV}")
    return df


def rasterize_subdivisions():
    df = pd.read_csv(SUB_CSV)
    out_dir = PNG_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    failed = []
    for _, row in df.iterrows():
        code = row["iso2"]
        out_png = out_dir / f"{code}.png"
        if out_png.exists():
            continue
        svg_path = SVG_DIR / f"{code}.svg"
        if not svg_path.exists():
            failed.append((code, "missing svg"))
            continue
        try:
            png = cairosvg.svg2png(url=str(svg_path), output_width=CANVAS[0])
            img = Image.open(io.BytesIO(png)).convert("RGBA")
            if img.height > CANVAS[1]:
                new_w = int(round(img.width * CANVAS[1] / img.height))
                img = img.resize((new_w, CANVAS[1]), Image.LANCZOS)
            canvas = Image.new("RGBA", CANVAS, (0, 0, 0, 0))
            x = (CANVAS[0] - img.width) // 2
            y = (CANVAS[1] - img.height) // 2
            canvas.paste(img, (x, y), img)
            canvas.save(out_png, optimize=True)
        except Exception as e:
            failed.append((code, str(e)))
    print(f"rasterized {len(df) - len(failed)}/{len(df)} subdivisions")
    if failed:
        print("FAILURES:")
        for c, e in failed[:10]:
            print(f"  {c}: {e}")


def embed_all(force: bool = False):
    df = build_unified_csv()
    out_npy = EMB_DIR / "dinov2_all.npy"
    out_order = EMB_DIR / "iso2_order_all.txt"
    if out_npy.exists() and out_order.exists() and not force:
        print(f"using cached {out_npy}")
        return df

    model = timm.create_model("vit_small_patch14_dinov2.lvd142m",
                              pretrained=True, num_classes=0).eval()
    cfg = resolve_data_config({}, model=model)
    cfg["input_size"] = (3, INPUT_SIZE, INPUT_SIZE)
    transform = create_transform(**cfg, is_training=False)

    embeddings = []
    iso2_codes = []
    with torch.inference_mode():
        for _, row in df.iterrows():
            code = row["iso2"]
            png_path = PNG_DIR / f"{code}.png"
            if not png_path.exists():
                continue
            img = Image.open(png_path).convert("RGBA")
            scale = min(INPUT_SIZE / img.width, INPUT_SIZE / img.height)
            new_w = max(1, int(round(img.width * scale)))
            new_h = max(1, int(round(img.height * scale)))
            resized = img.resize((new_w, new_h), Image.LANCZOS)
            canvas = Image.new("RGB", (INPUT_SIZE, INPUT_SIZE), (255, 255, 255))
            canvas.paste(resized, ((INPUT_SIZE - new_w) // 2,
                                   (INPUT_SIZE - new_h) // 2), resized)
            t = transform(canvas).unsqueeze(0)
            feats = model.forward_features(t)
            if isinstance(feats, dict):
                cls = feats.get("x_norm_clstoken", feats.get("cls_token"))
            else:
                cls = feats[:, 0] if feats.ndim == 3 else feats
            embeddings.append(cls.squeeze(0).cpu().numpy().astype(np.float32))
            iso2_codes.append(code)

    arr = np.stack(embeddings, axis=0)
    np.save(out_npy, arr)
    out_order.write_text("\n".join(iso2_codes) + "\n")
    print(f"saved {arr.shape} -> {out_npy}")
    return df


def project_all():
    X = np.load(EMB_DIR / "dinov2_all.npy")
    iso2 = (EMB_DIR / "iso2_order_all.txt").read_text().strip().splitlines()
    Xn = normalize(X, norm="l2")
    print(f"projecting {Xn.shape}")

    pca = PCA(n_components=2, random_state=0).fit_transform(Xn)
    print(f"  PCA done")
    tsne = TSNE(n_components=2, perplexity=30, metric="cosine",
                init="pca", learning_rate="auto",
                random_state=0).fit_transform(Xn)
    print(f"  t-SNE done")
    import phate
    phate_xy = phate.PHATE(n_components=2, knn=15, decay=20, t="auto",
                           random_state=0, verbose=0).fit_transform(Xn)
    print(f"  PHATE done")

    df_meta = pd.read_csv(ALL_CSV).set_index("iso2").loc[iso2].reset_index()
    df = pd.DataFrame({
        "iso2": iso2,
        "name": df_meta["name"].values,
        "vex_category": df_meta["vex_category"].values,
        "kind": df_meta["kind"].values,
        "parent": df_meta["parent"].fillna("").values,
        "pca_x": pca[:, 0], "pca_y": pca[:, 1],
        "tsne_x": tsne[:, 0], "tsne_y": tsne[:, 1],
        "phate_x": phate_xy[:, 0], "phate_y": phate_xy[:, 1],
    })
    df.to_parquet(PROJ_DIR / "projections_all.parquet")
    print(f"saved -> projections_all.parquet")


# ───────────── Render: joint hero figure (sovereigns + subdivisions) ──────

def render_joint_hero():
    df = pd.read_parquet(PROJ_DIR / "projections_all.parquet")

    PANELS = [
        ("PCA",   "pca_x",   "pca_y",   "global axes"),
        ("t-SNE", "tsne_x",  "tsne_y",  "perplexity 30, cosine"),
        ("PHATE", "phate_x", "phate_y", "knn 15, decay 20"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(34, 14))
    fig.patch.set_facecolor(BG)

    for ax, (title, xcol, ycol, sub) in zip(axes, PANELS):
        clean_axes(ax)
        all_pts = df[[xcol, ycol]].to_numpy()
        set_axis_limits(ax, all_pts[:, 0], all_pts[:, 1], pad_frac=0.05)

        # Subdivisions = small thumbnails (faded slightly).
        # Sovereigns = larger thumbnails (full opacity), drawn on top.
        for _, row in df.iterrows():
            is_sov = row["kind"] == "sovereign"
            cat = row["vex_category"]
            border = CATEGORY_COLORS.get(cat) if cat in CATEGORY_COLORS else "#888888"
            h = 30 if is_sov else 22
            thumb = load_thumb(row["iso2"], h, border, border_px=2)
            ax.add_artist(AnnotationBbox(
                OffsetImage(thumb, zoom=1.0, interpolation="lanczos"),
                (row[xcol], row[ycol]),
                frameon=False, pad=0, box_alignment=(0.5, 0.5),
                zorder=4 if is_sov else 2,
            ))

        ax.text(0.0, 1.02, title, transform=ax.transAxes,
                fontsize=15, color=TEXT, fontweight="medium", ha="left", va="bottom")
        ax.text(1.0, 1.02, sub, transform=ax.transAxes,
                fontsize=10, color=SUBTLE, ha="right", va="bottom")

    n_sov = (df["kind"] == "sovereign").sum()
    n_sub = (df["kind"] == "subdivision").sum()
    fig.suptitle("flag2vec — Phase 2",
                 fontsize=32, fontweight="medium", color=TEXT,
                 x=0.025, y=0.965, ha="left")
    fig.text(0.025, 0.92,
             f"DINOv2 visual embeddings of {n_sov} sovereign flags + {n_sub} subdivision flags "
             "(US states, Japanese prefectures, Swiss cantons, German Länder, Australian states, "
             "Canadian provinces, Brazilian states, UK constituent + Greenland/Faroe/Åland). "
             "Sovereigns drawn larger on top; subdivisions slightly smaller.",
             fontsize=12, color=SUBTLE, ha="left")
    plt.subplots_adjust(left=0.012, right=0.988, top=0.88,
                        bottom=0.03, wspace=0.04)

    out_dir = OUT_DIR / "phase2"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "hero.png", dpi=180, facecolor=BG)
    plt.close(fig)
    print(f"wrote {out_dir / 'hero.png'}")


# ───────────── Render: per-country subdivision highlight figures ──────────

PARENT_NICE = {
    "US": "United States",
    "CA": "Canada",
    "JP": "Japan",
    "CH": "Switzerland",
    "DE": "Germany",
    "AU": "Australia",
    "BR": "Brazil",
    "GB": "United Kingdom (constituent)",
    "AX": "Åland Islands",
    "FO": "Faroe Islands",
    "GL": "Greenland",
}


def render_per_country_subdivisions():
    df = pd.read_parquet(PROJ_DIR / "projections_all.parquet")
    out_dir = OUT_DIR / "phase2" / "subdivisions_by_country"
    out_dir.mkdir(parents=True, exist_ok=True)

    sub_df = df[df["kind"] == "subdivision"].copy()
    parents_to_show = ["US", "JP", "CH", "DE", "BR", "AU", "CA"]

    PANELS = [
        ("PCA",   "pca_x",   "pca_y"),
        ("t-SNE", "tsne_x",  "tsne_y"),
        ("PHATE", "phate_x", "phate_y"),
    ]

    for parent in parents_to_show:
        parent_subs = sub_df[sub_df["parent"] == parent]
        n = len(parent_subs)
        if n == 0:
            continue
        # Find the parent sovereign in df (lowercase iso2 for sovereigns)
        sov_iso2 = parent.lower()
        if parent == "GB":
            sov_iso2 = "gb"
        sov_row = df[(df["kind"] == "sovereign") & (df["iso2"] == sov_iso2)]

        fig, axes = plt.subplots(1, 3, figsize=(28, 11))
        fig.patch.set_facecolor(BG)
        fig.suptitle(PARENT_NICE.get(parent, parent),
                     fontsize=26, fontweight="medium", color=TEXT,
                     x=0.025, y=0.965, ha="left")
        fig.text(0.025, 0.92,
                 f"{n} subdivisions highlighted; the national flag of {PARENT_NICE.get(parent, parent)} "
                 "is drawn even larger with a gold border.",
                 fontsize=11, color=SUBTLE, ha="left")

        for ax, (title, xcol, ycol) in zip(axes, PANELS):
            clean_axes(ax)
            all_pts = df[[xcol, ycol]].to_numpy()
            set_axis_limits(ax, all_pts[:, 0], all_pts[:, 1], pad_frac=0.05)

            # Highlight cluster hull
            pts = parent_subs[[xcol, ycol]].to_numpy()
            color = CATEGORY_COLORS.get(parent_subs["vex_category"].iloc[0], "#3E7CB1")
            soft_hull(ax, pts, color, alpha_fill=0.12, alpha_edge=0.40)
            compact = category_compactness(pts, all_pts)

            # Draw all flags faded except the highlighted parent's subs
            highlight_codes = set(parent_subs["iso2"].tolist())
            for _, row in df.iterrows():
                is_hl = row["iso2"] in highlight_codes
                is_sov_parent = row["iso2"] == sov_iso2 and row["kind"] == "sovereign"
                if is_sov_parent:
                    h = 50; border = "#C08A3E"; faded = False; gray = False; bw = 4
                elif is_hl:
                    h = 30
                    border = CATEGORY_COLORS.get(row["vex_category"], color)
                    faded = False; gray = False; bw = 3
                else:
                    h = 18; border = None; faded = True; gray = True; bw = 2
                thumb = load_thumb(row["iso2"], h, border, border_px=bw,
                                   faded=faded, grayscale=gray)
                ax.add_artist(AnnotationBbox(
                    OffsetImage(thumb, zoom=1.0, interpolation="lanczos"),
                    (row[xcol], row[ycol]),
                    frameon=False, pad=0, box_alignment=(0.5, 0.5),
                    zorder=5 if is_sov_parent else (4 if is_hl else 2),
                ))

            ax.text(0.0, 1.02, title, transform=ax.transAxes,
                    fontsize=14, color=TEXT, fontweight="medium",
                    ha="left", va="bottom")
            ax.text(1.0, 1.02, f"compactness {compact:.2f}",
                    transform=ax.transAxes, fontsize=10, color=SUBTLE,
                    ha="right", va="bottom")

        plt.subplots_adjust(left=0.012, right=0.988, top=0.86,
                            bottom=0.03, wspace=0.04)
        out_path = out_dir / f"{parent}.png"
        fig.savefig(out_path, dpi=170, facecolor=BG)
        plt.close(fig)
        print(f"wrote {out_path}  (n={n})")


def main():
    configure_typography()
    print("=== rasterize subdivisions ===")
    rasterize_subdivisions()
    print("=== embed all ===")
    embed_all()
    print("=== project ===")
    project_all()
    print("=== render joint hero ===")
    render_joint_hero()
    print("=== render per-country ===")
    render_per_country_subdivisions()
    print("done")


if __name__ == "__main__":
    main()
