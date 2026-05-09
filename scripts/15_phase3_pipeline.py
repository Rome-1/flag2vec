"""Phase 3 pipeline: rasterize sovereigns + subdivisions + historical flags,
embed jointly via DINOv2, project to PCA / t-SNE / PHATE, save.

Outputs:
  data/png/<iso2>.png                    (sovereigns + subdivisions)
  data/png/hist-<code>.png               (historical)
  data/all_flags_phase3.csv              (joint metadata)
  data/embeddings/dinov2_phase3.npy      (joint embeddings)
  data/embeddings/iso2_order_phase3.txt
  data/projections/projections_all_phase3.parquet

Re-uses rasterize_subdivisions logic from scripts/13_phase2_pipeline.py.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cairosvg
import numpy as np
import pandas as pd
import torch
import timm
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize
from timm.data import create_transform, resolve_data_config

ROOT = Path(__file__).resolve().parent.parent
SVG_DIR = ROOT / "data" / "raw_svg"
PNG_DIR = ROOT / "data" / "png"
EMB_DIR = ROOT / "data" / "embeddings"
PROJ_DIR = ROOT / "data" / "projections"

SOV_CSV  = ROOT / "data" / "sovereign_flags.csv"
SUB_CSV  = ROOT / "data" / "subdivision_flags.csv"
HIST_CSV = ROOT / "data" / "historical_flags.csv"
ALL3_CSV = ROOT / "data" / "all_flags_phase3.csv"

CANVAS = (480, 320)
INPUT_SIZE = 518


def rasterize_one(svg_path: Path, out_png: Path) -> None:
    png = cairosvg.svg2png(url=str(svg_path), output_width=CANVAS[0])
    img = Image.open(io.BytesIO(png)).convert("RGBA")
    if img.height > CANVAS[1]:
        new_w = int(round(img.width * CANVAS[1] / img.height))
        img = img.resize((new_w, CANVAS[1]), Image.LANCZOS)
    canvas = Image.new("RGBA", CANVAS, (0, 0, 0, 0))
    x = (CANVAS[0] - img.width) // 2
    y = (CANVAS[1] - img.height) // 2
    canvas.paste(img, (x, y), img)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_png, optimize=True)


def rasterize_set(df: pd.DataFrame, label: str) -> None:
    failed = []
    n_done = 0
    for code in df["iso2"]:
        out_png = PNG_DIR / f"{code}.png"
        if out_png.exists():
            continue
        svg_path = SVG_DIR / f"{code}.svg"
        if not svg_path.exists():
            failed.append((code, "missing svg"))
            continue
        try:
            rasterize_one(svg_path, out_png)
            n_done += 1
        except Exception as e:
            failed.append((code, str(e)))
    print(f"  rasterized {n_done} {label} flags ({len(failed)} failures)")
    for c, e in failed[:10]:
        print(f"    FAIL {c}: {e}")


def build_unified_csv() -> pd.DataFrame:
    sov = pd.read_csv(SOV_CSV).copy()
    sov["kind"] = "sovereign"
    sov["parent"] = ""
    sov["successor_iso2"] = ""
    sov["era_start"] = pd.NA
    sov["era_end"] = pd.NA

    sub = pd.read_csv(SUB_CSV).copy()
    # subdivision csv already has kind=subdivision and parent
    sub["successor_iso2"] = ""
    sub["era_start"] = pd.NA
    sub["era_end"] = pd.NA

    hist = pd.read_csv(HIST_CSV).copy()
    hist["parent"] = ""

    cols = ["iso2", "name", "vex_category", "kind", "parent",
            "successor_iso2", "era_start", "era_end"]
    df = pd.concat([sov[cols], sub[cols], hist[cols]], ignore_index=True)
    df.to_csv(ALL3_CSV, index=False)
    print(f"  unified CSV: {len(df)} rows ({(df.kind == 'sovereign').sum()} sov "
          f"+ {(df.kind == 'subdivision').sum()} sub "
          f"+ {(df.kind == 'historical').sum()} hist) -> {ALL3_CSV}")
    return df


def embed_all(df: pd.DataFrame, force: bool = False) -> tuple[np.ndarray, list[str]]:
    out_npy = EMB_DIR / "dinov2_phase3.npy"
    out_order = EMB_DIR / "iso2_order_phase3.txt"
    EMB_DIR.mkdir(parents=True, exist_ok=True)

    if out_npy.exists() and out_order.exists() and not force:
        arr = np.load(out_npy)
        order = out_order.read_text().strip().splitlines()
        if len(order) == len(df):
            print(f"  using cached {out_npy} ({arr.shape})")
            return arr, order

    print("  loading DINOv2 ViT-S/14...")
    model = timm.create_model("vit_small_patch14_dinov2.lvd142m",
                              pretrained=True, num_classes=0).eval()
    cfg = resolve_data_config({}, model=model)
    cfg["input_size"] = (3, INPUT_SIZE, INPUT_SIZE)
    transform = create_transform(**cfg, is_training=False)

    embeddings = []
    iso2_codes = []
    with torch.inference_mode():
        for i, row in df.iterrows():
            code = row["iso2"]
            png_path = PNG_DIR / f"{code}.png"
            if not png_path.exists():
                print(f"    skip {code} (no PNG)")
                continue
            img = Image.open(png_path).convert("RGBA")
            scale = min(INPUT_SIZE / img.width, INPUT_SIZE / img.height)
            new_w = max(1, int(round(img.width * scale)))
            new_h = max(1, int(round(img.height * scale)))
            resized = img.resize((new_w, new_h), Image.LANCZOS)
            canvas = Image.new("RGB", (INPUT_SIZE, INPUT_SIZE), (255, 255, 255))
            canvas.paste(resized,
                         ((INPUT_SIZE - new_w) // 2, (INPUT_SIZE - new_h) // 2),
                         resized)
            t = transform(canvas).unsqueeze(0)
            feats = model.forward_features(t)
            if isinstance(feats, dict):
                cls = feats.get("x_norm_clstoken", feats.get("cls_token"))
            else:
                cls = feats[:, 0] if feats.ndim == 3 else feats
            embeddings.append(cls.squeeze(0).cpu().numpy().astype(np.float32))
            iso2_codes.append(code)
            if (i + 1) % 50 == 0:
                print(f"    embedded {i + 1}/{len(df)}")

    arr = np.stack(embeddings, axis=0)
    np.save(out_npy, arr)
    out_order.write_text("\n".join(iso2_codes) + "\n")
    print(f"  saved {arr.shape} -> {out_npy}")
    return arr, iso2_codes


def project(arr: np.ndarray, iso2_order: list[str], df: pd.DataFrame) -> pd.DataFrame:
    Xn = normalize(arr, norm="l2")
    print(f"  projecting {Xn.shape}")

    pca = PCA(n_components=2, random_state=0).fit_transform(Xn)
    print("  PCA done")
    tsne = TSNE(n_components=2, perplexity=30, metric="cosine",
                init="pca", learning_rate="auto",
                random_state=0).fit_transform(Xn)
    print("  t-SNE done")
    import phate
    phate_xy = phate.PHATE(n_components=2, knn=15, decay=20, t="auto",
                           random_state=0, verbose=0).fit_transform(Xn)
    print("  PHATE done")

    meta = df.set_index("iso2").loc[iso2_order].reset_index()
    out = pd.DataFrame({
        "iso2": iso2_order,
        "name": meta["name"].values,
        "vex_category": meta["vex_category"].values,
        "kind": meta["kind"].values,
        "parent": meta["parent"].fillna("").values,
        "successor_iso2": meta["successor_iso2"].fillna("").values,
        "era_start": meta["era_start"].values,
        "era_end": meta["era_end"].values,
        "pca_x": pca[:, 0], "pca_y": pca[:, 1],
        "tsne_x": tsne[:, 0], "tsne_y": tsne[:, 1],
        "phate_x": phate_xy[:, 0], "phate_y": phate_xy[:, 1],
    })
    PROJ_DIR.mkdir(parents=True, exist_ok=True)
    path = PROJ_DIR / "projections_all_phase3.parquet"
    out.to_parquet(path)
    print(f"  saved -> {path}")
    return out


def main() -> int:
    print("=== Phase 3 pipeline ===")
    print("[1/4] rasterize")
    sov = pd.read_csv(SOV_CSV)
    sub = pd.read_csv(SUB_CSV)
    hist = pd.read_csv(HIST_CSV)
    rasterize_set(sov, "sovereign")
    rasterize_set(sub, "subdivision")
    rasterize_set(hist, "historical")

    print("[2/4] build unified CSV")
    df = build_unified_csv()

    print("[3/4] embed jointly via DINOv2")
    arr, order = embed_all(df)

    print("[4/4] project (PCA / t-SNE / PHATE)")
    project(arr, order, df)

    print("done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
