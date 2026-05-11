"""Phase 4 — DINOv2 embedding sanity-check for Mars flags.

Self-contained: if `data/embeddings/dinov2_all.npy` doesn't exist, this
script embeds all 398 Earth flags first.  Otherwise it loads the cached
artifact from Phase 2.

For each generated Mars flag in `out/mars/flags/<region_id>.png`:

  1. Embed via DINOv2 ViT-S/14 at 518x518 (same as scripts/03_embed.py).
  2. Concatenate Mars embeddings to the existing all_flags embeddings.
  3. Joint re-projection with PCA / t-SNE / PHATE.
  4. For each Mars flag: nearest Earth tradition centroid (cosine).
     A "hit" = nearest tradition matches the inherited tradition.
     Report per-flag and overall hit rate (chance ≈ 1/15 ≈ 0.067).

Outputs:
  data/embeddings/dinov2_all.npy            (Earth, only if not cached)
  data/embeddings/iso2_order_all.txt        (Earth, only if not cached)
  data/all_flags.csv                        (Earth, only if not cached)
  data/mars_embeddings.npy
  data/mars_distance_table.csv
  data/projections/projections_all_phase4.parquet
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# Set threading BEFORE importing torch to ensure it takes effect.
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import torch
import timm
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize
from timm.data import create_transform, resolve_data_config

torch.set_num_threads(4)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
EMB_DIR = DATA_DIR / "embeddings"
PROJ_DIR = DATA_DIR / "projections"
PNG_DIR = DATA_DIR / "png"
ALL_CSV = DATA_DIR / "all_flags.csv"
SOV_CSV = DATA_DIR / "sovereign_flags.csv"
SUB_CSV = DATA_DIR / "subdivision_flags.csv"
MARS_CSV = DATA_DIR / "mars_regions.csv"
MARS_PNG = ROOT / "out" / "mars" / "flags"

INPUT_SIZE = 224  # 16*14 — Phase 4 uses a smaller input than Phase 1/2 (518) for CPU speed.
                  # The embedding manifold is the same DINOv2 ViT-S/14; the joint Earth+Mars
                  # analysis remains internally consistent because every flag here passes
                  # through the same resolution.  Phase 1/2 figures (out/latent_flags.png etc.)
                  # are unaffected — they keep their 518×518 embeddings under data/embeddings/.
MODEL_NAME = "vit_small_patch14_dinov2.lvd142m"
SOVEREIGNS_ONLY = True  # Phase 4 doesn't need the 201 subdivisions; halves CPU work.


def _letterbox(img: Image.Image, side: int = INPUT_SIZE) -> Image.Image:
    img = img.convert("RGBA")
    scale = min(side / img.width, side / img.height)
    new_w = max(1, int(round(img.width * scale)))
    new_h = max(1, int(round(img.height * scale)))
    resized = img.resize((new_w, new_h), Image.LANCZOS)
    canvas = Image.new("RGB", (side, side), (255, 255, 255))
    canvas.paste(resized, ((side - new_w) // 2, (side - new_h) // 2), resized)
    return canvas


def _build_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device} ; torch threads: {torch.get_num_threads()} ; input {INPUT_SIZE}")
    # Construct DINOv2 ViT-S/14 with the requested input size; positional
    # embeddings are interpolated from the pretrained 518×518 weights.
    model = timm.create_model(
        MODEL_NAME,
        pretrained=True,
        num_classes=0,
        img_size=INPUT_SIZE,
        dynamic_img_size=False,
    )
    model.eval().to(device)
    cfg = resolve_data_config({}, model=model)
    cfg["input_size"] = (3, INPUT_SIZE, INPUT_SIZE)
    transform = create_transform(**cfg, is_training=False)
    return model, transform, device


def _embed_paths(model, transform, device, paths: list[Path]) -> np.ndarray:
    out = []
    n = len(paths)
    t0 = time.time()
    with torch.inference_mode():
        for i, p in enumerate(paths):
            img = _letterbox(Image.open(p))
            t = transform(img).unsqueeze(0).to(device)
            feats = model.forward_features(t)
            if isinstance(feats, dict):
                cls = feats.get("x_norm_clstoken", feats.get("cls_token"))
            else:
                cls = feats[:, 0] if feats.ndim == 3 else feats
            out.append(cls.squeeze(0).cpu().numpy().astype(np.float32))
            if (i + 1) % 25 == 0 or i == n - 1:
                rate = (i + 1) / (time.time() - t0 + 1e-9)
                eta = (n - i - 1) / max(rate, 1e-9)
                print(
                    f"  embed {i + 1:3d}/{n}  "
                    f"({rate:.1f}/s, eta {eta:.0f}s)",
                    flush=True,
                )
    return np.stack(out, axis=0)


def build_unified_csv() -> pd.DataFrame:
    """Return the metadata DataFrame for what we will embed.

    Phase 4 doesn't overwrite the canonical Phase-2 `data/all_flags.csv`;
    if it already exists with the right shape we just load and filter it.
    """
    if ALL_CSV.exists():
        df = pd.read_csv(ALL_CSV)
        if SOVEREIGNS_ONLY:
            df = df[df["kind"] == "sovereign"].reset_index(drop=True)
        return df
    sov = pd.read_csv(SOV_CSV)
    sov["kind"] = "sovereign"
    sov["parent"] = ""
    if SUB_CSV.exists() and not SOVEREIGNS_ONLY:
        sub = pd.read_csv(SUB_CSV)
        df = pd.concat(
            [sov[["iso2", "name", "vex_category", "kind", "parent"]],
             sub[["iso2", "name", "vex_category", "kind", "parent"]]],
            ignore_index=True,
        )
    else:
        df = sov[["iso2", "name", "vex_category", "kind", "parent"]]
    df.to_csv(ALL_CSV, index=False)
    print(f"wrote unified {ALL_CSV}  ({len(df)} rows)")
    return df


def ensure_earth_embeddings() -> tuple[np.ndarray, list[str], pd.DataFrame]:
    EMB_DIR.mkdir(parents=True, exist_ok=True)
    suffix = "_phase4" if INPUT_SIZE != 518 else ""
    npy = EMB_DIR / f"dinov2_all{suffix}.npy"
    order = EMB_DIR / f"iso2_order_all{suffix}.txt"
    if npy.exists() and order.exists() and ALL_CSV.exists():
        print(f"using cached {npy}")
        X = np.load(npy)
        iso2 = order.read_text().strip().splitlines()
        meta = pd.read_csv(ALL_CSV).set_index("iso2").loc[iso2].reset_index()
        return X, iso2, meta

    print("=== embedding Earth flags (no cache) ===")
    df = build_unified_csv()
    paths = []
    iso2_codes = []
    for _, row in df.iterrows():
        p = PNG_DIR / f"{row['iso2']}.png"
        if not p.exists():
            print(f"  skip missing {p}")
            continue
        paths.append(p)
        iso2_codes.append(row["iso2"])

    model, transform, device = _build_model()
    X = _embed_paths(model, transform, device, paths)
    np.save(npy, X)
    order.write_text("\n".join(iso2_codes) + "\n")
    print(f"saved {X.shape} -> {npy}")
    meta = df.set_index("iso2").loc[iso2_codes].reset_index()
    return X, iso2_codes, meta


def embed_mars(mars_df: pd.DataFrame) -> np.ndarray:
    print("=== embedding Mars flags ===")
    paths = [MARS_PNG / f"{row['region_id']}.png" for _, row in mars_df.iterrows()]
    model, transform, device = _build_model()
    X = _embed_paths(model, transform, device, paths)
    return X


def joint_project(X_all: np.ndarray) -> dict[str, np.ndarray]:
    Xn = normalize(X_all, norm="l2")
    print(f"=== projecting {Xn.shape} ===")
    pca = PCA(n_components=2, random_state=0).fit_transform(Xn)
    print("  PCA done")
    tsne = TSNE(
        n_components=2, perplexity=30, metric="cosine",
        init="pca", learning_rate="auto", random_state=0,
    ).fit_transform(Xn)
    print("  t-SNE done")
    import phate
    phate_xy = phate.PHATE(
        n_components=2, knn=15, decay=20, t="auto",
        random_state=0, verbose=0,
    ).fit_transform(Xn)
    print("  PHATE done")
    return {"pca": pca, "tsne": tsne, "phate": phate_xy}


def compute_hit_table(
    mars_df: pd.DataFrame,
    mars_emb: np.ndarray,
    earth_emb: np.ndarray,
    earth_iso2: list[str],
    earth_meta: pd.DataFrame,
) -> pd.DataFrame:
    """Per-Mars-flag: nearest Earth tradition centroid, distance, hit/miss.

    Earth is restricted to sovereigns to keep tradition centroids clean.
    """
    sov_meta = earth_meta[earth_meta["kind"] == "sovereign"].copy()
    keep_iso = set(sov_meta["iso2"])
    sov_idx = [i for i, c in enumerate(earth_iso2) if c in keep_iso]
    sov_emb = earth_emb[sov_idx]
    sov_iso2 = [earth_iso2[i] for i in sov_idx]
    sov_meta_aligned = sov_meta.set_index("iso2").loc[sov_iso2].reset_index()

    sov_n = normalize(sov_emb, norm="l2")
    mars_n = normalize(mars_emb, norm="l2")

    traditions = sorted(sov_meta_aligned["vex_category"].unique())
    centroids = {}
    centroid_iso2 = {}
    for trad in traditions:
        mask = (sov_meta_aligned["vex_category"] == trad).to_numpy()
        if mask.sum() == 0:
            continue
        c = sov_n[mask].mean(axis=0)
        c = c / (np.linalg.norm(c) + 1e-9)
        centroids[trad] = c
        d_to_c = 1 - sov_n[mask] @ c
        nearest_local = int(np.argmin(d_to_c))
        nearest_global = sov_meta_aligned.index[mask][nearest_local]
        centroid_iso2[trad] = sov_meta_aligned.iloc[nearest_global]["iso2"]

    trad_names = list(centroids.keys())
    C = np.stack([centroids[t] for t in trad_names], axis=0)
    sims = mars_n @ C.T
    distances = 1.0 - sims

    rows = []
    for i, (_, row) in enumerate(mars_df.iterrows()):
        order = np.argsort(distances[i])
        nearest_trad = trad_names[order[0]]
        d_nearest = float(distances[i, order[0]])
        inherited = row["inherited_tradition"]
        if inherited in trad_names:
            inh_idx = trad_names.index(inherited)
            d_inherited = float(distances[i, inh_idx])
            rank_inherited = int(np.where(order == inh_idx)[0][0]) + 1
        else:
            d_inherited = float("nan")
            rank_inherited = -1
        rows.append({
            "region_id": row["region_id"],
            "name": row["name"],
            "inherited_tradition": inherited,
            "nearest_tradition": nearest_trad,
            "nearest_centroid_iso2": centroid_iso2[nearest_trad],
            "inherited_centroid_iso2": centroid_iso2.get(inherited, ""),
            "distance_nearest": d_nearest,
            "distance_inherited": d_inherited,
            "rank_inherited": rank_inherited,
            "hit": nearest_trad == inherited,
            "top3_hit": inherited in [trad_names[order[k]] for k in range(min(3, len(order)))],
        })
    return pd.DataFrame(rows)


def main() -> None:
    EMB_DIR.mkdir(parents=True, exist_ok=True)
    PROJ_DIR.mkdir(parents=True, exist_ok=True)

    earth_emb, earth_iso2, earth_meta = ensure_earth_embeddings()
    print(f"earth: {earth_emb.shape} ({len(earth_iso2)} flags)")

    mars_df = pd.read_csv(MARS_CSV)
    print(f"loading {len(mars_df)} Mars flag definitions")

    mars_emb = embed_mars(mars_df)
    np.save(DATA_DIR / "mars_embeddings.npy", mars_emb)
    print(f"saved {mars_emb.shape} -> data/mars_embeddings.npy")

    print("=== hit-rate analysis ===")
    hit_df = compute_hit_table(mars_df, mars_emb, earth_emb, earth_iso2, earth_meta)
    hit_df.to_csv(DATA_DIR / "mars_distance_table.csv", index=False)
    n_hit = int(hit_df["hit"].sum())
    n_top3 = int(hit_df["top3_hit"].sum())
    chance = 1.0 / hit_df["nearest_tradition"].nunique()
    print(f"top-1 hit rate: {n_hit}/{len(hit_df)} = {n_hit / len(hit_df):.2f}  "
          f"(chance ≈ {chance:.2f}, lift ≈ {n_hit / len(hit_df) / chance:.1f}x)")
    print(f"top-3 hit rate: {n_top3}/{len(hit_df)} = {n_top3 / len(hit_df):.2f}")
    print()
    print(
        hit_df[["region_id", "inherited_tradition", "nearest_tradition",
                "rank_inherited", "hit"]].to_string()
    )

    print()
    X_all = np.concatenate([earth_emb, mars_emb], axis=0)
    proj = joint_project(X_all)

    n_earth = len(earth_iso2)
    rows = []
    for i, code in enumerate(earth_iso2):
        meta_row = earth_meta.iloc[i]
        rows.append({
            "id": code,
            "name": meta_row["name"],
            "vex_category": meta_row["vex_category"],
            "kind": meta_row["kind"],
            "parent": meta_row.get("parent", "") or "",
            "is_mars": False,
            "inherited_tradition": "",
            "pca_x":  float(proj["pca"][i, 0]),  "pca_y":  float(proj["pca"][i, 1]),
            "tsne_x": float(proj["tsne"][i, 0]), "tsne_y": float(proj["tsne"][i, 1]),
            "phate_x": float(proj["phate"][i, 0]), "phate_y": float(proj["phate"][i, 1]),
        })
    for j, (_, mr) in enumerate(mars_df.iterrows()):
        idx = n_earth + j
        rows.append({
            "id": mr["region_id"],
            "name": mr["name"],
            "vex_category": mr["inherited_tradition"],
            "kind": "mars",
            "parent": "",
            "is_mars": True,
            "inherited_tradition": mr["inherited_tradition"],
            "pca_x":  float(proj["pca"][idx, 0]),  "pca_y":  float(proj["pca"][idx, 1]),
            "tsne_x": float(proj["tsne"][idx, 0]), "tsne_y": float(proj["tsne"][idx, 1]),
            "phate_x": float(proj["phate"][idx, 0]), "phate_y": float(proj["phate"][idx, 1]),
        })
    out_df = pd.DataFrame(rows)
    out_path = PROJ_DIR / "projections_all_phase4.parquet"
    out_df.to_parquet(out_path)
    print(f"saved {len(out_df)} rows -> {out_path}")


if __name__ == "__main__":
    main()
