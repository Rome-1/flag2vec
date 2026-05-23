"""Build the unified dataset that powers the GitHub Pages site.

Combines phase 1+2 (sovereigns + subdivisions, 518×518 embeddings) with phase 4
(mars flags + their phase-4 sovereign anchors, 224×224 embeddings) via orthogonal
Procrustes alignment on the 197 shared sovereigns. Projects the unified
423-vector matrix to 3D (PCA, t-SNE, UMAP) and 2D (PCA, t-SNE). Computes a
4-color median-cut palette per flag for the palette filter.

Output: docs/data/flags.json + docs/flags/<id>.png thumbnails (copies of
existing PNGs, resized to 256×170 for fast hover).
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"

# Embeddings + PNGs are build artifacts (not in git); they live in the canonical
# flag2vec checkout one level up from the crew worktree. Falls back to the
# crew copy if a file happens to exist locally.
def _pick(rel: str) -> Path:
    local = ROOT / rel
    if local.exists():
        return local
    upstream = ROOT.parent.parent / rel
    if upstream.exists():
        return upstream
    raise FileNotFoundError(f"{rel} not found at {local} or {upstream}")


DATA_LOCAL = ROOT / "data"
EMB_DIR = _pick("data/embeddings")
PROJ_DIR = _pick("data/projections")
PNG_DIR = _pick("data/png")
MARS_PNG_DIR = _pick("out/mars/flags")

# ────────────────────────────────────────────────────────────────────────────
# 1. Load both embeddings + their iso2 orderings.
# ────────────────────────────────────────────────────────────────────────────
print("loading embeddings…")

emb12 = np.load(EMB_DIR / "dinov2_all.npy")               # (398, 384) at 518 (sovereigns + subdivisions)
ord12 = (EMB_DIR / "iso2_order_all.txt").read_text().splitlines()
assert len(ord12) == emb12.shape[0] == 398

# phase-4 sovereign re-embeddings at 224×224 — to align with the mars vectors.
emb4_sov = np.load(EMB_DIR / "dinov2_all_phase4.npy")     # (197, 384) at 224 (sovereigns only)
ord4_sov = (EMB_DIR / "iso2_order_all_phase4.txt").read_text().splitlines()
assert len(ord4_sov) == emb4_sov.shape[0] == 197

# Mars flag embeddings (also at 224 — same DINOv2 forward pass as phase-4 sovs).
mars_vecs_raw = np.load(_pick("data/mars_embeddings.npy"))  # (25, 384)
mars_iso = pd.read_csv(DATA_LOCAL / "mars_regions.csv")["region_id"].tolist()
assert len(mars_iso) == mars_vecs_raw.shape[0] == 25

# l2-normalize so cosine ~= dot.
emb12 = normalize(emb12)
emb4_sov = normalize(emb4_sov)
mars_vecs_raw = normalize(mars_vecs_raw)

# Orthogonal Procrustes: align phase-4 (224-res) onto phase-1+2 (518-res) using
# the 197 shared sovereigns. R maps phase-4 frame → phase-1+2 frame.
idx12 = {c: i for i, c in enumerate(ord12)}
idx4 = {c: i for i, c in enumerate(ord4_sov)}
shared = [c for c in ord4_sov if c in idx12]
print(f"  shared sovereigns: {len(shared)}")

A = emb12[[idx12[c] for c in shared]]   # phase-1+2 frame
B = emb4_sov[[idx4[c] for c in shared]] # phase-4 frame
U, _, Vt = np.linalg.svd(B.T @ A, full_matrices=False)
R = U @ Vt

diff = np.linalg.norm((B @ R) - A, axis=1).mean()
print(f"  procrustes mean residual on shared sovereigns: {diff:.3f}")

# Apply alignment to mars vectors.
mars_vecs = mars_vecs_raw @ R

# ────────────────────────────────────────────────────────────────────────────
# 2. Stack 423 = 398 (sov+sub) + 25 mars.
# ────────────────────────────────────────────────────────────────────────────
print("building unified matrix…")
ids = ord12 + mars_iso
X = np.vstack([emb12, mars_vecs])
X = normalize(X)
print(f"  unified: {X.shape}")

# ────────────────────────────────────────────────────────────────────────────
# 3. Project to 3D + 2D.
# ────────────────────────────────────────────────────────────────────────────
print("computing 3D PCA…")
pca3 = PCA(n_components=3, random_state=0).fit_transform(X)

print("computing 3D t-SNE… (slow)")
tsne3 = TSNE(n_components=3, perplexity=20, metric="cosine",
             init="pca", learning_rate="auto", random_state=0,
             max_iter=1500).fit_transform(X)

print("computing 3D UMAP…")
import umap
umap3 = umap.UMAP(n_components=3, metric="cosine", n_neighbors=15,
                  min_dist=0.05, random_state=0).fit_transform(X)

# 2D for the "flat map" view (and to match the README's 2D figures).
print("computing 2D PCA + t-SNE…")
pca2 = PCA(n_components=2, random_state=0).fit_transform(X)
tsne2 = TSNE(n_components=2, perplexity=20, metric="cosine",
             init="pca", learning_rate="auto", random_state=0,
             max_iter=1500).fit_transform(X)


def normalize_coords(arr):
    """Center on 0, scale so the 99th percentile of |coord| is 1."""
    arr = arr - arr.mean(0, keepdims=True)
    scale = np.percentile(np.abs(arr), 99)
    return (arr / max(scale, 1e-9)).astype(np.float32)


pca3, tsne3, umap3 = (normalize_coords(a) for a in (pca3, tsne3, umap3))
pca2, tsne2 = (normalize_coords(a) for a in (pca2, tsne2))

# ────────────────────────────────────────────────────────────────────────────
# 4. Metadata join.
# ────────────────────────────────────────────────────────────────────────────
print("joining metadata…")
all_flags = pd.read_csv(DATA_LOCAL / "all_flags.csv")
regions = pd.read_csv(DATA_LOCAL / "regions.csv")
mars_regions = pd.read_csv(DATA_LOCAL / "mars_regions.csv")

all_flags = all_flags.merge(regions, on="iso2", how="left")
all_flags = all_flags.set_index("iso2")

# Build mars metadata frame to match all_flags schema.
mars_meta = mars_regions.set_index("region_id")

# Historical flags (in case they're referenced as parents/successors).
hist = pd.read_csv(DATA_LOCAL / "historical_flags.csv").set_index("iso2") if (DATA_LOCAL / "historical_flags.csv").exists() else None

# Era hints (sovereign / subdivision modern era).
def make_record(idx, fid, coords3, coords2):
    p3, t3, u3 = coords3
    p2, t2 = coords2
    rec = {
        "id": fid,
        "kind": None,
        "name": fid,
        "vex_category": None,
        "region": None,
        "subregion": None,
        "parent": None,
        "thumb": f"flags/{fid}.png",
        "pca3": [float(p3[0]), float(p3[1]), float(p3[2])],
        "tsne3": [float(t3[0]), float(t3[1]), float(t3[2])],
        "umap3": [float(u3[0]), float(u3[1]), float(u3[2])],
        "pca2": [float(p2[0]), float(p2[1])],
        "tsne2": [float(t2[0]), float(t2[1])],
    }
    if fid.startswith("mars-"):
        m = mars_meta.loc[fid]
        rec.update({
            "kind": "mars",
            "name": str(m["name"]),
            "vex_category": str(m["inherited_tradition"]),
            "feature_type": str(m["feature_type"]),
            "rationale": str(m["inheritance_rationale"]),
            "latitude": float(m["latitude"]),
            "longitude": float(m["longitude"]),
            "era_start": 2300,
            "era_end": None,
        })
    else:
        f = all_flags.loc[fid]
        rec.update({
            "kind": str(f["kind"]),
            "name": str(f["name"]),
            "vex_category": str(f["vex_category"]),
            "region": str(f["region"]) if pd.notna(f["region"]) else None,
            "subregion": str(f["subregion"]) if pd.notna(f["subregion"]) else None,
            "parent": str(f["parent"]) if pd.notna(f["parent"]) else None,
            "era_start": None,
            "era_end": None,
        })
    return rec


flags = []
for i, fid in enumerate(ids):
    flags.append(make_record(
        i, fid,
        (pca3[i], tsne3[i], umap3[i]),
        (pca2[i], tsne2[i]),
    ))


# ────────────────────────────────────────────────────────────────────────────
# 5. Palette extraction (median-cut, 4 colors), thumbnail copy.
# ────────────────────────────────────────────────────────────────────────────
print("extracting palettes + writing thumbnails…")
DOCS.mkdir(exist_ok=True)
(DOCS / "flags").mkdir(exist_ok=True)
(DOCS / "data").mkdir(exist_ok=True)

THUMB = (256, 170)

PALETTE_FAMILIES = {
    "red":     [(220, 30, 30)],
    "white":   [(245, 245, 245)],
    "black":   [(20, 20, 20)],
    "green":   [(20, 130, 60)],
    "blue":    [(30, 60, 180)],
    "yellow":  [(245, 215, 50)],
    "orange":  [(230, 140, 40)],
    "purple":  [(120, 50, 150)],
    "brown":   [(120, 80, 50)],
    "rust":    [(190, 90, 50)],
    "sage":    [(140, 170, 130)],
    "cyan":    [(70, 200, 220)],
}


def color_to_family(rgb):
    best = None
    bestd = 1e18
    for name, refs in PALETTE_FAMILIES.items():
        for r in refs:
            d = sum((a - b) ** 2 for a, b in zip(rgb, r))
            if d < bestd:
                bestd = d
                best = name
    return best


def palette_for(png_path: Path):
    img = Image.open(png_path).convert("RGB")
    # median cut, 4 colors
    small = img.copy()
    small.thumbnail((96, 64))
    pal = small.quantize(colors=4, method=Image.MEDIANCUT)
    palette = pal.getpalette()[:12]
    counts = sorted(pal.getcolors(), reverse=True)  # [(count, idx), ...]
    palette_rgb = []
    families = set()
    total = sum(c for c, _ in counts) or 1
    avg = [0, 0, 0]
    for cnt, idx in counts:
        r, g, b = palette[idx * 3: idx * 3 + 3]
        share = cnt / total
        palette_rgb.append({"hex": f"#{r:02x}{g:02x}{b:02x}", "share": round(share, 3)})
        families.add(color_to_family((r, g, b)))
        avg[0] += r * share
        avg[1] += g * share
        avg[2] += b * share
    return {
        "palette": palette_rgb,
        "families": sorted(families),
        "avg": f"#{int(avg[0]):02x}{int(avg[1]):02x}{int(avg[2]):02x}",
    }


missing = []
for rec in flags:
    fid = rec["id"]
    if fid.startswith("mars-"):
        src = MARS_PNG_DIR / f"{fid}.png"
    else:
        src = PNG_DIR / f"{fid}.png"
    if not src.exists():
        missing.append(fid)
        continue
    dst = DOCS / "flags" / f"{fid}.png"
    if not dst.exists() or dst.stat().st_size == 0:
        img = Image.open(src).convert("RGBA")
        # White-pad rather than blend; the canvas already letterboxes.
        img.thumbnail(THUMB)
        img.save(dst, optimize=True)
    rec.update(palette_for(src))

if missing:
    print(f"  WARNING: {len(missing)} flags missing PNG: {missing[:10]}")

# ────────────────────────────────────────────────────────────────────────────
# 6. Write the JSON + a metadata summary.
# ────────────────────────────────────────────────────────────────────────────
print("writing flags.json…")

# Collect vex categories, regions, palette families for filter chip lists.
vex_cats = sorted({f["vex_category"] for f in flags if f["vex_category"]})
regions_set = sorted({f["region"] for f in flags if f.get("region")})
subregions_set = sorted({f["subregion"] for f in flags if f.get("subregion")})
kinds = sorted({f["kind"] for f in flags if f["kind"]})
palette_set = sorted({fam for f in flags for fam in f.get("families", [])})

doc = {
    "meta": {
        "n_flags": len(flags),
        "vex_categories": vex_cats,
        "regions": regions_set,
        "subregions": subregions_set,
        "kinds": kinds,
        "palette_families": palette_set,
    },
    "flags": flags,
}

out_path = DOCS / "data" / "flags.json"
out_path.write_text(json.dumps(doc, separators=(",", ":")))
print(f"  wrote {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")

# Pretty-print summary
print("\nsummary:")
print(f"  flags: {len(flags)}")
print(f"  vex categories: {len(vex_cats)} ({vex_cats[:5]} …)")
print(f"  regions: {regions_set}")
print(f"  kinds: {kinds}")
print(f"  palette families used: {palette_set}")
