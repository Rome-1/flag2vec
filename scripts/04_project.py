"""Project DINOv2 embeddings to 2D with PCA, t-SNE, and PHATE."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize

ROOT = Path(__file__).resolve().parent.parent
EMB_DIR = ROOT / "data" / "embeddings"
PROJ_DIR = ROOT / "data" / "projections"
CSV_PATH = ROOT / "data" / "sovereign_flags.csv"


def main() -> None:
    PROJ_DIR.mkdir(parents=True, exist_ok=True)
    X = np.load(EMB_DIR / "dinov2_vits14.npy")
    iso2_order = (EMB_DIR / "iso2_order.txt").read_text().strip().splitlines()
    print(f"loaded {X.shape}")

    # L2-normalize so cosine becomes Euclidean
    Xn = normalize(X, norm="l2")

    print("PCA...")
    pca = PCA(n_components=2, random_state=0)
    pca_xy = pca.fit_transform(Xn)
    print(f"  explained variance: {pca.explained_variance_ratio_}")

    print("t-SNE...")
    tsne = TSNE(
        n_components=2,
        perplexity=20,
        metric="cosine",
        init="pca",
        learning_rate="auto",
        random_state=0,
    )
    tsne_xy = tsne.fit_transform(Xn)

    print("PHATE...")
    import phate

    phate_op = phate.PHATE(
        n_components=2,
        knn=10,
        decay=20,
        t="auto",
        random_state=0,
        verbose=0,
    )
    phate_xy = phate_op.fit_transform(Xn)

    with CSV_PATH.open() as f:
        meta = {row["iso2"]: row for row in csv.DictReader(f)}

    df = pd.DataFrame(
        {
            "iso2": iso2_order,
            "name": [meta[i]["name"] for i in iso2_order],
            "vex_category": [meta[i]["vex_category"] for i in iso2_order],
            "pca_x": pca_xy[:, 0],
            "pca_y": pca_xy[:, 1],
            "tsne_x": tsne_xy[:, 0],
            "tsne_y": tsne_xy[:, 1],
            "phate_x": phate_xy[:, 0],
            "phate_y": phate_xy[:, 1],
        }
    )
    df.to_parquet(PROJ_DIR / "projections.parquet")
    df.to_csv(PROJ_DIR / "projections.csv", index=False)
    print(f"saved {len(df)} rows -> {PROJ_DIR / 'projections.parquet'}")


if __name__ == "__main__":
    main()
