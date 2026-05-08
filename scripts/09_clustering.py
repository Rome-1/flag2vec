"""Quantitative cluster / probe / metric analyses on the DINOv2 embeddings.

Produces:
  - kmeans_confusion.png       k-means(k=15) vs hand vex categories
  - knn_purity.png             k=5 NN agreement, ranked by category
  - distance_histograms.png    same- vs cross-category cosine distances
  - ari_nmi_vs_k.png           cluster quality vs k, with shuffled null
  - mi_metadata.png            mutual information vs vex / region / category granularity
  - linear_probe.png           logistic-regression accuracy vs feature set
  - knn_purity_vs_dim.png      how much vex structure survives PCA reduction to d=2..384
  - dendrogram.png             Ward hierarchical clustering, colored leaves
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score,
                              confusion_matrix, f1_score)
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

from flag2vec.style import (
    BG, CATEGORY_COLORS, CSV_PATH, EMB_DIR, NICE_NAME, OUT_DIR,
    REGION_CSV, REGION_COLORS, SUBTLE, TEXT, configure_typography,
)

OUT_ANALYSIS = OUT_DIR / "analysis"


def load_data():
    X = np.load(EMB_DIR / "dinov2_vits14.npy")
    iso2 = (EMB_DIR / "iso2_order.txt").read_text().strip().splitlines()
    meta = pd.read_csv(CSV_PATH).set_index("iso2").loc[iso2].reset_index()
    regions = pd.read_csv(REGION_CSV).set_index("iso2").loc[iso2].reset_index()
    meta["region"] = regions["region"].values
    Xn = normalize(X, norm="l2")
    return X, Xn, iso2, meta


def setup_axes(ax, title: str = "", subtitle: str = ""):
    ax.set_facecolor(BG)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(colors=SUBTLE, length=0)


def figure(figsize, title, subtitle):
    fig, ax = plt.subplots(figsize=figsize)
    fig.patch.set_facecolor(BG)
    setup_axes(ax)
    # Place title + subtitle as figure-level text, stacked, top-left aligned.
    fig.suptitle(title, fontsize=18, fontweight="medium", color=TEXT,
                 x=0.04, y=0.97, ha="left")
    if subtitle:
        fig.text(0.04, 0.91, subtitle, fontsize=10.5, color=SUBTLE,
                 ha="left")
    return fig, ax


def fig_kmeans_confusion(X: np.ndarray, meta: pd.DataFrame):
    cats = sorted(meta["vex_category"].unique(),
                  key=lambda c: -(meta["vex_category"] == c).sum())
    cat_to_i = {c: i for i, c in enumerate(cats)}
    k = len(cats)
    km = KMeans(n_clusters=k, n_init=20, random_state=0).fit(X)
    labels = km.labels_

    # Reorder kmeans cluster ids to maximize diagonal — Hungarian-ish via
    # assigning each cluster to its dominant true category.
    counts = np.zeros((k, k), dtype=int)
    for true_cat, kl in zip(meta["vex_category"], labels):
        counts[cat_to_i[true_cat], kl] += 1
    order = []
    used = set()
    for i in range(k):
        # for each true category in order of size, pick the kmeans cluster
        # with the most overlap that hasn't been picked
        ranked = np.argsort(-counts[i])
        for j in ranked:
            if j not in used:
                order.append(j)
                used.add(j)
                break
    counts = counts[:, order]
    row_norm = counts / counts.sum(axis=1, keepdims=True).clip(min=1)

    fig, ax = figure((10, 9),
                     "k-means(k=15) vs vexillological categories",
                     f"row-normalized · ARI {adjusted_rand_score(meta['vex_category'], labels):.3f}"
                     f" · NMI {normalized_mutual_info_score(meta['vex_category'], labels):.3f}")
    im = ax.imshow(row_norm, aspect="auto", cmap="YlOrBr", vmin=0, vmax=1)
    ax.set_yticks(range(k))
    ax.set_yticklabels([NICE_NAME.get(c, c) for c in cats], fontsize=10)
    ax.set_xticks(range(k))
    ax.set_xticklabels([f"c{i}" for i in range(k)], fontsize=9)
    ax.set_xlabel("k-means cluster (re-ordered)", color=SUBTLE, fontsize=10.5)
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cb.outline.set_visible(False)
    cb.ax.tick_params(colors=SUBTLE, length=0)
    plt.subplots_adjust(left=0.18, right=0.95, top=0.86, bottom=0.08)
    fig.savefig(OUT_ANALYSIS / "kmeans_confusion.png", dpi=200, facecolor=BG)
    plt.close(fig)


def fig_knn_purity(Xn: np.ndarray, meta: pd.DataFrame, k: int = 5):
    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(Xn)
    _, idx = nn.kneighbors(Xn)
    cats = meta["vex_category"].to_numpy()
    purity = np.zeros(len(cats))
    for i in range(len(cats)):
        nbrs = cats[idx[i, 1:]]
        purity[i] = (nbrs == cats[i]).mean()
    meta = meta.copy()
    meta["purity"] = purity

    by_cat = (meta.groupby("vex_category")
              .agg(mean=("purity", "mean"),
                   n=("purity", "size"))
              .sort_values("mean", ascending=True))
    by_cat = by_cat[by_cat["n"] >= 2]

    fig, ax = figure((11, 8),
                     f"k-NN purity by category (k={k})",
                     f"global mean {purity.mean():.2f} · "
                     f"random baseline = max category share {(meta['vex_category'].value_counts(normalize=True).iloc[0]):.2f}")
    colors = [CATEGORY_COLORS.get(c, "#888888") for c in by_cat.index]
    bars = ax.barh(range(len(by_cat)), by_cat["mean"], color=colors,
                   edgecolor=BG, linewidth=1.2, height=0.78)
    ax.set_yticks(range(len(by_cat)))
    ax.set_yticklabels([f"{NICE_NAME.get(c, c)}  (n={int(n)})"
                        for c, n in zip(by_cat.index, by_cat["n"])], fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_xlabel("fraction of 5 nearest neighbors in same category",
                  color=SUBTLE, fontsize=10.5)
    ax.axvline(purity.mean(), color=SUBTLE, linewidth=1.0, alpha=0.6,
               linestyle="--", label="global mean")
    ax.legend(frameon=False, fontsize=9.5, labelcolor=TEXT, loc="lower right")
    plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.88])
    fig.savefig(OUT_ANALYSIS / "knn_purity.png", dpi=200, facecolor=BG)
    plt.close(fig)


def fig_distance_histograms(Xn: np.ndarray, meta: pd.DataFrame):
    from sklearn.metrics import roc_auc_score
    D = squareform(pdist(Xn, metric="cosine"))
    same = []
    diff = []
    cats = meta["vex_category"].to_numpy()
    n = len(cats)
    for i in range(n):
        for j in range(i + 1, n):
            (same if cats[i] == cats[j] else diff).append(D[i, j])
    same = np.array(same); diff = np.array(diff)

    y = np.r_[np.zeros_like(same), np.ones_like(diff)]
    s = np.r_[same, diff]
    auc = roc_auc_score(y, s)

    fig, ax = figure((10, 6.5),
                     "pairwise cosine distance, same- vs different-category",
                     f"same n={len(same)} · different n={len(diff)} · "
                     f"AUROC of distance as same/different classifier = {auc:.3f}")
    bins = np.linspace(0, max(same.max(), diff.max()), 60)
    ax.hist(same, bins=bins, color=CATEGORY_COLORS["pan_arab"], alpha=0.55,
            density=True, label="same category")
    ax.hist(diff, bins=bins, color=CATEGORY_COLORS["nordic_cross"], alpha=0.45,
            density=True, label="different category")
    ax.set_xlabel("cosine distance", color=SUBTLE, fontsize=10.5)
    ax.set_ylabel("density", color=SUBTLE, fontsize=10.5)
    ax.legend(frameon=False, fontsize=10, labelcolor=TEXT)
    plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.88])
    fig.savefig(OUT_ANALYSIS / "distance_histograms.png", dpi=200, facecolor=BG)
    plt.close(fig)


def fig_ari_vs_k(X: np.ndarray, meta: pd.DataFrame):
    rng = np.random.default_rng(0)
    cats = meta["vex_category"].to_numpy()
    ks = list(range(2, 41))
    aris = []; nmis = []
    null_aris = []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X)
        aris.append(adjusted_rand_score(cats, km.labels_))
        nmis.append(normalized_mutual_info_score(cats, km.labels_))
        # permutation null: shuffle the true labels, score against same kmeans
        nulls = []
        for _ in range(20):
            shuf = rng.permutation(cats)
            nulls.append(adjusted_rand_score(shuf, km.labels_))
        null_aris.append((np.mean(nulls), np.std(nulls)))
    null_mean = np.array([n[0] for n in null_aris])
    null_std = np.array([n[1] for n in null_aris])

    fig, ax = figure((11, 6.5),
                     "cluster quality vs k",
                     "ARI and NMI of k-means against vex labels, with shuffled-label null")
    ax.plot(ks, aris, marker="o", markersize=4,
            color=CATEGORY_COLORS["pan_arab"], label="ARI")
    ax.plot(ks, nmis, marker="s", markersize=4,
            color=CATEGORY_COLORS["nordic_cross"], label="NMI")
    ax.fill_between(ks, null_mean - 2 * null_std, null_mean + 2 * null_std,
                    color=SUBTLE, alpha=0.15, label="ARI null (±2σ)")
    ax.axvline(15, color=SUBTLE, alpha=0.4, linewidth=0.8, linestyle=":")
    ax.text(15.3, 0.02, "k = 15 (number of vex categories)",
            color=SUBTLE, fontsize=9, va="bottom")
    ax.set_xlabel("k", color=SUBTLE, fontsize=10.5)
    ax.set_ylabel("score", color=SUBTLE, fontsize=10.5)
    ax.legend(frameon=False, fontsize=10, labelcolor=TEXT, loc="lower right")
    plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.88])
    fig.savefig(OUT_ANALYSIS / "ari_nmi_vs_k.png", dpi=200, facecolor=BG)
    plt.close(fig)


def fig_mi_metadata(X: np.ndarray, meta: pd.DataFrame):
    rng = np.random.default_rng(0)
    label_sets = {
        "vex category": meta["vex_category"].to_numpy(),
        "region":       meta["region"].to_numpy(),
    }
    ks = [5, 10, 15, 20, 30]
    rows = []
    for k in ks:
        km = KMeans(n_clusters=k, n_init=10, random_state=0).fit(X)
        for name, lbl in label_sets.items():
            true_nmi = normalized_mutual_info_score(lbl, km.labels_)
            null = [normalized_mutual_info_score(rng.permutation(lbl), km.labels_)
                    for _ in range(50)]
            rows.append({"k": k, "label": name, "nmi": true_nmi,
                         "null_mean": np.mean(null), "null_std": np.std(null)})
    df = pd.DataFrame(rows)

    fig, ax = figure((11, 6),
                     "NMI between k-means clustering and metadata",
                     "vex labels vs UN region across k; shaded = ±2σ shuffled null")
    palette = {"vex category": CATEGORY_COLORS["pan_arab"],
               "region":       CATEGORY_COLORS["nordic_cross"]}
    for name, sub in df.groupby("label"):
        sub = sub.sort_values("k")
        ax.plot(sub["k"], sub["nmi"], marker="o", markersize=5,
                color=palette[name], label=name, linewidth=1.6)
        ax.fill_between(sub["k"],
                        sub["null_mean"] - 2 * sub["null_std"],
                        sub["null_mean"] + 2 * sub["null_std"],
                        color=palette[name], alpha=0.10)
    ax.set_xlabel("k", color=SUBTLE, fontsize=10.5)
    ax.set_ylabel("NMI", color=SUBTLE, fontsize=10.5)
    ax.legend(frameon=False, fontsize=10, labelcolor=TEXT, loc="upper left")
    plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.88])
    fig.savefig(OUT_ANALYSIS / "mi_metadata.png", dpi=200, facecolor=BG)
    plt.close(fig)


def fig_linear_probe(X: np.ndarray, meta: pd.DataFrame):
    """Logistic regression on DINOv2 features vs vex categories, 5-fold CV.
    Compare to a color-histogram baseline and random-feature control."""
    from PIL import Image
    cats = meta["vex_category"].to_numpy()

    # color-histogram baseline (HSV, 8x8x4) — vectorized
    from matplotlib.colors import rgb_to_hsv
    color_X = []
    for iso2 in meta["iso2"]:
        img = (Image.open(OUT_DIR.parent / "data" / "png" / f"{iso2}.png")
               .convert("RGBA"))
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[-1])
        rgb = np.asarray(bg.convert("RGB"), dtype=np.float32) / 255.0
        hsv = rgb_to_hsv(rgb)
        h = hsv[..., 0].ravel(); s = hsv[..., 1].ravel(); v = hsv[..., 2].ravel()
        H, _ = np.histogramdd((h, s, v), bins=(8, 8, 4),
                              range=((0, 1), (0, 1), (0, 1)))
        color_X.append(H.ravel() / max(1.0, H.sum()))
    color_X = np.array(color_X)

    rng = np.random.default_rng(0)
    rand_X = rng.normal(size=(len(cats), 384))

    feature_sets = {
        "DINOv2 (384d)": normalize(X, norm="l2"),
        "color histogram (256d)": color_X,
        "random (384d)": rand_X,
    }
    results = {}
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    # Need at least 2 samples per class for stratified-kfold; drop singletons.
    counts = pd.Series(cats).value_counts()
    keep_classes = counts[counts >= 5].index
    keep_mask = np.isin(cats, keep_classes)

    for name, feats in feature_sets.items():
        f1s = []
        accs = []
        for tr, te in skf.split(feats[keep_mask], cats[keep_mask]):
            clf = LogisticRegression(max_iter=2000, C=1.0)
            clf.fit(feats[keep_mask][tr], cats[keep_mask][tr])
            pred = clf.predict(feats[keep_mask][te])
            f1s.append(f1_score(cats[keep_mask][te], pred, average="macro"))
            accs.append((pred == cats[keep_mask][te]).mean())
        results[name] = (np.mean(f1s), np.std(f1s), np.mean(accs))
    print("linear probe:", results)

    n_classes = int(len(set(cats[keep_mask])))
    chance = float(pd.Series(cats[keep_mask]).value_counts(normalize=True).iloc[0])

    fig, ax = figure((10, 5.5),
                     "linear probe: predict vex category from features",
                     f"5-fold stratified CV · {n_classes} classes (n≥5) · "
                     f"chance baseline {chance:.2f}")
    names = list(results.keys())
    f1_means = np.array([results[n][0] for n in names])
    f1_stds  = np.array([results[n][1] for n in names])
    accs     = np.array([results[n][2] for n in names])
    x = np.arange(len(names))
    w = 0.36
    palette = [CATEGORY_COLORS["pan_arab"], CATEGORY_COLORS["pan_african"], SUBTLE]
    palette2 = [CATEGORY_COLORS["nordic_cross"], CATEGORY_COLORS["star_crescent"], "#888888"]
    b1 = ax.bar(x - w/2, accs, w, color=palette, edgecolor=BG, linewidth=1.2,
                label="accuracy")
    b2 = ax.bar(x + w/2, f1_means, w, yerr=f1_stds, color=palette2, alpha=0.85,
                edgecolor=BG, linewidth=1.2, capsize=3, label="macro-F1")
    ax.axhline(chance, color=SUBTLE, linewidth=0.9, linestyle="--", alpha=0.6)
    ax.text(len(names) - 0.5, chance + 0.005, f"chance = {chance:.2f}",
            color=SUBTLE, fontsize=9, ha="right", va="bottom")
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=10.5, color=TEXT)
    ax.set_ylim(0, max(0.6, accs.max() * 1.15))
    ax.set_ylabel("score", color=SUBTLE, fontsize=10.5)
    for bar, m in zip(b1, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, m + 0.01, f"{m:.2f}",
                ha="center", color=TEXT, fontsize=10, fontweight="medium")
    for bar, m in zip(b2, f1_means):
        ax.text(bar.get_x() + bar.get_width() / 2, m + 0.01, f"{m:.2f}",
                ha="center", color=TEXT, fontsize=10, fontweight="medium")
    ax.legend(frameon=False, fontsize=10, labelcolor=TEXT, loc="upper right")
    plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.88])
    fig.savefig(OUT_ANALYSIS / "linear_probe.png", dpi=200, facecolor=BG)
    plt.close(fig)


def fig_knn_purity_vs_dim(Xn: np.ndarray, meta: pd.DataFrame, k: int = 5):
    cats = meta["vex_category"].to_numpy()
    n_samples = Xn.shape[0]
    dims = [2, 5, 10, 20, 50, 100, 196, 384]
    purities = []
    for d in dims:
        if d >= min(n_samples, Xn.shape[1]):
            X_d = Xn
        else:
            X_d = PCA(n_components=d, random_state=0).fit_transform(Xn)
            X_d = normalize(X_d, norm="l2")
        nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(X_d)
        _, idx = nn.kneighbors(X_d)
        p = np.array([(cats[idx[i, 1:]] == cats[i]).mean()
                      for i in range(len(cats))]).mean()
        purities.append(p)

    fig, ax = figure((10, 5.5),
                     "k-NN purity vs PCA-reduced dimension",
                     f"k={k} · how much vex structure survives dimensional compression")
    ax.plot(dims, purities, marker="o", markersize=6,
            color=CATEGORY_COLORS["pan_arab"], linewidth=1.8)
    ax.set_xscale("log")
    ax.set_xticks(dims)
    ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, _: int(x)))
    ax.set_xlabel("dimensionality", color=SUBTLE, fontsize=10.5)
    ax.set_ylabel("k-NN purity", color=SUBTLE, fontsize=10.5)
    for d, p in zip(dims, purities):
        ax.annotate(f"{p:.2f}", (d, p), textcoords="offset points",
                    xytext=(0, 8), ha="center", color=TEXT, fontsize=9)
    plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.88])
    fig.savefig(OUT_ANALYSIS / "knn_purity_vs_dim.png", dpi=200, facecolor=BG)
    plt.close(fig)


def fig_dendrogram(Xn: np.ndarray, meta: pd.DataFrame):
    Z = linkage(Xn, method="ward")
    cats = meta["vex_category"].to_numpy()
    iso2 = meta["iso2"].to_numpy()
    leaf_colors = [CATEGORY_COLORS.get(c, "#999999") for c in cats]

    fig, ax = figure((22, 10),
                     "hierarchical clustering of DINOv2 embeddings",
                     "Ward linkage · leaves colored by hand vex category")
    ddata = dendrogram(Z, labels=iso2, leaf_rotation=90,
                       color_threshold=0, above_threshold_color="#999999",
                       ax=ax)
    leaves = ddata["leaves"]
    ax.set_xticks(np.arange(len(leaves)) * 10 + 5)
    ax.set_xticklabels([iso2[i].upper() for i in leaves], fontsize=6.5)
    for ticklabel, leaf_idx in zip(ax.get_xticklabels(), leaves):
        ticklabel.set_color(CATEGORY_COLORS.get(cats[leaf_idx], "#999999"))
    ax.set_ylabel("Ward distance", color=SUBTLE, fontsize=10.5)
    plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.88])
    fig.savefig(OUT_ANALYSIS / "dendrogram.png", dpi=200, facecolor=BG)
    plt.close(fig)


def main():
    configure_typography()
    OUT_ANALYSIS.mkdir(parents=True, exist_ok=True)
    X, Xn, iso2, meta = load_data()
    print(f"loaded {X.shape}, {len(meta)} flags")

    print("k-means confusion ...");  fig_kmeans_confusion(X, meta)
    print("k-NN purity ...");        fig_knn_purity(Xn, meta)
    print("distance histograms ..."); fig_distance_histograms(Xn, meta)
    print("ARI/NMI vs k ...");        fig_ari_vs_k(X, meta)
    print("MI metadata ...");        fig_mi_metadata(X, meta)
    print("linear probe ...");        fig_linear_probe(X, meta)
    print("k-NN purity vs dim ...");  fig_knn_purity_vs_dim(Xn, meta)
    print("dendrogram ...");          fig_dendrogram(Xn, meta)
    print("done")


if __name__ == "__main__":
    main()
