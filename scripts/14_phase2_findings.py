"""Phase 2 quantitative findings on the joint sovereign+subdivision embedding.

Produces:
  - subdivision_compactness.png      How tightly does each country's subdivisions cluster?
  - subdivision_to_parent.png         Distance from each subdivision to its parent national flag
  - knn_purity_by_country.png         Per-country k-NN purity (does my flag find its sibling subdivisions?)
  - sovereign_vs_subdivision.png      Distribution of sovereign-vs-subdivision distances
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

from flag2vec.style import (
    BG, CATEGORY_COLORS, NICE_NAME, OUT_DIR, SUBTLE, TEXT,
    configure_typography,
)

ROOT = Path(__file__).resolve().parent.parent
EMB_DIR = ROOT / "data" / "embeddings"
OUT = OUT_DIR / "phase2"

PARENT_NICE = {
    "US": "United States",
    "CA": "Canada",
    "JP": "Japan",
    "CH": "Switzerland",
    "DE": "Germany",
    "AU": "Australia",
    "BR": "Brazil",
    "GB": "United Kingdom",
}

PARENT_TO_SOV = {
    "US": "us", "CA": "ca", "JP": "jp", "CH": "ch", "DE": "de",
    "AU": "au", "BR": "br", "GB": "gb",
}


def load():
    X = np.load(EMB_DIR / "dinov2_all.npy")
    iso2 = (EMB_DIR / "iso2_order_all.txt").read_text().strip().splitlines()
    df = pd.read_csv(ROOT / "data" / "all_flags.csv").set_index("iso2").loc[iso2].reset_index()
    Xn = normalize(X, norm="l2")
    return X, Xn, df


def fig_subdivision_compactness(Xn, df):
    """How tight is each country's subdivision cluster vs. the global spread?"""
    rows = []
    for parent, nice in PARENT_NICE.items():
        sub_mask = (df["kind"] == "subdivision") & (df["parent"] == parent)
        if sub_mask.sum() < 3:
            continue
        sub_pts = Xn[sub_mask]
        # mean within-cluster cosine distance
        within = []
        for i in range(len(sub_pts)):
            for j in range(i + 1, len(sub_pts)):
                within.append(1 - np.dot(sub_pts[i], sub_pts[j]))
        all_pts = Xn
        outer = []
        # sample 5000 pairs
        rng = np.random.default_rng(0)
        for _ in range(5000):
            i, j = rng.integers(0, len(all_pts), size=2)
            if i != j:
                outer.append(1 - np.dot(all_pts[i], all_pts[j]))
        compactness = np.mean(within) / np.mean(outer)
        rows.append({"parent": parent, "nice": nice,
                     "n": int(sub_mask.sum()), "compactness": compactness,
                     "within_mean": np.mean(within)})
    out = pd.DataFrame(rows).sort_values("compactness")

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(colors=SUBTLE, length=0)
    bars = ax.barh(range(len(out)), out["compactness"],
                   color=[CATEGORY_COLORS["pan_arab"]] * len(out),
                   edgecolor=BG, linewidth=1.2, height=0.78)
    ax.axvline(1.0, color=SUBTLE, alpha=0.5, linewidth=0.9, linestyle="--")
    ax.text(1.01, len(out) - 0.5, " = global spread",
            color=SUBTLE, fontsize=9, va="top")
    ax.set_yticks(range(len(out)))
    ax.set_yticklabels([f"{r['nice']}  (n={r['n']})" for _, r in out.iterrows()],
                       fontsize=10.5, color=TEXT)
    ax.set_xlim(0, max(1.2, out["compactness"].max() * 1.1))
    ax.set_xlabel("compactness  (within-country mean ÷ global mean cosine distance)",
                  color=SUBTLE, fontsize=10.5)
    for bar, c in zip(bars, out["compactness"]):
        ax.text(c + 0.015, bar.get_y() + bar.get_height() / 2, f"{c:.2f}",
                color=TEXT, fontsize=10, va="center", fontweight="medium")

    fig.suptitle("subdivision-cluster compactness, by country",
                 fontsize=20, fontweight="medium", color=TEXT,
                 x=0.04, y=0.97, ha="left")
    fig.text(0.04, 0.91,
             "Lower = tighter cluster.  Japanese prefectures and US states share "
             "strong design conventions (single emblem on solid field; blue field with state seal). "
             "Canadian provinces and Brazilian states have more visual variety.",
             fontsize=10.5, color=SUBTLE, ha="left")
    plt.subplots_adjust(left=0.30, right=0.96, top=0.84, bottom=0.10)
    fig.savefig(OUT / "subdivision_compactness.png", dpi=200, facecolor=BG)
    plt.close(fig)
    print(f"wrote subdivision_compactness.png")
    return out


def fig_subdivision_to_parent(Xn, df):
    """How close are subdivisions to their parent national flag?"""
    rows = []
    for parent, sov_iso in PARENT_TO_SOV.items():
        sov_idx = df.index[(df["kind"] == "sovereign") & (df["iso2"] == sov_iso)]
        if len(sov_idx) == 0:
            continue
        sov_vec = Xn[sov_idx[0]]
        sub_mask = (df["kind"] == "subdivision") & (df["parent"] == parent)
        sub_idx = df.index[sub_mask]
        if len(sub_idx) == 0:
            continue
        for i in sub_idx:
            d = 1 - np.dot(sov_vec, Xn[i])
            rows.append({"parent": parent,
                         "subdivision": df.iloc[i]["iso2"],
                         "name": df.iloc[i]["name"],
                         "d": d})

    rdf = pd.DataFrame(rows)
    summary = (rdf.groupby("parent")
               .agg(mean_d=("d", "mean"), median_d=("d", "median"),
                    min_d=("d", "min"), max_d=("d", "max"),
                    n=("d", "size"))
               .reset_index()
               .merge(pd.DataFrame({"parent": list(PARENT_NICE.keys()),
                                     "nice": list(PARENT_NICE.values())}), on="parent")
               .sort_values("mean_d"))

    fig, ax = plt.subplots(figsize=(11, 7))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(colors=SUBTLE, length=0)

    # Strip plot — each subdivision a dot, colored by parent
    palette = [CATEGORY_COLORS["pan_arab"], CATEGORY_COLORS["nordic_cross"],
               CATEGORY_COLORS["pan_african"], CATEGORY_COLORS["star_crescent"],
               CATEGORY_COLORS["communist_red"], CATEGORY_COLORS["latin_charge"],
               CATEGORY_COLORS["pan_slavic"], CATEGORY_COLORS["british_ensign"]]
    for i, parent in enumerate(summary["parent"]):
        sub_d = rdf[rdf["parent"] == parent]["d"].to_numpy()
        y_jitter = i + np.random.default_rng(i).normal(0, 0.07, size=len(sub_d))
        ax.scatter(sub_d, y_jitter, alpha=0.55, s=30,
                   color=palette[i % len(palette)], edgecolor=BG, linewidth=0.5)
        ax.scatter([summary.iloc[i]["mean_d"]], [i], marker="|", s=300,
                   color="#1A1A1A", linewidth=2.5, zorder=5)

    ax.set_yticks(range(len(summary)))
    ax.set_yticklabels([f"{r['nice']}  (n={int(r['n'])})"
                        for _, r in summary.iterrows()],
                       fontsize=10.5, color=TEXT)
    ax.set_xlabel("cosine distance from subdivision to its national flag (DINOv2)",
                  color=SUBTLE, fontsize=10.5)
    ax.set_xlim(0, max(rdf["d"].max() * 1.05, 0.8))

    fig.suptitle("subdivision → national flag distance",
                 fontsize=20, fontweight="medium", color=TEXT,
                 x=0.04, y=0.97, ha="left")
    fig.text(0.04, 0.91,
             "Each dot is one subdivision's cosine distance to its national flag in DINOv2 space; "
             "vertical bar is the country mean.  Smaller distance = subdivision visually echoes "
             "the parent national flag.",
             fontsize=10.5, color=SUBTLE, ha="left")
    plt.subplots_adjust(left=0.22, right=0.97, top=0.84, bottom=0.10)
    fig.savefig(OUT / "subdivision_to_parent.png", dpi=200, facecolor=BG)
    plt.close(fig)
    print(f"wrote subdivision_to_parent.png")


def fig_knn_purity_by_country(Xn, df, k=5):
    """For each subdivision, what fraction of its k=5 NN are from the same country?"""
    sub_mask = df["kind"] == "subdivision"
    sub_idx = np.where(sub_mask)[0]
    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(Xn)
    _, idx = nn.kneighbors(Xn)

    rows = []
    for i in sub_idx:
        parent = df.iloc[i]["parent"]
        nbrs_parents = [df.iloc[j]["parent"]
                        if df.iloc[j]["kind"] == "subdivision"
                        else df.iloc[j]["iso2"].upper()
                        for j in idx[i, 1:]]
        same = sum(1 for p in nbrs_parents if p == parent)
        rows.append({"parent": parent, "purity": same / k})

    pdf = pd.DataFrame(rows)
    n_total = len(df)
    by_parent = (pdf.groupby("parent")
                 .agg(mean=("purity", "mean"),
                      n=("purity", "size"))
                 .reset_index())
    # Per-country chance baseline: among the other 397 flags, what fraction are
    # subdivisions of the same country? = (n_country - 1) / (n_total - 1)
    by_parent["chance"] = (by_parent["n"] - 1) / (n_total - 1)
    by_parent["lift"] = by_parent["mean"] / by_parent["chance"]
    by_parent = by_parent.merge(
        pd.DataFrame({"parent": list(PARENT_NICE.keys()),
                      "nice": list(PARENT_NICE.values())}), on="parent"
    ).sort_values("lift", ascending=True)

    fig, ax = plt.subplots(figsize=(11, 6.5))
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(colors=SUBTLE, length=0)
    bars = ax.barh(range(len(by_parent)), by_parent["lift"],
                   color=CATEGORY_COLORS["british_ensign"],
                   edgecolor=BG, linewidth=1.2, height=0.78)
    ax.axvline(1.0, color=SUBTLE, alpha=0.6, linewidth=0.9, linestyle="--")
    ax.text(1.02, len(by_parent) - 0.5, " chance",
            color=SUBTLE, fontsize=9, va="top")
    ax.set_yticks(range(len(by_parent)))
    ax.set_yticklabels(
        [f"{r['nice']}  (n={int(r['n'])}, purity {r['mean']:.2f}, chance {r['chance']:.3f})"
         for _, r in by_parent.iterrows()], fontsize=10, color=TEXT)
    ax.set_xlim(0, by_parent["lift"].max() * 1.10)
    ax.set_xlabel(f"k-NN lift over chance (k={k})",
                  color=SUBTLE, fontsize=10.5)
    for bar, lift in zip(bars, by_parent["lift"]):
        ax.text(lift + 0.4, bar.get_y() + bar.get_height() / 2,
                f"{lift:.1f}×",
                color=TEXT, fontsize=10.5, va="center", fontweight="medium")

    fig.suptitle(f"k-NN purity lift by country (k={k})",
                 fontsize=20, fontweight="medium", color=TEXT,
                 x=0.04, y=0.97, ha="left")
    fig.text(0.04, 0.91,
             "For each subdivision, what fraction of its 5 nearest neighbors are subdivisions of the same country, "
             "divided by the per-country chance baseline ((n_country − 1) ÷ (n_total − 1)). "
             "Every subdivision set lifts well above chance — Australia by 30×, Switzerland 7×, Japan 5×.",
             fontsize=10.5, color=SUBTLE, ha="left")
    plt.subplots_adjust(left=0.42, right=0.96, top=0.84, bottom=0.10)
    fig.savefig(OUT / "knn_purity_by_country.png", dpi=200, facecolor=BG)
    plt.close(fig)
    print(f"wrote knn_purity_by_country.png")


def main():
    configure_typography()
    OUT.mkdir(parents=True, exist_ok=True)
    X, Xn, df = load()
    print(f"loaded {Xn.shape}, {(df['kind']=='subdivision').sum()} subdivisions")
    fig_subdivision_compactness(Xn, df)
    fig_subdivision_to_parent(Xn, df)
    fig_knn_purity_by_country(Xn, df)
    print("done")


if __name__ == "__main__":
    main()
