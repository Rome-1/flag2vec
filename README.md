# flag2vec

A flags-domain analogue of Figure 3 in [emoji2vec](https://arxiv.org/abs/1609.08359) (Eisner et al., 2016) — and an exploration of what DINOv2's visual embedding space says about national flags.

![hero](out/latent_flags.png)

emoji2vec's Figure 3 projects 1,661 emoji embeddings to 2D with t-SNE and renders each emoji glyph at its position. Clusters of similar emojis (smileys, animals, fruits, flags) emerge from a *language*-derived embedding space — and notably, in the original paper, all the country flags pile into one undifferentiated cluster, because their Unicode keyword sets are dominated by "flag" + a country name. They're indistinguishable in language-space.

This project asks the dual question. Do *visual* embeddings of national flags recover the *cultural / heraldic / political* groupings a vexillologist would draw by hand — Nordic crosses, pan-Arab tricolor, British ensigns, French tricolor lineage, star-and-crescent, communist red, Latin American horizontal-tricolor-with-charge, and so on?

**Short answer: yes, partially.** Structurally distinctive categories (Nordic cross 64% k=5 NN purity, British ensign 60%, horizontal tricolor 58%) cluster cleanly in DINOv2 space. Color-tradition-only categories (Pan-Slavic 12%, Communist red 6%) don't — because their structural content varies too much. A linear probe on frozen DINOv2 features achieves 35% accuracy across 14 vex categories (chance ≈ 21%), beating both a color-histogram baseline (24%) and a random-feature control (19%). k-means recovers visible diagonal structure against the hand labels (ARI 0.22 / NMI 0.43). Read on for the figures.

## Method

- **Source:** [hampusborgos/country-flags](https://github.com/hampusborgos/country-flags) SVGs (MIT-licensed).
- **Scope (Phase 1):** 197 sovereign UN members + observer states (Vatican, Palestine) + Taiwan + Kosovo.
- **Rasterization:** SVG → PNG, letterboxed to 480×320 (2:3) on transparent background.
- **Embedding:** DINOv2 ViT-S/14 at 518×518, `[CLS]` token (384-dim).
- **Projections:** PCA, t-SNE (perplexity 20, cosine), PHATE (knn 10, decay 20).
- **Annotation:** hand-curated vex categories (`data/sovereign_flags.csv`) and UN regions (`data/regions.csv`).

## The hero figure

The three-panel hero (`out/latent_flags.png`, top of this README) shows all 197 flags rendered at their 2D positions in PCA, t-SNE, and PHATE, with thin colored borders for hand-curated vex categories. Soft hulls appear only where a category genuinely clusters in the projection — `compactness < 0.65 × global mean pairwise distance` — keeping the figure honest.

## By projection

Three larger single-projection figures, one per method:

| | |
|---|---|
| **PCA** — 21% + 11% variance | [`out/projections/pca.png`](out/projections/pca.png) |
| **t-SNE** — perplexity 20, cosine | [`out/projections/tsne.png`](out/projections/tsne.png) |
| **PHATE** — knn 10, decay 20 | [`out/projections/phate.png`](out/projections/phate.png) |

PCA's top two components capture 32% of variance and read roughly as a *color/contrast* axis (high-contrast bicolors at one extreme, washed-out heraldic at the other) versus a *structural complexity* axis (plain fields → multi-element). t-SNE and PHATE separate clusters more crisply but PCA is the most honest about distances — pairs nearby in PCA really are nearby in 384-dim DINOv2 space.

## By vex category

`out/categories/<category>.png` — one figure per category, with that category highlighted across all three projections, the rest faded to grayscale. Compactness numbers per panel make it instantly visible which categories cluster in which projections.

The clearest stories:
- **Nordic cross** — tightest cluster across all three projections. Sweden, Denmark, Norway, Iceland, Finland are visual near-twins to DINOv2.
- **British ensign** — UK, Australia, New Zealand, Fiji, Tuvalu cluster despite very different fly designs.
- **Pan-Arab** — clean cluster anchored on the red/white/black/green horizontal tricolor base.
- **Pan-African** — diffuse. The category includes vertical tricolors (Mali, Senegal), horizontal tricolors (Ethiopia, Ghana), six-stripers (Uganda) and AK-47-bearing flags (Mozambique). DINOv2 sees them as *visually heterogeneous*, even though they share a heritage.
- **Pan-Slavic** — also diffuse, for the same reason: red/white/blue gets used in horizontal tricolors of varied stripe orderings, with very different state symbols on top.
- **Communist red** — almost zero clustering. China, Vietnam, North Korea, Belarus, Mongolia, Angola differ enormously in structure beyond "red field."

## By geographic region

`out/regions/<region>.png` — Africa / Americas / Asia / Europe / Oceania, same highlight treatment.

Region clustering is *much weaker* than vex-category clustering. Europe's compactness is ~1.0 (≈ no preference) — European flags span the gamut from Nordic cross to Mediterranean tricolor to Slavic patterns to British ensign-style. Africa is similarly diffuse. The strongest geographic signal is **Latin America** within Americas: their horizontal-tricolor-with-central-charge convention forms a tight visual subcluster. **Western Asia / MENA** also clusters tightly via shared pan-Arab heritage. Geographic similarity is driven by political/cultural traditions inside the region, not by region itself.

## Quantitative analyses

`out/analysis/` contains a suite of figures probing the embedding from different angles.

### k-NN purity by category — the headline

![knn purity](out/analysis/knn_purity.png)

For each flag, what fraction of its k=5 nearest neighbors share its hand-labeled vex category? Random baseline ≈ 0.21 (size of the largest category). DINOv2 mean = 0.39, with a clear ranking: structural categories beat color-tradition categories.

### "Most prototypical flag" per category

![centroid flags](out/analysis/centroid_flags.png)

For each vex category, the flag whose DINOv2 vector is closest (cosine) to the mean of its category. The Netherlands as the prototypical horizontal tricolor (it *invented* the form). Norway as the prototypical Nordic cross. Liberia as the prototypical stars-and-stripes (a US-derived flag). Tuvalu as the prototypical British ensign. Côte d'Ivoire as the prototypical vertical tricolor. China as the prototypical communist-red. These are not cherry-picked — they fall out of the embedding.

### Most-distant flag pairs

![distant pairs](out/analysis/distant_pairs.png)

The 10 pairs of flags with the largest cosine distance in DINOv2 space — visual extremes of the dataset.

### Nearest cross-category neighbors

![cross-category neighbors](out/analysis/cross_neighbors.png)

The closest pairs whose two flags belong to *different* vex categories — visual cousins across symbol-family boundaries. Guinea (Pan-African) ↔ Romania (Vertical tricolor), Chad ↔ Mali, Colombia ↔ Russia: at this distance, the categorical line is pure heritage, not appearance.

### Most visually unique flags (LOF)

![lof outliers](out/analysis/lof_outliers.png)

Local Outlier Factor (k=15, cosine) on DINOv2 features. Eswatini, Kenya, Saudi Arabia, Japan, Kiribati, Mozambique, Kosovo, Malawi — the flags that sit farthest from any visual neighborhood. These are the "irreducible" flags of the dataset.

### k-means vs vexillology — confusion matrix

![kmeans confusion](out/analysis/kmeans_confusion.png)

k-means(k=15) on the 384-dim embeddings, then row-normalize against hand vex labels (each kmeans cluster reordered to maximize diagonal). ARI ≈ 0.18 / NMI ≈ 0.40 — significant agreement (well above random null ≈ 0) but far from a perfect match. The off-diagonals tell where DINOv2 disagrees with vexillology.

### Cluster quality vs k

![ari vs k](out/analysis/ari_nmi_vs_k.png)

ARI plateaus around k=11–25 at ~0.20, while NMI keeps climbing through k=40. The gray null band is ARI under shuffled labels (mean ≈ 0). DINOv2's clustering doesn't have a single sharp natural granularity that matches the 15-category taxonomy — it has *more* fine structure than the labels capture.

### Linear probe — how separable is vex structure?

![linear probe](out/analysis/linear_probe.png)

Logistic regression (5-fold stratified CV, classes with n≥5) predicting vex category from features. DINOv2 reaches 35% accuracy / 0.13 macro-F1, beating the color-histogram baseline (24% / 0.06) and random features (19% / 0.07). Macro-F1 is suppressed by the long-tail classes (Nordic cross n=5, British ensign n=5, etc., where 5-fold CV gives 1 sample per class per fold). Accuracy is the more meaningful headline here.

### Pairwise distance histograms

![distance histograms](out/analysis/distance_histograms.png)

Same-category vs different-category cosine distances. AUROC of distance as a same/different classifier ≈ 0.60 — meaningful signal but heavy overlap. The visible left tail in same-category distances is the "near-twin" pairs (Nordic crosses with each other, near-identical pan-Arab tricolors).

### k-NN purity vs PCA-reduced dimensionality

![knn purity vs dim](out/analysis/knn_purity_vs_dim.png)

k=5 NN purity rises sharply from 0.21 at d=2 (≈ chance) to 0.38 at d=5, plateaus through d=100 at ~0.40, and at full d=384 sits at 0.39. The 2D projections (t-SNE / PHATE / PCA, what the hero figures actually render) retain only ~half the vex structure of the full embedding — which sets a ceiling on what visual inspection of the 2D plots can reveal.

### Mutual information across metadata

![mi metadata](out/analysis/mi_metadata.png)

NMI between k-means clusters and three independent label sets — vex categories vs UN regions, swept across k. DINOv2 organizes flags by vex symbolism much more strongly than by geography.

### Color count vs distance from global centroid

![color count vs radius](out/analysis/color_count_radius.png)

A test of FIAV's design principle "keep it simple." Does DINOv2 push complex flags (more distinct colors) toward the periphery? The slope is mild — color count alone doesn't drive embedding position much.

### Symmetry vs t-SNE position

![symmetry](out/analysis/symmetry_scatter.png)

Each flag colored by horizontal- and vertical-mirror symmetry score. Tricolors and crosses score high on at least one axis; canton-based and asymmetric designs (British ensigns, USA, Tonga) score low.

### Average flag per category — four ways

![average flag per category](out/analysis/average_flag_per_category.png)

The naive way to compute a category average is to mean the pixel values — which produces a blurry, *off-manifold* image (DINOv2 has never seen a flag that smudgy). Four approaches here, with each row showing one category:

1. **Pixel mean** — naive averaging. Looks washed-out; embeds far from the cluster it represents.
2. **Embedding centroid** — the real flag whose DINOv2 vector is closest to the category's mean. This is on-manifold by definition.
3. **5-nearest mosaic** — the five real flags closest to the centroid. Gives a sense of the cloud, not just the point.
4. **Procedural composition** — rule-based SVG generation: pick the canonical pattern for that category (Nordic cross / pan-Arab triangle+stripes / British ensign / horizontal tricolor) and color-sample from the category's observed palette frequencies. Also on-manifold — a synthetic flag that *could exist*.

### Average flag → UN flag

![average distance to UN](out/analysis/average_distance_to_un.png)

How far is each category's average flag from the United Nations flag (light blue field with white globe-and-olive-branches emblem) in DINOv2 space? Sorted by centroid distance.

Surprising winner: **British ensign** is closest (0.31). The UN flag is essentially a "blue field with white emblem"; UK ensigns are also "blue field with white-and-colored insignia in canton" — DINOv2 sees the structural rhyme. Latin (charge), Solid+emblem, and Communist red follow. Horizontal tricolor and Vertical tricolor sit farthest — pure stripes don't have the field+emblem structure UN does.

The procedural ("on-manifold rule-based") and centroid ("on-manifold real flag") distances tend to agree closely. The pixel-mean and latent-mean-vector approaches are systematically *closer* to UN than the on-manifold approaches — but only because they collapse to a smudgy direction in vector space that no real flag ever lives at. They're shorter-distance shortcuts that don't correspond to anything real. The point of the on-manifold approaches is that the resulting "average flag" is *itself an interpretable flag*.

### Procedural average flags

![procedural averages](out/analysis/procedural_average_flags.png)

The full set of rule-based "average flags" — what a typical flag from each category looks like, generated by composing field-division templates (horizontal-tricolor / vertical-tricolor / Nordic-cross / pan-Arab-triangle / British-ensign / communist-red / star-and-crescent / solid+emblem / stars-and-stripes) with the category's most-frequent palette colors. Bonus: "all sovereign" gets a global procedural that uses the dominant pattern (horizontal tricolor, n=26) and palette (red/white/green).

### Hierarchical clustering dendrogram

![dendrogram](out/analysis/dendrogram.png)

Ward-linkage hierarchical clustering on the full 384-dim space, leaves colored by vex category. The Nordic-cross subtree, the Pan-Arab subtree, and the British-ensign subtree are all visible as monochromatic spans.

## Phase plan

- **Phase 1** — sovereign / observer states, 197 flags. *This commit.*
- **Phase 2** — major subdivisions (US states, Swiss cantons, German Länder, Indian states, Brazilian states, Canadian provinces). Will produce a hero figure for subdivisions plus a cross-tab figure showing whether US states cluster with each other or near similar national flags.
- **Phase 3** — historical flags (USSR, Yugoslavia, Czechoslovakia, Rhodesia, Apartheid SA, French royal banners, Confederate, pre-1991 Eastern Bloc, Ottoman). Trajectories from historical flag → modern successor visualized as arrows in latent space.

## Run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

python scripts/02_rasterize.py        # 197 PNGs
python scripts/03_embed.py            # DINOv2 -> data/embeddings/
python scripts/04_project.py          # PCA / t-SNE / PHATE
python scripts/05_render.py           # hero figure
python scripts/06_per_projection.py   # 3 single-projection figures
python scripts/07_per_category.py     # 15 vex-category highlight figures
python scripts/08_per_region.py       # 5 region highlight figures
python scripts/09_clustering.py       # quantitative metric figures
python scripts/10_galleries.py        # image-based galleries
```

CPU-only inference. End-to-end runs in ~10 min on a 2-core machine.

## Why the name

A nod to [emoji2vec](https://arxiv.org/abs/1609.08359) — same construction, different domain, opposite question.

## License

MIT for code. Flag SVGs are MIT-licensed via [hampusborgos/country-flags](https://github.com/hampusborgos/country-flags).
