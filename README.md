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

## Phase 2 — non-sovereign flags

201 subdivision flags joined the embedding alongside the 197 sovereigns: 57 US states/territories, 47 Japanese prefectures, 27 Brazilian states, 26 Swiss cantons, 16 German Länder, 13 Canadian provinces/territories, 8 Australian states/territories, 4 UK constituent countries, plus Greenland, the Faroe Islands, and Åland (the Nordic autonomous regions). Vendored from [amckenna41/iso3166-flags](https://github.com/amckenna41/iso3166-flags) and [google/region-flags](https://github.com/google/region-flags).

### Joint hero

![phase 2 hero](out/phase2/hero.png)

All 398 flags in DINOv2 space, sovereigns drawn slightly larger and on top, subdivisions slightly smaller behind. PCA / t-SNE / PHATE side-by-side.

### Per-country highlights

Drilling into each country's subdivisions: the parent national flag is drawn extra-large with a gold border so you can see where the parent sits relative to the cluster.

| | n | compactness (lower = tighter) | k-NN lift over chance | |
|---|---:|---:|---:|---|
| Australia      |  8 | 0.54 | 31× | [`out/phase2/subdivisions_by_country/AU.png`](out/phase2/subdivisions_by_country/AU.png) |
| Japan          | 47 | 0.75 |  5.4× | [`JP.png`](out/phase2/subdivisions_by_country/JP.png) |
| Brazil         | 27 | 0.75 |  2.7× | [`BR.png`](out/phase2/subdivisions_by_country/BR.png) |
| Canada         | 13 | 1.05 |  2.5× | [`CA.png`](out/phase2/subdivisions_by_country/CA.png) |
| Germany        | 16 | 1.07 |  4.6× | [`DE.png`](out/phase2/subdivisions_by_country/DE.png) |
| Switzerland    | 26 | 1.27 |  7.1× | [`CH.png`](out/phase2/subdivisions_by_country/CH.png) |
| United States  | 57 | 1.27 |  3.0× | [`US.png`](out/phase2/subdivisions_by_country/US.png) |

### Subdivision compactness

![subdivision compactness](out/phase2/subdivision_compactness.png)

Mean within-country pairwise cosine distance ÷ global mean. **Australia** is the tightest (0.54): all 8 of its state/territory flags are British-ensign-derived blue fields, and DINOv2 puts them right next to each other. **Japan** and **Brazil** are next, the prefectures and Brazilian states sharing strong national conventions. **US, Switzerland, UK** sit *above* global spread — meaning US states are, on average, *farther* from each other than two random flags are. The "blue-field-with-state-seal" convention is much weaker than it looks at a glance, because the seals themselves vary enormously (Maryland, New Mexico, Texas, California are all wildly distinct).

### k-NN lift over chance

![knn lift](out/phase2/knn_purity_by_country.png)

Compactness undersells the within-country convention because it averages all pairwise distances. **k-NN lift** compares the fraction of each subdivision's 5 nearest neighbors that come from the same country to the per-country chance baseline. Every subdivision set lifts well above chance: **Australia 31×**, **Switzerland 7×**, **UK 7×**, **Japan 5×**, **Germany 5×**, **US 3×**, **Brazil/Canada ≈ 2.5×**. Even visually-varied sets (Switzerland's heraldic miniatures, the four UK home nations) have neighborhoods full of their siblings — DINOv2 sees the family resemblance even when the global spread is wide.

### Subdivision → national flag distance

![subdivision to parent](out/phase2/subdivision_to_parent.png)

Each dot is one subdivision's cosine distance to its parent national flag in DINOv2 space; the vertical bar is the country mean. Australia's state flags are *closest* to their parent (they are British ensigns, like the AU national flag itself). Japan's mean is also low. The UK's is high — the Union Jack is visually unrelated to the St George's Cross, the saltire of St Andrew, the Welsh dragon, or the Ulster Banner.

## Phase 3 — historical flags and trajectories

30 historical flags joined the embedding alongside the 197 sovereigns and 201 subdivisions: USSR, Russian Empire (Romanov), SFR Yugoslavia, Kingdom of Yugoslavia, Czechoslovakia, East Germany, Rhodesia, Apartheid South Africa, South / North Yemen, South Vietnam, the two Confederate flags (Stars-and-Bars and Battle), British Raj, Republic of China (mainland), Manchukuo, Imperial Japan (Rising Sun), Pahlavi Iran, the Khmer Republic and Democratic Kampuchea, Republic of Texas, Kingdom of Hawaiʻi, independent Tibet, Biafra, Katanga, the Ottoman Empire, Austria-Hungary, the Holy Roman Empire, the Bogd Khanate of Mongolia, and the United Arab Republic. Each is paired with one canonical "modern successor" sovereign (USSR → Russia, Confederate → United States, Manchukuo → China, etc.). Sourced from Wikimedia Commons and rasterized through the same pipeline as Phases 1 and 2.

### Joint hero

![phase 3 hero](out/phase3/hero.png)

All 428 flags in DINOv2 space — sovereigns and subdivisions in their phase-2 positions, with historical flags rendered in sepia and a gold border, drawn on top so you can read where each predecessor sits relative to its modern successor.

### Trajectories — the headline figure

![trajectories](out/phase3/trajectories.png)

For each historical flag, an arrow runs from its position to its modern successor's position in t-SNE space. Arrow color encodes era midpoint (viridis: dark = older). Sovereigns and subdivisions are faded to grayscale in the background so the trajectories are legible. Historical flags carry a gold border; successors are drawn full-color.

The 5 longest and 5 shortest trajectories are labeled. Long trajectories are flags whose successor is visually unlike the predecessor — the Holy Roman Empire's black eagle on yellow → Germany's modern horizontal tricolor, Kingdom-Yugoslavia's pan-Slavic tricolor → Serbia's modern flag with central emblem, the DDR → modern Germany. Short trajectories are flags whose successor kept the predecessor's design — Czechoslovakia → Czech Republic (the Czech Republic kept the flag verbatim), the Ottoman Empire → modern Turkey (almost the same red-with-crescent-and-star design).

### Trajectory length

![trajectory lengths](out/phase3/trajectory_lengths.png)

Bar chart of cosine distance between each historical flag and its successor in 384-dim DINOv2 space, sorted ascending. Bars colored by historical-flag vex category. The shortest trajectories are near-identical pairs: **Czechoslovakia → Czech Republic** (d ≈ 0.000 — literally the same flag) and **Ottoman → Turkey** (d ≈ 0.007 — modern Turkey kept the Ottoman crescent-and-star on red). The longest trajectories are radical visual breaks: **Holy Roman Empire → Germany** (d ≈ 0.86, black eagle on yellow → horizontal tricolor), **Kingdom Yugoslavia → Serbia** (0.62), **DDR → modern Germany** (0.57, the East German horizontal tricolor carried a hammer-and-compass emblem at center), **Rhodesia → Zimbabwe** (0.56, green-white-green with shield → seven-stripe pan-African), and **Imperial Japan (Rising Sun) → Hinomaru** (0.43, 16-ray sunburst → solid red disc).

### Per-successor

![per successor](out/phase3/per_successor.png)

For modern countries with multiple historical predecessors (USSR + Russian Empire → Russia; SFR + Kingdom Yugoslavia → Serbia; ROC + Manchukuo + Tibet → China; Stars-and-Bars + Battle Flag + Republic of Texas + Kingdom of Hawaiʻi → United States; Khmer Republic + Democratic Kampuchea → Cambodia; North + South Yemen → Yemen), small-multiples show each predecessor next to the modern flag with the cosine distance between them. The figure also includes the most-radical singleton trajectories.

A few patterns worth noting:

- **Identity-preserving successions are detectable as near-zero distances.** Czechoslovakia ↔ Czech Republic sits at d ≈ 0; Ottoman ↔ Turkey at 0.007.
- **Color-tradition continuity beats emblem change.** The Russian Empire's Romanov tricolor (black-yellow-white) → modern Russia (white-blue-red) is only 0.07 apart — same horizontal-tricolor structure, even though all three stripe colors changed. The CSA Stars-and-Bars → US (0.08) is similarly close because both share the canton-plus-stripes blueprint.
- **Modernization radicals.** Imperial Japan → Hinomaru is one of the longest (d ≈ 0.43): the Rising Sun is a 16-ray sunburst structure, while the modern flag is a single solid disc — DINOv2 sees them as visually different objects despite the shared red/white palette. Democratic Kampuchea → Cambodia (0.47) is even further: a stylized red Angkor-Wat silhouette on red vs. modern Cambodia's blue/red/blue with full Angkor-Wat illustration.
- **Successor multiplicity.** When several historical flags map to the same modern successor — USSR (0.14) + Russian Empire (0.07) → Russia; CSA Stars-and-Bars (0.08) + CSA Battle (0.28) + Texas (0.15) + Hawaiʻi (0.15) → United States — DINOv2 places the predecessors in *different* parts of the latent space, so the modern flag "inherits from" multiple regions of historical-flag space at once.

## Phase plan

- **Phase 1** — sovereign / observer states, 197 flags. *Shipped.*
- **Phase 2** — major subdivisions, 201 flags. *Shipped.*
- **Phase 3** — 30 historical flags + trajectory arrows to modern successors. *Shipped.*
- **Phase 4** — beaded creative spin-off: Mars-terraformed flags. Earth flag tradition geography → Martian regional inheritance → procedurally generated Martian flags → embed sanity-check.

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
python scripts/11_average_flags.py    # avg-flag analysis vs UN flag

# Phase 2 — joint with subdivisions
python scripts/12_build_subdivision_csv.py  # build subdivision_flags.csv
python scripts/13_phase2_pipeline.py        # rasterize + embed + project + render
python scripts/14_phase2_findings.py        # subdivision-specific findings

# Phase 3 — historical flags + trajectories
python scripts/fetch_historical_svgs.py     # download 30 SVGs from Wikimedia Commons
python scripts/15_phase3_pipeline.py        # rasterize + joint embed + project
python scripts/16_phase3_render.py          # hero, trajectories, lengths, per-successor
```

CPU-only inference. End-to-end runs in ~20 min on a 2-core machine.

## Why the name

A nod to [emoji2vec](https://arxiv.org/abs/1609.08359) — same construction, different domain, opposite question.

## License

MIT for code. Flag SVGs are MIT-licensed via [hampusborgos/country-flags](https://github.com/hampusborgos/country-flags).
