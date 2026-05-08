# flag2vec

A flags-domain analogue of Figure 3 in [emoji2vec](https://arxiv.org/abs/1609.08359) (Eisner et al., 2016).

![hero figure](out/latent_flags.png)

emoji2vec's Figure 3 projects 1,661 emoji embeddings to 2D with t-SNE and renders each emoji glyph at its position. Clusters of similar emojis (smileys, animals, fruits, flags) emerge from a *language-derived* embedding space — and notably, in the original paper, all the country flags pile into one undifferentiated cluster, because their Unicode keyword sets are dominated by "flag" + a country name. They're indistinguishable in language-space.

This project asks the dual question: do *visual* embeddings of national flags recover the *cultural / heraldic / political* groupings a vexillologist would draw by hand — Nordic crosses, pan-African colors, pan-Arab tricolor, British ensigns, French tricolors, star-and-crescent, Soviet-derived, Latin American horizontal tricolors with central charges, and so on?

## Method

- **Source:** [hampusborgos/country-flags](https://github.com/hampusborgos/country-flags) (MIT-licensed SVGs).
- **Scope (Phase 1):** 197 sovereign UN members + observer states + Taiwan + Kosovo.
- **Rasterization:** SVG → PNG (480×320, 2:3, letterboxed on transparent background).
- **Embedding:** DINOv2 ViT-S/14 at 518×518, `[CLS]` token (384-dim).
- **Projection:** PCA, t-SNE, PHATE — all three rendered side-by-side.
- **Annotation:** hand-curated vexillological category labels (`data/sovereign_flags.csv`) drawn as colored borders on each flag; soft hulls appear *only where a category genuinely clusters in a given projection* — keeping the figure honest.

## Phase plan

- **Phase 1** — sovereign / observer states (197 flags). This repo, in this commit.
- **Phase 2** — major subdivisions (US states, Swiss cantons, German Länder, Indian states, Brazilian states, Canadian provinces). Adds ~150–250 flags. Will produce a "subdivisions clustering" hero figure and a side-by-side "do US states cluster with each other or with similar national flags?" cross-tab.
- **Phase 3** — historical flags (USSR, Yugoslavia, Rhodesia/Zimbabwe, French royal banners, Confederate flag, pre-1991 Eastern Bloc, Apartheid-era SA). Trajectories from historical flag → modern successor visualized as arrows in latent space.

## Run

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/02_rasterize.py
python scripts/03_embed.py
python scripts/04_project.py
python scripts/05_render.py
```

SVGs are vendored under `data/raw_svg/` (from hampusborgos/country-flags, MIT). Outputs land in `out/`.

## Why the name

A nod to [emoji2vec](https://arxiv.org/abs/1609.08359) — same construction, different domain, opposite question.

## License

MIT for code. Flag SVGs are MIT-licensed via [hampusborgos/country-flags](https://github.com/hampusborgos/country-flags).
