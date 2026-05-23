# flag2vec — site (`/docs`)

This directory is the source for the project's GitHub Pages site: an
interactive 3D atlas of 423 flags (197 sovereigns + 201 subdivisions + 25
Martian) placed in DINOv2 embedding space and projected to 3D via PCA, t-SNE,
and UMAP.

## Live

Once GitHub Pages is enabled with **Source → GitHub Actions**, every push to
`main` that touches `docs/` re-deploys via `.github/workflows/pages.yml`.

URL: `https://<owner>.github.io/flag2vec/` (resolves once Pages is enabled).

## What lives here

```
docs/
  index.html              hero + canvas + side panels + tour
  css/style.css           dark, layered glass aesthetic
  js/app.js               three.js scene, raycasting, filter logic
  data/flags.json         the 423-flag dataset (~293 KB)
  flags/<id>.png          per-flag thumbnails (256×170, ~4.4 MB total)
```

## Regenerating the data

`docs/data/flags.json` and the `docs/flags/*.png` thumbnails are derived
artifacts. They're committed so Pages can serve a pure static build with no
build step, but to rebuild them after upstream embeddings change:

```bash
python3 scripts/20_build_site_data.py
```

The script:
1. Loads `data/embeddings/dinov2_all.npy` (398 sovereign+subdivision vectors,
   518×518 input) and `data/embeddings/dinov2_all_phase4.npy` (197 sovereign
   vectors, 224×224 input).
2. Runs orthogonal Procrustes on the 197 shared sovereigns to align the two
   into a single 384-d frame.
3. Stacks 398 + 25 Mars vectors → 423 in the unified frame.
4. Projects to 3D PCA, 3D t-SNE (perplexity 20, cosine), 3D UMAP (n=15,
   min_dist=0.05).
5. Extracts a 4-color median-cut palette per flag, classifies into 12 named
   families.
6. Writes the JSON and copies/resizes PNG thumbnails.

## License

MIT. Flag SVGs via [hampusborgos/country-flags](https://github.com/hampusborgos/country-flags).
