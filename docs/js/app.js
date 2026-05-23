/**
 * flag2vec — three.js scene + interaction layer.
 *
 * Renders 423 flag billboards in 3D embedding space (PCA / t-SNE / UMAP),
 * with hover, click, filter, color-by, projection-swap, and tour-camera moves.
 *
 * Performance approach: each flag is a THREE.Sprite with its own ~256-px
 * texture, lazy-loaded as the scene boots. With ~420 sprites the draw-call
 * count is comfortable on any laptop GPU; the bottleneck is texture decode at
 * load time, which we mask with the boot ring.
 */

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

// ─── DOM refs ─────────────────────────────────────────────────────────────
const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));

const canvas      = $("#scene");
const boot        = $("#boot");
const tooltip     = $("#tooltip");
const tipImg      = $("#tip-img");
const tipName     = $("#tip-name");
const tipMeta     = $("#tip-meta");
const tipPalette  = $("#tip-palette");
const detail      = $("#detail");
const detailImg   = $("#detail-img");
const detailName  = $("#detail-name");
const detailKvs   = $("#detail-kvs");
const detailPal   = $("#detail-palette");
const detailLink  = $("#detail-link");
const detailClose = $("#detail-close");
const filterCount = $("#filter-count");
const legend      = $("#legend");
const resetBtn    = $("#reset-view");
const clearBtn    = $("#clear-filters");
const searchInp   = $("#search");

// ─── Color tokens (mirror style.css) ──────────────────────────────────────
const PALETTE_COLORS = {
  red:"#e53e3e", white:"#f5f5f5", black:"#222", green:"#2bb673",
  blue:"#4d6bff", yellow:"#ffd166", orange:"#ff8a3d", purple:"#a05cff",
  brown:"#8b5a3c", rust:"#c84a2f", sage:"#9ab388", cyan:"#6dd5ed",
};
const TRADITION_COLORS = {
  nordic_cross:"#6dd5ed", british_ensign:"#4d6bff", pan_arab:"#2bb673",
  pan_african:"#e7b417", pan_slavic:"#a05cff", communist_red:"#e53e3e",
  latin_charge:"#ff6a3d", stars_stripes:"#5089ff", horizontal_tricolor:"#5cd6c0",
  vertical_tricolor:"#f08aa8", star_crescent:"#14b8a6", solid_emblem:"#c44569",
  heraldic:"#c89178", saltire:"#7aa6ff", unique:"#8892a6",
  japanese_geometric:"#f08aa8",  // observed in data but uncommon
};
const REGION_COLORS = {
  Africa:"#f59e0b", Americas:"#ff6a3d", Asia:"#e53e3e",
  Europe:"#4d6bff", Oceania:"#2bb673",
};
const KIND_COLORS = {
  sovereign:"#6dd5ed", subdivision:"#c89178",
  historical:"#ffd166", mars:"#ff5a48",
};
const FALLBACK = "#aaaaaa";

const colorFor = (mode, flag) => {
  if (mode === "vex_category") return TRADITION_COLORS[flag.vex_category] || FALLBACK;
  if (mode === "region") return flag.kind === "mars" ? "#ff5a48" : (REGION_COLORS[flag.region] || FALLBACK);
  if (mode === "kind") return KIND_COLORS[flag.kind] || FALLBACK;
  if (mode === "palette") return flag.avg || FALLBACK;
  return FALLBACK;
};

const formatTradition = (t) => (t || "—").replaceAll("_", " ");

// vex_categories that actually have a published figure in out/categories/.
// The 3 missing ones (us_state_seal, swiss_canton, japanese_geometric) are
// subdivision-only categories — those flags get a phase-2 country figure
// instead, or fall back to the README. Keeping this list explicit so a
// broken link can't slip in unnoticed.
const CATEGORY_FIGURES = new Set([
  "british_ensign","communist_red","heraldic","horizontal_tricolor",
  "latin_charge","nordic_cross","pan_african","pan_arab","pan_slavic",
  "saltire","solid_emblem","star_crescent","stars_stripes","unique",
  "vertical_tricolor",
]);
const PHASE2_COUNTRIES = new Set(["au","br","ca","ch","de","jp","us"]);
const README_ANCHORS = {
  vex: "https://github.com/Rome-1/flag2vec#by-vex-category",
  region: "https://github.com/Rome-1/flag2vec#by-geographic-region",
  phase2: "https://github.com/Rome-1/flag2vec#phase-2--non-sovereign-flags",
  phase3: "https://github.com/Rome-1/flag2vec#phase-3--historical-flags-and-trajectories",
  mars: "https://github.com/Rome-1/flag2vec#phase-4--mars-terraformed-flags",
};

function projectLinkFor(f){
  if (f.kind === "mars") return README_ANCHORS.mars;
  if (f.kind === "subdivision"){
    const parent = (f.parent || "").toLowerCase();
    if (PHASE2_COUNTRIES.has(parent)){
      return `https://github.com/Rome-1/flag2vec/blob/main/out/phase2/subdivisions_by_country/${parent.toUpperCase()}.png`;
    }
    return README_ANCHORS.phase2;
  }
  if (f.vex_category && CATEGORY_FIGURES.has(f.vex_category)){
    return `https://github.com/Rome-1/flag2vec/blob/main/out/categories/${f.vex_category}.png`;
  }
  return README_ANCHORS.vex;
}

// ─── State ────────────────────────────────────────────────────────────────
// Filters that the page boots with — and that "reset" restores. Only sovereigns
// by default; subdivisions and Mars flags are opt-in via the kind chips.
const DEFAULT_FILTERS = () => ({
  vex_category: new Set(),
  region: new Set(),
  kind: new Set(["sovereign"]),
  palette_families: new Set(),
});

const state = {
  projection: "tsne3",
  colorBy: "vex_category",
  filters: DEFAULT_FILTERS(),
  query: "",
  data: null,
  byId: new Map(),
  sprites: [],     // THREE.Sprite[], parallel to data.flags
  visibleMask: [], // boolean[]
  hoveredIdx: -1,
  selectedIdx: -1,
  scene: null, camera: null, renderer: null, controls: null,
  raycaster: new THREE.Raycaster(),
  mouseNDC: new THREE.Vector2(),
  group: null,
};

// ─── Boot ─────────────────────────────────────────────────────────────────
async function boot_app(){
  const resp = await fetch("data/flags.json");
  state.data = await resp.json();
  state.data.flags.forEach((f) => state.byId.set(f.id, f));
  $("#hero-count").textContent = state.data.flags.length;
  populateHeroMarquee();

  initScene();
  buildSprites();
  buildFilters();
  buildLegend();
  wireToolbar();
  wireFilters();
  wireTour();
  wireSearch();
  applyFilters();
  applyColors();

  // boot fade
  requestAnimationFrame(() => boot.classList.add("gone"));
  setTimeout(() => boot.remove(), 700);
}

function populateHeroMarquee(){
  const strip = $("#hero-strip");
  // Pick a curated mix: every tradition gets a representative, rest filled in.
  const seen = new Set();
  const picks = [];
  for (const f of state.data.flags){
    if (f.kind === "subdivision") continue;
    if (seen.has(f.vex_category)) continue;
    picks.push(f);
    seen.add(f.vex_category);
    if (picks.length > 24) break;
  }
  // Tile twice for seamless marquee loop.
  const html = picks.concat(picks).map(f => `<img src="${f.thumb}" alt="${f.name}" loading="lazy" />`).join("");
  strip.innerHTML = html;
}

// ─── Scene setup ──────────────────────────────────────────────────────────
function initScene(){
  const wrap = canvas.parentElement;
  const w = wrap.clientWidth, h = wrap.clientHeight;

  state.scene = new THREE.Scene();
  state.scene.fog = new THREE.Fog(0x08090d, 18, 60);

  state.camera = new THREE.PerspectiveCamera(40, w/h, 0.1, 200);
  state.camera.position.set(7, 5, 9);

  state.renderer = new THREE.WebGLRenderer({
    canvas, antialias: true, alpha: false,
    powerPreference: "high-performance",
  });
  state.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  state.renderer.setSize(w, h, false);
  state.renderer.setClearColor(0x08090d, 1);

  // Soft global ambient — sprites use MeshBasicMaterial-equivalent, no lighting
  // needed, but a hint of background glow helps depth perception.
  state.scene.add(new THREE.AmbientLight(0xffffff, 0.6));

  // Subtle floor + grid for depth cue
  const grid = new THREE.GridHelper(40, 40, 0x1a1d28, 0x12141b);
  grid.position.y = -2.6;
  grid.material.opacity = 0.5; grid.material.transparent = true;
  state.scene.add(grid);

  // Group all sprites under one parent so we can re-target the raycaster cheaply.
  state.group = new THREE.Group();
  state.scene.add(state.group);

  // Category-color indicator (instantiated in buildSprites once N is known).
  state._discGeom = new THREE.CircleGeometry(0.12, 24);
  state._discMat = new THREE.MeshBasicMaterial({
    transparent: true, opacity: 0.85, depthWrite: false,
  });

  state.controls = new OrbitControls(state.camera, canvas);
  state.controls.enableDamping = true;
  state.controls.dampingFactor = 0.07;
  state.controls.rotateSpeed = 0.7;
  state.controls.minDistance = 1.5;
  state.controls.maxDistance = 50;
  state.controls.autoRotate = true;
  state.controls.autoRotateSpeed = 0.35;

  // Stop auto-rotation as soon as the user interacts.
  canvas.addEventListener("pointerdown", () => { state.controls.autoRotate = false; }, { once: true });
  canvas.addEventListener("wheel", () => { state.controls.autoRotate = false; }, { once: true, passive: true });

  // Hover + click
  canvas.addEventListener("pointermove", onPointerMove);
  canvas.addEventListener("pointerleave", () => { hideTooltip(); state.hoveredIdx = -1; });
  canvas.addEventListener("click", onPointerClick);

  // Resize
  const ro = new ResizeObserver(() => {
    const W = wrap.clientWidth, H = wrap.clientHeight;
    state.renderer.setSize(W, H, false);
    state.camera.aspect = W/H;
    state.camera.updateProjectionMatrix();
  });
  ro.observe(wrap);

  // Buttons
  resetBtn.addEventListener("click", () => {
    // Restore default sovereign-only filter + recenter the camera.
    state.filters = DEFAULT_FILTERS();
    syncChipsFromState();
    applyFilters();
    animateCamera(new THREE.Vector3(7,5,9), new THREE.Vector3(0,0,0), 1.2);
    state.controls.autoRotate = true;
  });
  detailClose.addEventListener("click", () => { detail.hidden = true; state.selectedIdx = -1; refreshHighlights(); });

  // Fullscreen toggle (on the canvas wrapper, not just the canvas — keeps the
  // toolbar / filter panel / legend overlaid inside fullscreen).
  const fsBtn = $("#fullscreen");
  if (fsBtn) fsBtn.addEventListener("click", () => {
    if (document.fullscreenElement){
      document.exitFullscreen();
    } else {
      wrap.requestFullscreen().catch((e) => console.warn("fullscreen denied", e));
    }
  });
  document.addEventListener("fullscreenchange", () => {
    if (fsBtn) fsBtn.classList.toggle("on", !!document.fullscreenElement);
    // canvas size changes — let the ResizeObserver pick it up on next frame
  });

  // The hover tooltip lives inside #canvas-wrap, so when the user scrolls the
  // page (and the canvas moves under the still-stationary cursor) we never get
  // a pointermove/pointerleave event — the tooltip would stay pinned. Hide it
  // on any scroll. Same for window blur (alt-tab away mid-hover).
  const onScrollOrBlur = () => { if (!tooltip.hidden) hideTooltip(); state.hoveredIdx = -1; };
  window.addEventListener("scroll", onScrollOrBlur, { passive: true });
  window.addEventListener("blur", onScrollOrBlur);
  document.addEventListener("visibilitychange", () => { if (document.hidden) onScrollOrBlur(); });

  // Animation loop
  state.renderer.setAnimationLoop(tick);
}

// ─── Sprites ──────────────────────────────────────────────────────────────
function buildSprites(){
  const loader = new THREE.TextureLoader();
  const flags = state.data.flags;
  state.visibleMask = new Array(flags.length).fill(true);

  // Instantiate the disc InstancedMesh + force-init the instanceColor attribute.
  state.discs = new THREE.InstancedMesh(state._discGeom, state._discMat, flags.length);
  state.discs.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
  state.discs.instanceColor = new THREE.InstancedBufferAttribute(new Float32Array(flags.length * 3), 3);
  state.scene.add(state.discs);

  const dummy = new THREE.Object3D();
  flags.forEach((f, i) => {
    const c = f[state.projection];
    dummy.position.set(c[0]*4, c[1]*4 - 0.18, c[2]*4);
    dummy.scale.set(1, 1, 1);
    dummy.lookAt(state.camera.position);
    dummy.updateMatrix();
    state.discs.setMatrixAt(i, dummy.matrix);
  });
  state.discs.instanceMatrix.needsUpdate = true;

  flags.forEach((f, i) => {
    // Material starts with a 1×1 placeholder, swapped when the real texture loads.
    const mat = new THREE.SpriteMaterial({
      color: 0xffffff,
      transparent: true,
      depthTest: true,
      depthWrite: false,
      sizeAttenuation: true,
    });
    const sprite = new THREE.Sprite(mat);
    sprite.userData.flag = f;
    sprite.userData.index = i;
    sprite.scale.set(0.42, 0.28, 1);  // 3:2 aspect, world units
    placeSprite(sprite, f, state.projection);
    state.group.add(sprite);
    state.sprites.push(sprite);

    loader.load(f.thumb, (tex) => {
      tex.colorSpace = THREE.SRGBColorSpace;
      tex.anisotropy = state.renderer.capabilities.getMaxAnisotropy();
      mat.map = tex;
      mat.needsUpdate = true;
    });
  });
}

function placeSprite(sprite, flag, projection){
  const c = flag[projection];
  // Scale spread so it sits comfortably in [-4, 4] cube
  sprite.position.set(c[0] * 4, c[1] * 4, c[2] * 4);
}

// Place the colored disc indicator below each flag — billboarded per frame.
function updateDiscs(){
  if (!state.discs) return;
  const dummy = new THREE.Object3D();
  const camPos = state.camera.position;
  for (let i = 0; i < state.sprites.length; i++){
    const s = state.sprites[i];
    if (!state.visibleMask[i]){
      dummy.scale.set(0, 0, 0);
      dummy.position.copy(s.position);
      dummy.updateMatrix();
      state.discs.setMatrixAt(i, dummy.matrix);
      continue;
    }
    // Offset the disc slightly DOWN in screen space → use a vector perpendicular
    // to the camera direction. Simplest: offset by world-y so the disc sits
    // just below the flag. Looks fine for nearly-horizontal cameras.
    dummy.position.set(s.position.x, s.position.y - 0.22, s.position.z);
    dummy.lookAt(camPos);
    const sc = i === state.selectedIdx ? 1.4 : 1.0;
    dummy.scale.set(sc, sc, 1);
    dummy.updateMatrix();
    state.discs.setMatrixAt(i, dummy.matrix);
  }
  state.discs.instanceMatrix.needsUpdate = true;
}

// ─── Animation loop ───────────────────────────────────────────────────────
function tick(){
  state.controls.update();
  updateDiscs();
  state.renderer.render(state.scene, state.camera);
}

// ─── Hover / click via raycasting ─────────────────────────────────────────
function onPointerMove(ev){
  const rect = canvas.getBoundingClientRect();
  state.mouseNDC.x = ((ev.clientX - rect.left) / rect.width) * 2 - 1;
  state.mouseNDC.y = -((ev.clientY - rect.top) / rect.height) * 2 + 1;
  state.raycaster.setFromCamera(state.mouseNDC, state.camera);

  const visibleSprites = state.sprites.filter((s, i) => state.visibleMask[i]);
  const hits = state.raycaster.intersectObjects(visibleSprites, false);

  if (hits.length){
    const idx = hits[0].object.userData.index;
    state.hoveredIdx = idx;
    showTooltip(idx, ev.clientX, ev.clientY);
    bringForward(hits[0].object);
  } else {
    hideTooltip();
    state.hoveredIdx = -1;
  }
}

function onPointerClick(){
  if (state.hoveredIdx >= 0){
    selectFlag(state.hoveredIdx);
  }
}

function bringForward(sprite){
  // Sprites use depthTest so we don't need explicit ordering, but a small scale
  // bump on hover makes the interaction tactile.
  state.sprites.forEach((s, i) => {
    if (i === sprite.userData.index){
      s.scale.set(0.7, 0.467, 1);
      s.material.opacity = 1;
    } else if (state.visibleMask[i]) {
      s.scale.set(0.42, 0.28, 1);
    }
  });
}

function showTooltip(idx, x, y){
  const f = state.data.flags[idx];
  const wrapRect = canvas.parentElement.getBoundingClientRect();
  const dx = x - wrapRect.left + 14;
  const dy = y - wrapRect.top + 14;
  tooltip.style.left = `${Math.min(dx, wrapRect.width - 290)}px`;
  tooltip.style.top  = `${Math.min(dy, wrapRect.height - 110)}px`;
  tipImg.src = f.thumb;
  tipName.textContent = f.name;
  const meta = [];
  if (f.vex_category) meta.push(formatTradition(f.vex_category));
  if (f.kind === "mars") meta.push(`Mars · ${f.feature_type || "region"}`);
  else if (f.kind === "subdivision" && f.parent) meta.push(`subdivision of ${(state.byId.get(f.parent)||{}).name || f.parent.toUpperCase()}`);
  else if (f.region) meta.push(`${f.region}${f.subregion ? " · " + f.subregion : ""}`);
  tipMeta.textContent = meta.join(" · ");
  tipPalette.innerHTML = (f.palette || []).map(p => `<span style="background:${p.hex}"></span>`).join("");
  tooltip.hidden = false;
}

function hideTooltip(){ tooltip.hidden = true; }

function selectFlag(idx){
  state.selectedIdx = idx;
  const f = state.data.flags[idx];
  detail.hidden = false;
  detailImg.src = f.thumb;
  detailName.textContent = f.name;

  const rows = [];
  rows.push(["tradition", `<span class="tradition">${formatTradition(f.vex_category)}</span>`]);
  if (f.kind === "mars"){
    rows.push(["world", "Mars"]);
    rows.push(["feature", f.feature_type || "—"]);
    rows.push(["lat/long", `${f.latitude}°, ${f.longitude}°`]);
    if (f.rationale) rows.push(["rationale", f.rationale]);
  } else {
    rows.push(["kind", f.kind]);
    if (f.region) rows.push(["region", `${f.region}${f.subregion ? " · " + f.subregion : ""}`]);
    if (f.parent){
      const p = state.byId.get(f.parent);
      rows.push(["parent", p ? p.name : f.parent.toUpperCase()]);
    }
  }
  detailKvs.innerHTML = rows.map(([k,v]) => `<div class="k">${k}</div><div class="v">${v}</div>`).join("");
  detailPal.innerHTML = (f.palette || []).map(p => `<span style="background:${p.hex}" title="${p.hex}"></span>`).join("");

  detailLink.href = projectLinkFor(f);

  refreshHighlights();
  flyToFlag(idx);
}

function flyToFlag(idx){
  const sprite = state.sprites[idx];
  const target = sprite.position.clone();
  const direction = state.camera.position.clone().sub(state.controls.target).normalize();
  const newCam = target.clone().add(direction.multiplyScalar(3.5));
  animateCamera(newCam, target, 0.9);
}

function animateCamera(camTo, targetTo, duration){
  const camFrom = state.camera.position.clone();
  const tgtFrom = state.controls.target.clone();
  const t0 = performance.now();
  function step(){
    const t = Math.min(1, (performance.now() - t0) / (duration * 1000));
    const e = t < 0.5 ? 2*t*t : 1 - Math.pow(-2*t+2, 2)/2; // easeInOutQuad
    state.camera.position.lerpVectors(camFrom, camTo, e);
    state.controls.target.lerpVectors(tgtFrom, targetTo, e);
    state.controls.update();
    if (t < 1) requestAnimationFrame(step);
  }
  step();
}

function refreshHighlights(){
  state.sprites.forEach((s, i) => {
    const visible = state.visibleMask[i];
    const isSelected = i === state.selectedIdx;
    s.visible = visible;
    if (isSelected){
      s.scale.set(0.85, 0.567, 1);
      s.material.opacity = 1;
    } else if (visible){
      s.scale.set(0.42, 0.28, 1);
      s.material.opacity = state.selectedIdx === -1 ? 1 : 0.4;
    }
  });
}

// ─── Filters ──────────────────────────────────────────────────────────────
function buildFilters(){
  const m = state.data.meta;
  buildChips("vex_category", m.vex_categories, (v) => TRADITION_COLORS[v] || FALLBACK, formatTradition);
  buildChips("region", m.regions, (v) => REGION_COLORS[v] || FALLBACK);
  buildChips("kind", m.kinds, (v) => KIND_COLORS[v] || FALLBACK);
  buildChips("palette_families", m.palette_families, (v) => PALETTE_COLORS[v] || FALLBACK);
}

function buildChips(filterKey, values, colorFn, labelFn = (v) => v){
  const container = document.querySelector(`.chips[data-filter="${filterKey}"]`);
  container.innerHTML = "";
  const active = state.filters[filterKey];
  values.forEach(v => {
    const chip = document.createElement("button");
    chip.className = "chip" + (active.has(v) ? " on" : "");
    chip.dataset.filter = filterKey;
    chip.dataset.value = v;
    chip.innerHTML = `<span class="dot-color" style="background:${colorFn(v)}"></span>${labelFn(v)}`;
    chip.addEventListener("click", () => toggleFilter(filterKey, v, chip));
    container.appendChild(chip);
  });
  document.getElementById("cnt-" + (filterKey === "vex_category" ? "vex" : filterKey === "palette_families" ? "palette" : filterKey)).textContent = values.length;
}

function toggleFilter(key, value, chipEl){
  const set = state.filters[key];
  if (set.has(value)) { set.delete(value); chipEl.classList.remove("on"); }
  else { set.add(value); chipEl.classList.add("on"); }
  applyFilters();
}

function applyFilters(){
  const f = state.filters;
  const q = state.query.trim().toLowerCase();
  let n = 0;
  state.data.flags.forEach((flag, i) => {
    let ok = true;
    if (f.vex_category.size && !f.vex_category.has(flag.vex_category)) ok = false;
    if (ok && f.region.size){
      // Mars flags don't have a region; treat them as their own "region" for filter purposes.
      const r = flag.kind === "mars" ? "Mars" : flag.region;
      if (!f.region.has(r)) ok = false;
    }
    if (ok && f.kind.size && !f.kind.has(flag.kind)) ok = false;
    if (ok && f.palette_families.size){
      const fam = flag.families || [];
      if (!fam.some(x => f.palette_families.has(x))) ok = false;
    }
    if (ok && q){
      const hay = `${flag.name} ${flag.id} ${flag.vex_category}`.toLowerCase();
      if (!hay.includes(q)) ok = false;
    }
    state.visibleMask[i] = ok;
    if (ok) n++;
  });
  filterCount.textContent = n;
  refreshHighlights();
}

function wireFilters(){
  // "Clear filters" — fully empty all chip sets. Escape hatch from the
  // sovereign-only default. (Reset, by contrast, restores defaults.)
  clearBtn.addEventListener("click", () => {
    Object.values(state.filters).forEach(s => s.clear());
    $$(".chip.on").forEach(c => c.classList.remove("on"));
    searchInp.value = "";
    state.query = "";
    applyFilters();
  });
}

function syncChipsFromState(){
  // Reconcile the chip .on classes with state.filters Sets.
  $$(".chip").forEach(chip => {
    const k = chip.dataset.filter;
    const v = chip.dataset.value;
    if (!k) return;
    chip.classList.toggle("on", state.filters[k]?.has(v));
  });
  searchInp.value = state.query || "";
}

function wireSearch(){
  searchInp.addEventListener("input", (e) => {
    state.query = e.target.value;
    applyFilters();
  });
}

// ─── Toolbar (projection + color) ─────────────────────────────────────────
function wireToolbar(){
  $$(".toolbar .seg").forEach(seg => {
    const ctrl = seg.dataset.control;
    seg.querySelectorAll(".seg-btn").forEach(btn => {
      btn.addEventListener("click", () => {
        seg.querySelectorAll(".seg-btn").forEach(b => b.classList.remove("active"));
        btn.classList.add("active");
        if (ctrl === "projection") setProjection(btn.dataset.value);
        if (ctrl === "color") setColorMode(btn.dataset.value);
      });
    });
  });
}

function setProjection(p){
  if (p === state.projection) return;
  state.projection = p;
  // Animate each sprite to its new position.
  const targets = state.sprites.map((s) => {
    const c = s.userData.flag[p];
    return new THREE.Vector3(c[0]*4, c[1]*4, c[2]*4);
  });
  const from = state.sprites.map((s) => s.position.clone());
  const t0 = performance.now();
  const D = 900;
  function step(){
    const t = Math.min(1, (performance.now()-t0)/D);
    const e = t<0.5 ? 2*t*t : 1-Math.pow(-2*t+2,2)/2;
    state.sprites.forEach((s,i) => s.position.lerpVectors(from[i], targets[i], e));
    if (t<1) requestAnimationFrame(step);
  }
  step();
}

function setColorMode(mode){
  state.colorBy = mode;
  applyColors();
  buildLegend();
}

function applyColors(){
  if (!state.discs) return;
  const tmp = new THREE.Color();
  for (let i = 0; i < state.sprites.length; i++){
    const f = state.data.flags[i];
    tmp.set(colorFor(state.colorBy, f));
    state.discs.setColorAt(i, tmp);
  }
  if (state.discs.instanceColor) state.discs.instanceColor.needsUpdate = true;
}

function buildLegend(){
  const items = legendItems(state.colorBy);
  legend.innerHTML = items.map(([label, color]) =>
    `<div class="row"><span class="swatch" style="background:${color}"></span>${label}</div>`
  ).join("");
}

function legendItems(mode){
  if (mode === "vex_category"){
    const cats = state.data.meta.vex_categories;
    return cats.map(c => [formatTradition(c), TRADITION_COLORS[c] || FALLBACK]);
  }
  if (mode === "region"){
    const r = [...state.data.meta.regions, "Mars"];
    return r.map(c => [c, REGION_COLORS[c] || (c === "Mars" ? "#ff5a48" : FALLBACK)]);
  }
  if (mode === "kind"){
    return state.data.meta.kinds.map(c => [c, KIND_COLORS[c] || FALLBACK]);
  }
  if (mode === "palette"){
    return state.data.meta.palette_families.map(c => [c, PALETTE_COLORS[c] || FALLBACK]);
  }
  return [];
}

// ─── Tour ─────────────────────────────────────────────────────────────────
const TOURS = {
  nordic: { filter: { vex_category: ["nordic_cross"] }, focus: "nordic_cross" },
  british:{ filter: { vex_category: ["british_ensign"] }, focus: "british_ensign" },
  latin:  { filter: { vex_category: ["latin_charge"] }, focus: "latin_charge" },
  communist:{ filter: { vex_category: ["communist_red"] }, focus: "communist_red" },
  mars:   { filter: { kind: ["mars"] }, focus: "mars" },
};

function wireTour(){
  $$(".tour-card .tour-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      const key = btn.parentElement.dataset.tour;
      runTour(key);
    });
  });
}

function runTour(key){
  const t = TOURS[key];
  if (!t) return;
  // Clear current filters, then apply the tour's
  Object.values(state.filters).forEach(s => s.clear());
  $$(".chip.on").forEach(c => c.classList.remove("on"));

  for (const [k, vals] of Object.entries(t.filter)){
    for (const v of vals){
      state.filters[k].add(v);
      const chip = document.querySelector(`.chip[data-filter="${k}"][data-value="${v}"]`);
      if (chip) chip.classList.add("on");
    }
  }
  applyFilters();

  // Smooth scroll to the canvas, then center camera on cluster centroid.
  document.getElementById("viz").scrollIntoView({ behavior: "smooth", block: "start" });
  setTimeout(() => {
    const idxs = state.visibleMask.map((b,i) => b ? i : -1).filter(i => i >= 0);
    if (!idxs.length) return;
    const centroid = new THREE.Vector3();
    idxs.forEach(i => centroid.add(state.sprites[i].position));
    centroid.divideScalar(idxs.length);

    // Estimate cluster spread to pick a sensible camera distance.
    let radius = 0;
    idxs.forEach(i => { radius = Math.max(radius, state.sprites[i].position.distanceTo(centroid)); });
    const dist = Math.max(radius * 2.2, 3.5);
    const dir = new THREE.Vector3(0.6, 0.6, 1).normalize();
    const camTo = centroid.clone().add(dir.multiplyScalar(dist));
    state.controls.autoRotate = false;
    animateCamera(camTo, centroid, 1.4);
  }, 450);
}

// ─── Go ────────────────────────────────────────────────────────────────────
boot_app().catch(err => {
  console.error(err);
  boot.querySelector(".boot-text").textContent = "failed to load — " + err.message;
});
