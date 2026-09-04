"use strict";
/* UFONet Water Annotator — single-page canvas labeler.
   Full-res binary mask in a Uint8Array (0 / 255); painted / auto-seeded / cleaned,
   saved as water_mask.png. Every save/reject POSTs a labeling-decision record. */

const $ = (s) => document.querySelector(s);

/* Always resolves to an object with at least {ok}. Network / non-JSON / HTTP
   errors come back as {ok:false, error:"…"} so every call site's `if (!r.ok)`
   handles them uniformly instead of throwing. */
async function api(p, opt) {
  let r;
  try {
    r = await fetch(p, opt);
  } catch (e) {
    return { ok: false, error: "network error — is the server running? (" + e.message + ")" };
  }
  const txt = await r.text();
  try {
    return JSON.parse(txt);
  } catch {
    return { ok: false, error: `HTTP ${r.status}: ${txt.slice(0, 300) || r.statusText}` };
  }
}

const WATER_RGB = [0, 200, 255];
const IDLE_MS = 12000;          // activity older than this doesn't count as "active"
const UNDO_CAP = 30;
// per-row preview colours; distinguishable under common colour-vision deficiencies
const PREVIEW_COLOURS = [
  [255, 255, 255], [0, 190, 200], [245, 180, 0], [230, 70, 200],
  [120, 170, 255], [255, 130, 60], [150, 220, 120], [200, 160, 255],
];

const S = {
  cfg: null,
  scenes: [],
  scene: null,                  // {id,name,size:[H,W], existing_masks}
  W: 0, H: 0,
  mask: null,                   // Uint8Array(W*H)  0/255
  maskImage: null,              // ImageData for #mask
  undo: [], redo: [],
  tool: "brush",
  brush: 24, tol: 14, zoom: 1,
  drawing: false, panning: false, spaceHeld: false,
  last: null,                   // last pointer px {x,y}
  strokeDirty: false,
  grayCache: {},                // layer -> Float array for fill
  // decision-tracking
  seedBackend: null,            // backend name OR on-disk filename OR null
  backendParams: {},
  segMeta: {},                  // {mean_entropy, low_conf_frac} from last segformer run
  ensemble: {},                 // {agreement, disagreement_frac, n_backends_run} from last Run all
  votesUrl: null,               // disagreement-map overlay for the current scene
  showVotes: false,             // is the disagreement map pinned on?
  imgCache: {},                 // url -> HTMLImageElement (preview hovers)
  nStrokes: 0, nUndos: 0,
  activeSeconds: 0, lastActivity: 0, timer: null,
};

/* ------------------------------------------------------------------ boot */
async function boot() {
  await loadConfig();
  wireControls();
  wireTrain();
  await refreshScenes();
  S.timer = setInterval(tick, 1000);
}

async function loadConfig() {
  S.cfg = await api("/api/config");
  const cfgName = S.cfg.config_file ? S.cfg.config_file.split("/").pop() : "built-in defaults";
  const dev = S.cfg.device || "cpu";                    // e.g. "cuda (Radeon … · ROCm 6.2)"
  const devShort = dev.split(" ")[0];
  const accel = devShort.startsWith("cuda") || devShort === "xpu";
  initAnnotatorName(S.cfg.annotator);
  $("#who").textContent = "config: " + cfgName + "  ·  " + devShort;
  if (S.cfg.device_hint && !S._hintShown) { S._hintShown = 1; toast(S.cfg.device_hint, "bad"); }
  $("#who").title = "config file: " + (S.cfg.config_file || "(none — built-in defaults)")
    + "\ndevice: " + dev + (S.cfg.device_hint ? "  [" + S.cfg.device_hint + "]" : "")
    + "\nphoto roots:\n"
    + (S.cfg.roots || []).map((r) => "  " + (S.cfg.scenes_per_root[r] ?? 0) + "  " + r).join("\n")
    + "\nTIFF names: " + (S.cfg.tiff_names || []).join(", ");
  buildBackendSelect();
  // sensible training defaults for the detected device
  if (accel && $("#tp-batch") && !$("#tp-batch").dataset.touched) {
    $("#tp-batch").value = 8;
    $("#tp-note").textContent = "Running on " + dev
      + (devShort.startsWith("cuda") ? " — fp16 mixed precision on." : ".")
      + " Raise batch further if memory allows.";
  }
}

/* ------------------------------------------------------- who is labeling
   The name is stored per browser (localStorage), not on the server, so two
   people sharing one running instance are attributed separately. It is sent
   with every save/reject; the server falls back to its --annotator value if
   the field is blank. */
const NAME_KEY = "ufonet.annotator";

function annotatorName() {
  return ($("#inp-annotator").value || "").trim();
}

function markNameState() {
  const el = $("#inp-annotator");
  const blank = !annotatorName() || annotatorName() === "anon";
  el.classList.toggle("empty", blank);
  el.title = blank
    ? "Set your name — labels are currently attributed to the server default."
    : `Labels you save are recorded as “${annotatorName()}”.`;
}

function initAnnotatorName(serverDefault) {
  const el = $("#inp-annotator");
  if (!el.value) el.value = localStorage.getItem(NAME_KEY) || serverDefault || "";
  el.oninput = () => {
    localStorage.setItem(NAME_KEY, annotatorName());
    markNameState();
  };
  markNameState();
}

function buildBackendSelect() {
  const sel = $("#sel-backend");
  const prev = sel.value;
  sel.innerHTML = "";
  for (const b of S.cfg.backends) {
    const o = document.createElement("option");
    o.value = b.name;
    o.textContent = b.label + (b.available ? "" : "  (unavailable)");
    o.disabled = !b.available;
    o.title = b.available ? "" : b.reason;
    sel.appendChild(o);
  }
  const stillOk = S.cfg.backends.find((b) => b.name === prev && b.available);
  const first = stillOk || S.cfg.backends.find((b) => b.available);
  if (first) sel.value = first.name;
  renderBackendParams();
}

function renderBackendParams() {
  const b = S.cfg.backends.find((x) => x.name === $("#sel-backend").value);
  const box = $("#backend-params");
  box.innerHTML = "";
  if (!b) return;
  for (const p of b.params) {
    const l = document.createElement("label");
    l.title = p.help || "";
    let ctl;
    if (p.type === "bool") {
      ctl = document.createElement("input");
      ctl.type = "checkbox";
      ctl.checked = !!p.default;
      l.append(ctl, " " + p.name);
    } else if (p.type === "select") {
      ctl = document.createElement("select");
      for (const opt of p.options || []) {
        const o = document.createElement("option");
        o.value = o.textContent = opt;
        ctl.appendChild(o);
      }
      ctl.value = p.default;
      l.append(p.name + " ", ctl);
    } else {
      ctl = document.createElement("input");
      ctl.type = "number";
      ctl.value = p.default; ctl.min = p.min; ctl.max = p.max; ctl.step = p.step;
      l.append(p.name + " ", ctl);
    }
    ctl.dataset.pname = p.name;
    ctl.dataset.ptype = p.type || "number";
    box.appendChild(l);
  }
}

function collectParams() {
  const out = {};
  $("#backend-params").querySelectorAll("[data-pname]").forEach((el) => {
    const t = el.dataset.ptype;
    if (t === "bool") out[el.dataset.pname] = el.checked;
    else if (t === "select") out[el.dataset.pname] = el.value;
    else {
      const v = parseFloat(el.value);
      out[el.dataset.pname] = Number.isNaN(v) ? el.value : v;
    }
  });
  return out;
}

/* ------------------------------------------------------------ scene list */
async function refreshScenes() {
  const r = await api("/api/scenes");
  S.scenes = r.scenes;
  const done = S.scenes.filter((s) => s.review_status !== "pending").length;
  $("#progress").textContent = `${done} / ${S.scenes.length} reviewed`;
  const ul = $("#scene-list");
  ul.innerHTML = "";
  for (const s of S.scenes) {
    const li = document.createElement("li");
    li.dataset.id = s.id;
    if (S.scene && s.id === S.scene.id) li.className = "active";
    const nm = document.createElement("span");
    nm.className = "nm"; nm.textContent = s.name;
    const chip = document.createElement("span");
    chip.className = "chip " + s.review_status;
    chip.textContent = s.review_status === "pending" ? "•" : s.review_status.slice(0, 4);
    li.append(nm, chip);
    li.onclick = () => loadScene(s.id);
    ul.appendChild(li);
  }
}

/* --------------------------------------------------------- load a scene */
async function loadScene(id) {
  if (S.scene && S.undo.length && !confirm("Discard unsaved edits on this scene?")) return;
  const d = await api("/api/scene/" + id);
  if (d.error) return toast(d.error, "bad");
  S.scene = d;
  [S.H, S.W] = d.size;
  S.mask = new Uint8Array(S.W * S.H);
  S.undo = []; S.redo = [];
  S.grayCache = {};
  S.seedBackend = null; S.backendParams = {}; S.segMeta = {}; S.ensemble = {};
  S.votesUrl = null; S.showVotes = false; S.imgCache = {};
  $("#ensemble-panel").hidden = true;
  $("#ens-legend").hidden = true;
  $("#ens-votes").classList.remove("on");
  S.nStrokes = 0; S.nUndos = 0;
  S.activeSeconds = 0; S.lastActivity = Date.now();

  $("#placeholder").style.display = "none";
  const bg = $("#bg"), mk = $("#mask"), pv = $("#preview");
  bg.width = mk.width = pv.width = S.W;
  bg.height = mk.height = pv.height = S.H;
  S.maskImage = mk.getContext("2d").createImageData(S.W, S.H);
  clearPreview();

  // existing-mask dropdown
  const se = $("#sel-existing");
  se.innerHTML = '<option value="">—</option>';
  for (const m of d.existing_masks) {
    const o = document.createElement("option");
    o.value = m.name; o.textContent = m.label;
    se.appendChild(o);
  }

  await drawBackground();
  // auto-load a previously saved manual mask if present
  if (d.existing_masks.some((m) => m.name === "water_mask.png")) {
    await loadMaskFile("water_mask.png", /*asSeed=*/ false);
  }
  S.undo = []; S.redo = [];          // loading is not an edit
  renderMask();
  applyZoom();
  refreshScenes();
  $("#backend-status").textContent = "";
  $("#decision-hint").textContent = "";
}

async function drawBackground() {
  const layer = $("#sel-layer").value;
  const img = await loadImg(`/api/scene/${S.scene.id}/view?layer=${layer}&t=${Date.now()}`);
  const ctx = $("#bg").getContext("2d");
  ctx.clearRect(0, 0, S.W, S.H);
  ctx.drawImage(img, 0, 0, S.W, S.H);
}

/* --------------------------------------------------------------- render */
function renderMask() {
  const d = S.maskImage.data;
  const [wr, wg, wb] = WATER_RGB;
  for (let i = 0, j = 0; i < S.mask.length; i++, j += 4) {
    if (S.mask[i]) { d[j] = wr; d[j + 1] = wg; d[j + 2] = wb; d[j + 3] = 255; }
    else { d[j + 3] = 0; }
  }
  $("#mask").getContext("2d").putImageData(S.maskImage, 0, 0);
  updateWaterPct();
}

function updateWaterPct() {
  let n = 0;
  for (let i = 0; i < S.mask.length; i++) if (S.mask[i]) n++;
  const pct = (100 * n / S.mask.length).toFixed(1);
  $("#water-pct").textContent = pct + " %";
}

function applyZoom() {
  const z = S.zoom;
  for (const c of [$("#bg"), $("#preview"), $("#mask")]) {
    c.style.width = (S.W * z) + "px";
    c.style.height = (S.H * z) + "px";
  }
  $("#canvas-pad").style.width = (S.W * z) + "px";
  $("#canvas-pad").style.height = (S.H * z) + "px";
}

/* ------------------------------------------------- preview overlay layer
   A read-only canvas between the photo and the editable mask. Used for
   hover-previewing another backend's result and for the disagreement map.
   It never touches S.mask, so previewing is always non-destructive. */

function clearPreview() {
  const pv = $("#preview");
  if (pv.width) pv.getContext("2d").clearRect(0, 0, pv.width, pv.height);
}

async function cachedImg(url) {
  if (!S.imgCache[url]) S.imgCache[url] = await loadImg(url);
  return S.imgCache[url];
}

/** Draw an already-coloured RGBA overlay (the disagreement map) as-is. */
async function drawOverlay(url) {
  const img = await cachedImg(url);
  clearPreview();
  $("#preview").getContext("2d").drawImage(img, 0, 0, S.W, S.H);
}

/** Draw a binary mask PNG tinted `rgb`, at low alpha, without altering S.mask. */
async function drawMaskPreview(url, rgb) {
  const img = await cachedImg(url);
  const c = document.createElement("canvas");
  c.width = S.W; c.height = S.H;
  const cx = c.getContext("2d");
  cx.drawImage(img, 0, 0, S.W, S.H);
  const src = cx.getImageData(0, 0, S.W, S.H);
  const d = src.data;
  const [r, g, b] = rgb;
  for (let j = 0; j < d.length; j += 4) {
    if (d[j] > 127) { d[j] = r; d[j + 1] = g; d[j + 2] = b; d[j + 3] = 150; }
    else { d[j + 3] = 0; }
  }
  clearPreview();
  $("#preview").getContext("2d").putImageData(src, 0, 0);
}

/** Back to whatever should be showing when nothing is hovered. */
function restorePreview() {
  if (S.showVotes && S.votesUrl) drawOverlay(S.votesUrl).catch(() => {});
  else clearPreview();
}

/* ---------------------------------------------------------- undo / redo */
function snapshot() {
  S.undo.push(S.mask.slice());
  if (S.undo.length > UNDO_CAP) S.undo.shift();
  S.redo.length = 0;
}
function undo() {
  if (!S.undo.length) return;
  S.redo.push(S.mask.slice());
  S.mask = S.undo.pop();
  S.nUndos++;
  renderMask();
  bump();
}
function redo() {
  if (!S.redo.length) return;
  S.undo.push(S.mask.slice());
  S.mask = S.redo.pop();
  renderMask();
  bump();
}

/* --------------------------------------------------------------- paint */
function evToPx(e) {
  const r = $("#mask").getBoundingClientRect();
  return {
    x: Math.floor((e.clientX - r.left) / S.zoom),
    y: Math.floor((e.clientY - r.top) / S.zoom),
  };
}

function stamp(cx, cy, val) {
  const rad = S.brush / 2, r2 = rad * rad;
  const x0 = Math.max(0, Math.floor(cx - rad)), x1 = Math.min(S.W - 1, Math.ceil(cx + rad));
  const y0 = Math.max(0, Math.floor(cy - rad)), y1 = Math.min(S.H - 1, Math.ceil(cy + rad));
  for (let y = y0; y <= y1; y++) {
    for (let x = x0; x <= x1; x++) {
      const dx = x - cx, dy = y - cy;
      if (dx * dx + dy * dy <= r2) {
        const i = y * S.W + x;
        if (S.mask[i] !== val) { S.mask[i] = val; S.strokeDirty = true; }
      }
    }
  }
}

function paintLine(a, b, val) {
  const dist = Math.hypot(b.x - a.x, b.y - a.y);
  const steps = Math.max(1, Math.round(dist / Math.max(1, S.brush / 4)));
  for (let s = 0; s <= steps; s++) {
    const t = s / steps;
    stamp(a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t, val);
  }
}

/* ---------------------------------------------------------- flood fill */
async function grayForFill() {
  let layer = $("#sel-layer").value;
  if (!["nir", "thermal", "ndwi"].includes(layer)) layer = "nir";
  if (S.grayCache[layer]) return S.grayCache[layer];
  const img = await loadImg(`/api/scene/${S.scene.id}/view?layer=${layer}&raw=1&t=${Date.now()}`);
  const c = document.createElement("canvas");
  c.width = S.W; c.height = S.H;
  const cx = c.getContext("2d");
  cx.drawImage(img, 0, 0, S.W, S.H);
  const data = cx.getImageData(0, 0, S.W, S.H).data;
  const g = new Uint8Array(S.W * S.H);
  for (let i = 0, j = 0; i < g.length; i++, j += 4) g[i] = data[j];
  S.grayCache[layer] = g;
  return g;
}

async function floodFill(px) {
  const g = await grayForFill();
  const i0 = px.y * S.W + px.x;
  if (i0 < 0 || i0 >= g.length) return;
  const target = g[i0], tol = S.tol;
  snapshot();
  const stack = [i0], seen = new Uint8Array(g.length);
  let count = 0;
  while (stack.length) {
    const i = stack.pop();
    if (seen[i]) continue;
    seen[i] = 1;
    if (Math.abs(g[i] - target) > tol) continue;
    S.mask[i] = 255; count++;
    const x = i % S.W, y = (i / S.W) | 0;
    if (x > 0) stack.push(i - 1);
    if (x < S.W - 1) stack.push(i + 1);
    if (y > 0) stack.push(i - S.W);
    if (y < S.H - 1) stack.push(i + S.W);
  }
  renderMask();
  toast(`filled ${count.toLocaleString()} px (tol ${tol})`);
}

/* -------------------------------------------------- morphological clean */
function morph(src, grow) {
  // 3x3 dilate (grow=1) or erode (grow=0) on binary 0/255
  const out = new Uint8Array(src.length);
  const W = S.W, H = S.H;
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      let hit = grow ? 0 : 1;
      for (let dy = -1; dy <= 1 && hit === (grow ? 0 : 1); dy++) {
        const yy = y + dy; if (yy < 0 || yy >= H) { if (!grow) { hit = 0; } continue; }
        for (let dx = -1; dx <= 1; dx++) {
          const xx = x + dx; if (xx < 0 || xx >= W) { if (!grow) { hit = 0; break; } continue; }
          const v = src[yy * W + xx] ? 1 : 0;
          if (grow && v) { hit = 1; break; }
          if (!grow && !v) { hit = 0; break; }
        }
      }
      out[y * W + x] = hit ? 255 : 0;
    }
  }
  return out;
}

function fillHoles(src) {
  const W = S.W, H = S.H, outside = new Uint8Array(src.length), st = [];
  for (let x = 0; x < W; x++) { st.push(x); st.push((H - 1) * W + x); }
  for (let y = 0; y < H; y++) { st.push(y * W); st.push(y * W + W - 1); }
  while (st.length) {
    const i = st.pop();
    if (outside[i] || src[i]) continue;
    outside[i] = 1;
    const x = i % W, y = (i / W) | 0;
    if (x > 0) st.push(i - 1);
    if (x < W - 1) st.push(i + 1);
    if (y > 0) st.push(i - W);
    if (y < H - 1) st.push(i + W);
  }
  const out = src.slice();
  for (let i = 0; i < out.length; i++) if (!src[i] && !outside[i]) out[i] = 255;
  return out;
}

function clean() {
  if (!S.scene) return;
  snapshot();
  let m = S.mask;
  m = morph(m, 0); m = morph(m, 1);      // open  (despeckle)
  m = morph(m, 1); m = morph(m, 0);      // close (bridge gaps)
  m = fillHoles(m);
  S.mask = m;
  renderMask();
  toast("cleaned (open + close + fill holes)");
}

/* --------------------------------------------------------- auto-segment */
async function runBackend() {
  if (!S.scene) return toast("load a scene first", "bad");
  const name = $("#sel-backend").value;
  const params = collectParams();
  $("#backend-status").textContent = "running " + name + "…";
  bump();
  const r = await api(`/api/scene/${S.scene.id}/autoseg`, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ backend: name, params }),
  });
  if (!r.ok) { $("#backend-status").textContent = ""; return toast(r.error || "backend failed", "bad"); }
  await loadImgToMask(r.mask_url, /*asSeed=*/ true, name, params);
  if (name === "segformer") {
    S.segMeta = {
      mean_entropy: r.meta.mean_entropy,
      low_conf_frac: r.meta.low_conf_frac,
    };
  }
  const bits = [`${r.water_pct ?? "?"}% water`, `${r.elapsed_s}s`];
  if (r.meta.mean_entropy != null) bits.push(`entropy ${r.meta.mean_entropy}`);
  $("#backend-status").textContent = `${name}: ` + bits.join(" · ");
  $("#decision-hint").textContent = "seeded from " + name + " — correct, then Save";
}

async function loadMaskFile(name, asSeed) {
  await loadImgToMask(
    `/api/scene/${S.scene.id}/maskfile?name=${encodeURIComponent(name)}&t=${Date.now()}`,
    asSeed, name, {});
}

/* ------------------------------------------------------ run-all ensemble */
async function runEnsemble() {
  if (!S.scene) return toast("load a scene first", "bad");
  $("#backend-status").textContent = "running all backends…";
  bump();
  const e = await api(`/api/scene/${S.scene.id}/ensemble`, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({}),
  });
  $("#backend-status").textContent = "";
  if (!e.ok) return toast(e.error || "ensemble failed", "bad");

  S.ensemble = {
    agreement: e.agreement, disagreement_frac: e.disagreement_frac, n_backends_run: e.n_ran,
  };
  S.votesUrl = e.votes_url || null;
  S.imgCache = {};                                   // masks changed; drop hover cache

  const disPct = e.disagreement_frac == null ? null : 100 * e.disagreement_frac;
  $("#ens-summary").textContent =
    `${e.n_ran} ran · mean IoU ${e.agreement ?? "?"} · ` +
    `${disPct == null ? "?" : disPct.toFixed(0)}% of pixels contested`;
  $("#ens-summary").title =
    "Mean IoU = how much the auto-segmenters overlap (1.0 = identical, 0 = no overlap). "
    + "Contested = share of pixels some called water and others didn't.";

  // plain-language verdict, so this is readable without knowing what IoU is
  const verdict = disPct == null ? ""
    : disPct > 40
      ? "⚠ The auto-segmenters strongly disagree — treat every result as a rough draft and expect to correct it by hand."
      : disPct > 15
        ? "The auto-segmenters mostly agree. The consensus is a reasonable starting point; check the contested edges."
        : "The auto-segmenters closely agree. The consensus is probably good — still look before you save.";
  $("#ens-plain").textContent =
    (e.water_present ? "Water appears to be present. " : "No water detected in this scene. ") + verdict;

  const rows = $("#ens-rows");
  rows.innerHTML = "";
  let ci = 0;
  const addRow = (label, pct, meta, url, seedName, help) => {
    const colour = PREVIEW_COLOURS[ci++ % PREVIEW_COLOURS.length];
    const div = document.createElement("div");
    div.className = "ens-row";
    div.title = help || "";

    const sw = document.createElement("i");
    sw.className = "swatch";
    sw.style.background = `rgb(${colour.join(",")})`;

    const b = document.createElement("button");
    b.textContent = "Use this";
    b.title = `Load ${label} into the editor so you can correct and save it`;
    b.onclick = () => loadImgToMask(url, true, seedName, {});

    const txt = document.createElement("span");
    txt.textContent = `${label}` + (pct == null ? "" : `  —  ${pct}% water`)
      + (meta ? `  ·  ${meta}` : "");

    div.append(sw, b, txt);
    div.addEventListener("mouseenter", () => drawMaskPreview(url, colour).catch(() => {}));
    div.addEventListener("mouseleave", restorePreview);
    rows.appendChild(div);
  };

  if (e.consensus_url) {
    addRow("consensus (majority vote)", e.consensus_water_pct, null,
      e.consensus_url, "_annotator_ensemble.png",
      "What most of the auto-segmenters agreed on — usually the best starting point.");
  }
  for (const r of e.results) {
    if (r.error) { addRowError(rows, r.backend, r.error); continue; }
    addRow(r.backend, r.water_pct, `${r.elapsed_s}s`, r.mask_url, r.backend,
      BACKEND_HELP[r.backend] || "");
  }

  $("#ensemble-panel").hidden = false;
  $("#ens-legend").hidden = !S.showVotes;
  $("#decision-hint").textContent = disPct > 40
    ? "auto-segmenters disagree — this scene needs careful manual work"
    : "auto-segmenters mostly agree — the consensus is a good starting point";
  restorePreview();
}

const BACKEND_HELP = {
  sam2: "SAM 2.1 — highest quality, slowest. Follows the prompt very literally.",
  sam: "FastSAM — quick first pass. Looser, but often good enough.",
  spectral: "Physics-based: combines NIR darkness, NDWI, thermal smoothness and texture. Struggles with snow and ice.",
  nir: "Simple threshold on the near-infrared band. Water absorbs NIR, so it looks dark.",
  thermal: "Uses the thermal camera plus NIR. Helpful at night.",
  change: "Compares this photo against a 'dry' background built from other captures of the same view.",
  tinysam: "TinySAM — small prompt-based model.",
  segformer: "The model you are training. It improves as you label more scenes.",
};

function addRowError(rows, name, msg) {
  const div = document.createElement("div");
  div.className = "ens-row err";
  div.textContent = `${name} — could not run: ${msg}`;
  div.title = "This auto-segmenter is unavailable for this scene. It is skipped, not failed.";
  rows.appendChild(div);
}

async function loadImgToMask(url, asSeed, seedName, params) {
  const img = await loadImg(url);
  const c = document.createElement("canvas");
  c.width = S.W; c.height = S.H;
  const cx = c.getContext("2d");
  cx.drawImage(img, 0, 0, S.W, S.H);
  const data = cx.getImageData(0, 0, S.W, S.H).data;
  snapshot();
  for (let i = 0, j = 0; i < S.mask.length; i++, j += 4) S.mask[i] = data[j] > 127 ? 255 : 0;
  if (asSeed) { S.seedBackend = seedName; S.backendParams = params || {}; }
  renderMask();
}

/* -------------------------------------------------------- save / reject */
function maskToPngDataUrl() {
  const c = document.createElement("canvas");
  c.width = S.W; c.height = S.H;
  const cx = c.getContext("2d");
  const im = cx.createImageData(S.W, S.H);
  for (let i = 0, j = 0; i < S.mask.length; i++, j += 4) {
    const v = S.mask[i] ? 255 : 0;
    im.data[j] = im.data[j + 1] = im.data[j + 2] = v; im.data[j + 3] = 255;
  }
  cx.putImageData(im, 0, 0);
  return c.toDataURL("image/png");
}

function session() {
  return {
    annotator: annotatorName(),
    seed_backend: S.seedBackend,
    backend_params: S.backendParams,
    active_seconds: S.activeSeconds,
    n_strokes: S.nStrokes,
    n_undos: S.nUndos,
    seg_mean_entropy: S.segMeta.mean_entropy ?? "",
    seg_low_conf_frac: S.segMeta.low_conf_frac ?? "",
    ensemble_agreement: S.ensemble.agreement ?? "",
    ensemble_disagreement_frac: S.ensemble.disagreement_frac ?? "",
    n_backends_run: S.ensemble.n_backends_run ?? "",
  };
}

async function save() {
  if (!S.scene) return;
  $("#btn-save").disabled = true;
  const r = await api(`/api/scene/${S.scene.id}/save`, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ mask_png_b64: maskToPngDataUrl(), session: session() }),
  });
  $("#btn-save").disabled = false;
  if (!r.ok) return toast(r.error || "save failed", "bad");
  const iou = r.auto_vs_final_iou == null ? "" : ` · IoU ${r.auto_vs_final_iou.toFixed(3)}`;
  toast(`saved — route: ${r.route}${iou} · ${S.activeSeconds}s`, "ok");
  await advance(r.next);
}

async function reject() {
  if (!S.scene) return;
  if (!confirm("Mark this scene unusable? It is logged and excluded from training.")) return;
  const r = await api(`/api/scene/${S.scene.id}/review`, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ status: "rejected", session: session() }),
  });
  if (!r.ok) return toast(r.error || "failed", "bad");
  toast("scene rejected", "ok");
  await advance(r.next);
}

async function advance(nextId) {
  await refreshScenes();
  S.undo = [];                 // saved -> no "discard edits" prompt
  if (nextId) loadScene(nextId);
  else toast("no pending scenes left", "ok");
}

/* --------------------------------------------------------------- timer */
function bump() { S.lastActivity = Date.now(); }
function tick() {
  if (S.scene && Date.now() - S.lastActivity < IDLE_MS) S.activeSeconds++;
}

/* --------------------------------------------------------------- toast */
let toastT;
function toast(msg, kind) {
  const t = $("#toast");
  t.textContent = msg;
  t.className = "show " + (kind || "");
  clearTimeout(toastT);
  toastT = setTimeout(() => (t.className = ""), 3200);
}

/* ---------------------------------------------------------- image util */
function loadImg(url) {
  return new Promise((res, rej) => {
    const im = new Image();
    im.onload = () => res(im);
    im.onerror = () => rej(new Error("img load failed: " + url));
    im.src = url;
  });
}

/* ------------------------------------------------------------- wiring */
function wireControls() {
  $("#sel-layer").onchange = () => { drawBackground(); };
  $("#rng-opacity").oninput = (e) => { $("#mask").style.opacity = e.target.value; };
  $("#mask").style.opacity = $("#rng-opacity").value;
  $("#rng-zoom").oninput = (e) => { S.zoom = parseFloat(e.target.value); applyZoom(); };
  $("#sel-backend").onchange = renderBackendParams;
  $("#btn-run").onclick = runBackend;
  $("#btn-ensemble").onclick = runEnsemble;
  $("#ens-close").onclick = () => {
    $("#ensemble-panel").hidden = true;
    S.showVotes = false;
    $("#ens-votes").classList.remove("on");
    $("#ens-legend").hidden = true;
    clearPreview();
  };
  $("#ens-votes").onclick = () => {
    if (!S.votesUrl) return toast("run “Run all” first", "bad");
    S.showVotes = !S.showVotes;
    $("#ens-votes").classList.toggle("on", S.showVotes);
    $("#ens-votes").textContent = S.showVotes ? "Hide disagreement map" : "Show disagreement map";
    $("#ens-legend").hidden = !S.showVotes;
    restorePreview();
  };
  $("#btn-load").onclick = () => {
    const n = $("#sel-existing").value;
    if (n) loadMaskFile(n, true);
  };
  $("#btn-refresh").onclick = async () => { await loadConfig(); await refreshScenes(); };
  $("#btn-undo").onclick = undo;
  $("#btn-redo").onclick = redo;
  $("#btn-clean").onclick = clean;
  $("#btn-clear").onclick = () => { if (S.scene) { snapshot(); S.mask.fill(0); renderMask(); } };
  $("#btn-save").onclick = save;
  $("#btn-reject").onclick = reject;
  $("#btn-skip").onclick = async () => {
    if (!S.scene) return;
    const r = await api(`/api/scene/${S.scene.id}/review`, {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ status: "skip" }),
    });
    advance(r.next);
  };

  document.querySelectorAll(".tool").forEach((b) => {
    b.onclick = () => {
      document.querySelectorAll(".tool").forEach((x) => x.classList.remove("active"));
      b.classList.add("active");
      S.tool = b.dataset.tool;
    };
  });

  const mk = $("#mask");
  mk.addEventListener("pointerdown", (e) => {
    if (!S.scene) return;
    bump();
    if (S.spaceHeld || e.button === 1) { S.panning = true; S.last = { x: e.clientX, y: e.clientY }; return; }
    const px = evToPx(e);
    if (S.tool === "fill") { floodFill(px); return; }
    S.drawing = true; S.strokeDirty = false;
    snapshot();
    S.last = px;
    stamp(px.x, px.y, S.tool === "erase" ? 0 : 255);
    renderMask();
    mk.setPointerCapture(e.pointerId);
  });
  mk.addEventListener("pointermove", (e) => {
    if (S.panning) {
      const w = $("#canvas-wrap");
      w.scrollLeft -= e.clientX - S.last.x;
      w.scrollTop -= e.clientY - S.last.y;
      S.last = { x: e.clientX, y: e.clientY };
      return;
    }
    if (!S.drawing) return;
    bump();
    const px = evToPx(e);
    paintLine(S.last, px, S.tool === "erase" ? 0 : 255);
    S.last = px;
    renderMask();
  });
  const endStroke = () => {
    if (S.drawing && S.strokeDirty) S.nStrokes++;
    else if (S.drawing && !S.strokeDirty) S.undo.pop();   // no-op stroke: drop snapshot
    S.drawing = false; S.panning = false;
  };
  mk.addEventListener("pointerup", endStroke);
  mk.addEventListener("pointerleave", endStroke);

  window.addEventListener("keydown", (e) => {
    if (e.target.tagName === "INPUT" || e.target.tagName === "SELECT") return;
    if (e.code === "Space") { S.spaceHeld = true; return; }
    const k = e.key.toLowerCase();
    if (k === "b") setTool("brush");
    else if (k === "e") setTool("erase");
    else if (k === "f") setTool("fill");
    else if (k === "u") undo();
    else if (k === "r") redo();
    else if (k === "[") setBrush(S.brush - 4);
    else if (k === "]") setBrush(S.brush + 4);
    else if (e.key === "Enter") save();
  });
  window.addEventListener("keyup", (e) => { if (e.code === "Space") S.spaceHeld = false; });
  window.addEventListener("beforeunload", (e) => {
    if (S.scene && S.undo.length) { e.preventDefault(); e.returnValue = ""; }
  });

  $("#rng-brush").oninput = (e) => { S.brush = parseInt(e.target.value, 10); };
  $("#rng-tol").oninput = (e) => { S.tol = parseInt(e.target.value, 10); };
}

/* ------------------------------------------------------------- train panel */
const T = { polling: null };

function startPolling() {
  pollTrain();
  if (!T.polling) T.polling = setInterval(pollTrain, 2500);
}
function stopPolling() {
  if (T.polling) { clearInterval(T.polling); T.polling = null; }
}

function wireTrain() {
  $("#btn-train").onclick = () => {
    const p = $("#train-panel");
    p.hidden = !p.hidden;
    if (!p.hidden) startPolling();
  };
  $("#tp-close").onclick = () => {
    $("#train-panel").hidden = true;
    if (!T.lastRunning) stopPolling();     // keep polling only to catch an active run finishing
  };
  $("#tp-batch").oninput = (e) => { e.target.dataset.touched = "1"; };

  $("#tp-export").onclick = async () => {
    $("#tp-dataset").textContent = "exporting…";
    const r = await api("/api/train/export", { method: "POST", headers: { "Content-Type": "application/json" }, body: "{}" });
    if (!r.ok) { $("#tp-dataset").textContent = "export failed — " + (r.log || "").split("\n").slice(-2).join(" "); return; }
    const routes = Object.entries(r.routes || {}).map(([k, v]) => `${k}:${v}`).join("  ");
    $("#tp-dataset").textContent = `gold dataset: ${r.train} train · ${r.val} val   ${routes}`;
  };

  $("#tp-start").onclick = async () => {
    const body = JSON.stringify({
      epochs: +$("#tp-epochs").value, lr: +$("#tp-lr").value,
      img_size: +$("#tp-size").value, batch: +$("#tp-batch").value,
    });
    const r = await api("/api/train/start", { method: "POST", headers: { "Content-Type": "application/json" }, body });
    if (!r.ok) return toast(r.error || "start failed", "bad");
    toast("fine-tune started: run " + r.run, "ok");
    startPolling();
  };
  $("#tp-stop").onclick = async () => { await api("/api/train/stop", { method: "POST" }); startPolling(); };

  $("#tp-use").onclick = async () => {
    const r = await api("/api/train/use", { method: "POST" });
    if (!r.ok) return toast(r.error || "failed", "bad");
    toast("segformer backend now uses the fine-tuned model", "ok");
    await loadConfig();
  };
  $("#tp-onnx").onclick = async () => {
    $("#tp-onnx-msg").textContent = "exporting ONNX…";
    const r = await api("/api/train/export-onnx", { method: "POST" });
    $("#tp-onnx-msg").textContent = r.ok ? "ONNX: " + r.path
      : (r.log || "").split("\n").filter(Boolean).slice(-1)[0] || "export failed";
  };
}

async function pollTrain() {
  const s = await api("/api/train/status");
  if (!s || s.ok === false) return;
  T.lastRunning = s.running;
  const chip = $("#tp-phase");
  chip.className = "chip " + (s.phase === "idle" ? "pending" : s.phase);
  chip.textContent = s.phase + (s.run ? " · " + s.run : "");
  $("#tp-stop").hidden = !s.running;
  $("#tp-start").disabled = s.running;

  const last = s.steps[s.steps.length - 1];
  const bits = [];
  if (s.device) bits.push(`device ${s.device}${s.gpu ? " · " + s.gpu : ""}${s.amp ? " · fp16" : ""}`);
  if (last) bits.push(`step ${last.step}  loss ${last.loss}  lr ${last.lr}`);
  if (s.best_miou != null) bits.push(`best mIoU ${(+s.best_miou).toFixed(3)}`);
  if (s.error) bits.push("ERROR: " + s.error);
  if (s.info && s.info.length) bits.push(...s.info.map((x) => "· " + x));
  $("#tp-live").textContent = bits.join("\n");

  $("#tp-evals").innerHTML = "";
  for (const e of s.evals.slice(-8)) {
    const d = document.createElement("div");
    if (e.best) d.className = "best";
    d.textContent = `epoch ${e.epoch}:  mIoU ${(+e.miou).toFixed(3)}   water ${e.iou_water != null ? (+e.iou_water).toFixed(3) : "?"}${e.best ? "   ← best" : ""}`;
    $("#tp-evals").appendChild(d);
  }
  $("#tp-done").hidden = !s.has_ckpt;

  // stop polling once terminal, unless the panel is open (so a new run can start it again)
  if (!s.running && ["done", "error", "stopped", "idle"].includes(s.phase)
      && $("#train-panel").hidden) {
    stopPolling();
  }
}

function setTool(t) {
  S.tool = t;
  document.querySelectorAll(".tool").forEach((x) => x.classList.toggle("active", x.dataset.tool === t));
}
function setBrush(v) {
  S.brush = Math.max(2, Math.min(120, v));
  $("#rng-brush").value = S.brush;
}

boot().catch((e) => toast(String(e), "bad"));
