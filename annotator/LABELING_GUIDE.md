# UFONet Water Labeling Guide

**Version:** 0.1 (DRAFT — decisions marked `[DECIDE]` need your sign-off)
**Last updated:** 2026-09-01
**Applies to:** masks produced by `photo_processing/annotator/`, exported via
`export_dataset.py --gold-only`, used to fine-tune the 5-band SegFormer.

---

## Why this document exists

SegFormer cannot be more consistent than its labels. Every systematic choice
made here — include ice or not, where exactly the waterline goes — gets learned
faithfully and shows up in every downstream number. Deciding once, up front, is
far cheaper than re-labeling 50 scenes later.

It also matters for the RL routing work: `auto_vs_final_iou` (the `Q_i` term in
the reward) only means "how good was the auto mask" if *human* masks are
consistent. Inconsistent labeling shows up as noise in `Q_i` and the policy
learns nothing.

**The operational question this dataset answers:** *is there water present in
this scene where it affects the ground, and which pixels are it.* Not "is this
surface damp", not "is this H₂O in any form". Every rule below follows from that.

---

## Classes

| value | class | meaning |
|---|---|---|
| `0` | background | everything else, including sky |
| `1` | water | standing or flowing water on/over the ground surface |

Binary, 2-class. The annotator paints `1`; `export_dataset.py` writes
single-channel PNGs with exactly these values.

---

## Decision table

Each row is a call I'd make and why. **Change any of them** — this is a
starting point to react to, not a fait accompli.

| # | Case | Call | Rationale |
|---|---|---|---|
| 1 | **Snow / ice, frozen surface** | `[DECIDE]` **background** | The node exists to detect flood hazard; frozen surfaces aren't one, and NIR physically separates them (snow reflects, water absorbs) so the model *can* learn the distinction. Counter-argument: a hydrology framing would call a frozen pond "water". If you want that, it should be a **third class**, not merged — merging destroys the NIR signal that makes 5-band worth having. |
| 2 | **Partially frozen — open water next to ice** | label the open water only | The waterline moves; this is exactly the transition a flood camera should track. |
| 3 | **Wet pavement / dark sheen, no depth** | `[DECIDE]` **background** | "Wet" vs "flooded" is *the* operational distinction. If damp asphalt counts as water, every rainy scene is 60% water and the class stops meaning anything. |
| 4 | **Standing water on pavement (visible puddle, reflection, depth)** | **water** | This is the flood signal. |
| 5 | **Sky reflected on a water surface** | **water** | Those pixels *are* water; they just happen to be specular. Excluding them punches holes in every calm-water mask. |
| 6 | **Shadowed water** | **water** | Illumination doesn't change class. |
| 7 | **Water seen through thin occluders** (railings, reeds, bare branches) | label the water surface; **don't cut around every twig** | Below the mask's useful resolution, and the effort is enormous for no model benefit. Cut around occluders only when they're solid and larger than ~20 px wide. |
| 8 | **Large solid foreground objects over water** (pole, car, person, boat hull) | **background** | Above the waterline = not water. |
| 9 | **Partially submerged objects** | water surface around them = water; the object above the waterline = background | |
| 10 | **Rain droplets / spray on the lens** | **background** | An imaging artifact, not scene water. If labeled as water, the model learns to fire on a dirty lens — a false-positive source in exactly the weather where you need it to work. |
| 11 | **Sky, clouds, fog** | **background** | Always. The spectral backend's horizon cut already does this. |
| 12 | **Distant water (far shore, horizon)** | **water** | The class is "is this water". Range/usability is the georeferencing pipeline's job, not the segmenter's — don't bake an arbitrary distance cut into the labels. |
| 13 | **Water in containers** (bottle, bucket, birdbath) | **background** | Not ground-surface water; rare enough that including it only adds confusion. |
| 14 | **Very small blobs** | **background** if < ~20 px area **or** < ~2 px thick | Below the model's effective resolution at 512²; adds label noise and drags boundary metrics down for no gain. |
| 15 | **Thin sheet flow across pavement** | `[DECIDE]` **water** if you can see it, subject to rule 14 | This is early-flood signal and worth catching — but it's also the case the spectral backends miss most, so expect to draw it by hand. |

---

## Boundary convention

- Put the waterline **at the visible edge of the water surface** — where the
  reflective / NIR-dark surface visibly ends.
- On soft edges (vegetated bank, gradual shore) follow the surface, not the
  vegetation line. Being consistent matters more than being precise.
- **A few pixels of slop is expected and fine.** The training pipeline can
  ignore a dilated boundary ring, so don't spend time pixel-polishing edges.
  Spend it on more scenes instead.

---

## When to reject a scene

Reject (**Reject scene** button) rather than guess. A wrong gold label is worse
than a missing one — it corrupts training *and* the RL reward signal.

- Co-registration failed (bands visibly misaligned)
- Lens obscured, fogged, or heavily rain-covered
- Exposure blown out or too dark to judge
- Indoor / bench-test / accidental captures
- **You genuinely can't tell whether it's water** — ambiguity is a reject, not a coin flip

Rejections are logged with their scene features, so the router learns which
scenes aren't worth an annotator's time. They're never exported as training data.

---

## Workflow notes

- **Colour-IR** is the best default view — water goes dark. Cross-check with
  **NDWI** and **NIR** when unsure; **thermal** helps at night.
- Seed from `sam`/`sam2` when the auto mask is close, and correct it. That's
  the `auto_edited` route and it's the cheapest good label.
- **Don't accept an auto mask you haven't actually looked at.** A rubber-stamped
  `auto_accepted` is a silver label wearing a gold badge and it poisons both the
  training set and `Q_i`.
- High **ensemble disagreement** = the scene is genuinely hard. Slow down there;
  those are the scenes worth the most.

---

## Held-out validation set

Set aside **~20–40 scenes labeled entirely by hand** (no auto seed) as a
validation set, spanning conditions: clear water, ice, snow, night/thermal,
rain, empty (no water). This is the only trustworthy mIoU. The auto train/val
split in `--gold-only` mixes seeded masks into val, so its numbers measure
"does SegFormer imitate SAM", not "is SegFormer right".

`[DECIDE]` how to mark these — simplest is a dedicated `roots` entry (e.g.
`val_scenes/`) so they never mix.

---

## Change log

Record every rule change here — a dataset labeled under two different rulebooks
is worse than either one alone. If a rule changes, note which scenes predate it.

| date | change | scenes affected |
|---|---|---|
| 2026-09-01 | Initial draft, no scenes labeled yet | — |
