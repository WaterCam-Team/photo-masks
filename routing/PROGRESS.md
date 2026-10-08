# Progress log: labelling-route RL project

Dated log of the RL course project in `routing/`: an agent that decides which
labelling route each image takes, trading mask quality against annotator time.
Newest entry first. Add an entry each time a run finishes, a decision is made
or the scope changes. Say what was done, what it showed and where the results
are. The design lives in [DESIGN.md](DESIGN.md) and the labelling session in
the labeling plan (`LABELING_PLAN.md`, kept locally). This file records what happened and when.

**Last updated:** 2026-10-01

## Current status

| | |
|---|---|
| **Problem (as of the 2026-09-28 scope cut)** | Budgeted queue MDP: 12 images per episode, actions {MANUAL, SAM2}, frozen SegFormer-B0, reward 100·q per step |
| **Built** | Queue env + DP optimum, 5 baselines, Sarsa (+ `sarsa_x` ablation), metrics A-C, `check_queue.py` 12/12; also the old retrain-loop env and bandit, the frame selector, and eval-set enforcement in the annotator |
| **Not built** | Nothing on the critical path. The real run needs data, not code |
| **Blocked on** | The labelling session (R / T / P sets): proposals exist, live lists not saved. Also a decision on the budget rule (README, "Open design issue") |
| **Code state** | `routing/` and the annotator changes are **uncommitted** |

## Next steps

1. [x] Frame lists reviewed and saved (2026-10-02): `eval_sets.csv` holds 36 reward and 25
   test frames, including 11 relabels; `pool_queue.csv` holds 21. Annotator changes committed
   (`f951617`), and enforcement verified on a live server.
2. [ ] Labelling session: R and T from scratch with four classes, P with the
   AUTO timing protocol. Estimate 1.5–3.5 h. Label a few R scenes first to
   replace the time estimate.
3. [ ] Regenerate the manifest and rerun `gen_auto_masks` / `prep_env_data` on
   the new scenes.
4. [x] Build the queue env and the hindsight DP optimum (ceiling). (2026-10-01)
5. [x] Baselines: all-SAM2, random, first-come, RIPU-threshold, RIPU-ranked. (2026-10-01)
6. [ ] Decide the budget rule: proposal vs review-inclusive.
7. [ ] Run linear semi-gradient Sarsa on the real R / T data. (Code done 2026-10-01.) Report the share of the (optimum − all-SAM2)
   gain it captures, paired against RIPU-threshold, at budget ρ = 0.25 and 0.5,
   per session.
8. [ ] Commit `routing/` and the annotator eval-set changes.

---

## 2026-10-08

**Labelling started; fallback triggered.** Five frames were saved. The two Brooklyn
Bridge Park reward frames (15:45 and 15:49) came out 100% background because of the
drag-stroke bug, fixed in `c296eeb` at 16:09. They must be redone, and the redo rows'
timing excluded, keeping the first attempts' 440 s and 238 s. The three later frames are
correct. Median active time is about 7 min a frame, over the proposal's 3-minute rule, so
`eval_sets.csv` was trimmed to **30 R / 20 T**: 11 relabels, all of Summer 2025 and every
labelled frame kept, 4 frames per UFO007 placement. The backup is
`eval_sets.csv.bak-20261008`. The one P frame labelled so far was drawn from scratch, so
it has no `review_seconds`; the remaining P frames must use SAM2 Run.

## 2026-10-01

**Queue MDP code built** on the 46 existing gold scenes (README): `queue_features.py`,
`queue_env.py`, `queue_policies.py`, `run_queue.py`. `optimum()` matches exhaustive
simulation on 60 queues. Not yet aligned with the revised proposal: no SAM2 review cost
c_s, the predicted manual time is the median instead of a linear fit, the RIPU threshold
is a quantile rather than grid-tuned, and Sarsa runs one seed with its own ε schedule.

**Code aligned with the revised proposal** (same day):
- c_s charged on every SAM2 step.
- c_hat is a linear fit on 4 label-free features, fitted on training scenes only.
- The RIPU-threshold tau is grid-tuned on training queues.
- Sarsa: 5 seeds, eps 0.2 -> 0.01 over 5,000 episodes, alpha grid, learning curves.
- DP optimum over (t, b) in `queue_dp.py`, checked against enumeration.
- `check_queue.py` passes 12/12.
- `sarsa_x` (interaction features) is kept as a labelled ablation.

**Found while aligning: the proposal's budget rule conflicts with c_s.** B = rho*N*c_bar
cannot also pay for reviews. At c_s = 5 s, all-SAM2 alone overruns B = 48 s at rho = 0.25.
`--budget review_inclusive` is the alternative. **Decision needed.**

**First full runs** (`results/queue-{proposal,reviewincl}-cs5-20261001/`). Interim data,
c_s = 5 s placeholder, session folds A (eval Brooklyn Dec 2025) and B (the reverse), 100
paired queues each.
- **Success criterion met in 1 of 8 cells:** review-inclusive budget, fold B, rho = 0.5.
  There Sarsa beats RIPU-threshold by +49.2 [+29.5, +68.8], but only ties First-come
  (+0.6 [-15.3, +16.6]).
- **First-come is the strongest baseline.** In fold A the tuned tau is 0, so RIPU-threshold
  *is* First-come. In fold B, First-come beats RIPU-threshold by +21 to +49. This matches
  RIPU's ~0 correlation with SAM2 failure.
- **Sarsa learns on its training sessions but does not transfer.** Held-out training return
  rises from 521 to 840-965 (fold A) and from 1040 to 1120-1160 (fold B). In fold A's
  evaluation it then *underspends*: 1.4-2.4 MANUAL per queue against 4-10 for First-come,
  with up to half the budget left unused. With budget left at the last image, MANUAL is
  always at least as good. A policy linear in the budget features evidently cannot
  represent "spend what is left before the queue ends" under the fold's feature shift.
- **sarsa_x collapsed to never-MANUAL** in one cell (proposal budget, fold A, rho = 0.25):
  its learning curve stayed flat at 521.
- None of this is the project's result. The data is the circular interim set, c_s is a
  placeholder, and the folds are not the R / T design.

**Frozen-SegFormer features** (`results/queue-features-20261001/`): 8 leave-one-session-out
SegFormer-B0 models, 40 epochs each, on CPU (~2 h). RIPU follows the paper's region
definition (3×3, k = 1). The image mean is ~0.39 × the fraction of predicted-boundary pixels.

**RIPU barely predicts SAM2 failure on this data.** Spearman against 1 − q_SAM2 over 46 scenes:
- RIPU +0.08
- entropy +0.04
- low-confidence fraction +0.10
- IoU(SegFormer, SAM2) −0.89

RIPU fails worst where the model predicts no water at all. On UFO006 and Onondaga
it predicts 0–2% water, so there is no boundary and RIPU ≈ 0. SAM2 is poor there anyway
(mean q 0.24 and 0.55). A boundary-based score reads "model missed the water" as "model is
certain".

The −0.89 is partly circular: most gold masks were made by steering SAM2, and SegFormer was
trained on them (DESIGN.md issue 4). The R / T sets labelled from scratch are what will
test it.

## 2026-09-28

**Second scope cut (document of record: `67-proposal-revised.pdf`).**
- SegFormer-B0 is no longer retrained in the loop. It is trained once and frozen,
  and supplies only features and the image-level RIPU score.
- New problem: an episode is a queue of N = 12 images under a budget
  B = ρ·N·median manual seconds. Each step is one image, with actions
  {MANUAL, SAM2}. MANUAL is allowed only while budget remains.
- State: 7 label-free image features, plus budget left, images left, and budget
  per remaining image.
- Reward: 100·q per step, where q = 1 for MANUAL and IoU(SAM2, hand mask) for
  SAM2. Time enters only through the budget.
- `env.py` and `bandit.py` (the retrain loop) are off the critical path. The
  four-class routing env is dropped: labels stay four-class, and the env uses
  collapsed water masks.

**Labelling session planned** (`LABELING_PLAN.md`, kept locally).
- Sets: R (reward, ~35 scenes from 7 sessions), T (test, ~25: UFO007, plus two
  sites used nowhere else) and P (pool additions, ~30, which time the AUTO route).
- Sessions are atomic: each one sits wholly in R, T or P.
- Night frames are dropped.
- The UFO007 record was reduced to its outdoor deployment periods. The rest is
  lab, bench, indoor or fogged-lens frames.

**Tools built for the session.**
- `select_frames.py` proposes frames per session: daytime only, duplicates
  merged, a 30-min gap for node series, farthest-point sampling on scene
  features, and the old seeded validation scenes always included for relabelling.
- Annotator: scenes in `eval_sets.csv` open blank. Every auto seed is withheld
  in the UI and refused by the server (403 / 409).
- `export_dataset` and `training.manifest` take R and T from the sidecar, never
  from the hash split. They keep no-water eval scenes and keep eval sessions
  out of training.
- New log columns:
  - `review_seconds`: the measured AUTO cost.
  - `seg_model`: the checkpoint the uncertainty columns came from.
  - `eval_set`
- SegFormer uncertainty is now computed server-side for every save.

**Swapped-roles check** (`results/swap-*`). Reward = Summer 2025 +
UFO006, test = Brooklyn Bridge Park + Onondaga.
- Reward-vs-test gain correlation: 0.17 (0.12 in the original split).
- The policy ranking inverts in both directions (Spearman −0.75).
- The bandit wins on whichever pair it is rewarded on and never on the other.
- **Conclusion:** the objective depends on which sessions define it. A
  2-session reward set cannot stand for the network.

## 2026-09-27

**Inner-loop benchmark** (`results/bench-finetune-{cpu,gpu}-20260927/`, RTX 3080).
- Do fast fine-tunes rank batches the way a full retrain does? No.
- Single run vs single run, Spearman ρ against a full retrain:

  | variant | ρ |
  |---|---|
  | full retrain (against itself) | 0.93 |
  | warm-start, 150 steps | 0.82 |
  | warm-start, 60 steps | 0.75 |
  | decoder only | 0.71 |
  | any variant scored at long side 640 | 0.35–0.65 |

- A 2-seed CPU run had suggested otherwise. With 4 seeds, that turned out to be luck.
- **Decision:** full retrain every step, score the final model (never best-of),
  fix band statistics per campaign.

**Auto masks generated** (`results/auto_masks-20260927/`). SAM2.1-L, FastSAM and
spectral masks for 46 gold scenes, and `env_scenes.json` with features, measured
manual time and auto-mask quality.

**Retrain-loop environment** (`env.py`). The episode is a campaign of 4
batches of 4 scenes, after 4 free starting scenes. Actions {MANUAL, AUTO, SKIP}.
Reward = 100·ΔmIoU − λ·minutes.
- Fixed the same day: the per-scene manual cost was visible to the agent before
  it decided. It now observes the predicted (median) time and is charged the
  actual time.

**Baselines over 20 campaigns** (`results/baselines20-20260927/`).
`uncert0.5+skip` was best: +3.8 ± 5.3 return against `all_manual`, better on
15/20. Every other heuristic was within noise.

**Contextual bandit** (`bandit.py`, `results/bandit-20260927/`): Bayesian linear
regression with Thompson sampling and 240 posterior updates. Compared with
`uncert0.5+skip`:

| | bandit | `uncert0.5+skip` |
|---|---|---|
| paired return difference | +6.7 ± 7.4 | — |
| better on | 16/20 campaigns | — |
| final mIoU | 0.635 | 0.579 |
| annotator minutes | 1.8 | 2.4 |

Its largest weights are on scene-appearance features, which hints that it
learned which scenes resemble the validation sessions.

**Generalisation checks** (`results/split-*`, `results/loso-*`).
- On the test scenes the bandit's advantage disappears: −6.0 ± 1.6 against the
  heuristic, 3/20.
- Reward-set gain and test-set gain correlate at 0.15 across 220 campaigns.
- **Conclusion:** the limit is the reward's validation set, not the policy class.
  A DQN would overfit it further. This finding motivated the labelling session.

**Label log cleaned for cost data.** 13 pilot rows were excluded from timing in
`log_exclusions.csv` (decided 2026-09-23). Their masks stay gold.

## Before 2026-09-27 (groundwork in the main pipeline)

- **2026-09-04:** SAM2 click sessions logged as their own route,
  `interactive_clicks`, with seed IoU and edit fraction. This is the route
  every timed gold mask was made with.
- **2026-09-04:** the `training/` package and leave-one-session-out CV. The
  baseline to beat is SegFormer-B0 5-band at 0.709 ± 0.220 mIoU.
