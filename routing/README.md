# routing: the labeling-route RL agent

Full design (state, transition, reward, known issues): [DESIGN.md](DESIGN.md).
Next labeling session (eval sets, AUTO timing): the labeling plan is kept locally
(`LABELING_PLAN.md`, not in the repo, since it names sites). `select_frames.py` reads its
session plan from a local `sessions.local.json`; see `sessions.example.json` for the format.
Run outputs under `results/` (checkpoints, auto masks, logs) also stay local.

**Current design (since 2026-09-28): no SegFormer retrain in the loop.** SegFormer-B0 is trained once, frozen, and only supplies features
and the image-level RIPU score. The new problem:
- episode = queue of N=12 images with a budget B = rho*N*median manual seconds
  (rho 0.25, 0.5), plus review time for the rest; step = one image; actions {MANUAL, SAM2}, MANUAL only while budget > 0;
- state = 7 label-free image features (RIPU, entropy, low-conf, pred water, IoU
  SegFormer-vs-SAM2, SAM2 water frac, predicted manual time) + budget left, images left,
  budget per remaining image;
- reward every step = 100*q: 1 for MANUAL, IoU(SAM2, hand mask) for SAM2; time enters
  only through the budget;
- baselines: all-SAM2, random, first-come (gamma=0), RIPU-threshold (main), RIPU-ranked
  (sees whole queue), hindsight DP optimum (ceiling); learner = linear semi-gradient Sarsa;
- primary metric: share of (optimum - all-SAM2) gain captured, paired vs RIPU-threshold.
`env.py`/`bandit.py` (retrain loop) are no longer on the project's critical path.
The annotator is not changed.

**Queue MDP built 2026-10-01.** It runs on
the 46 existing gold scenes until R and T are labelled:

```
queue_features.py   frozen SegFormer-B0 (one per held-out session on the interim data;
                    --train-groups for one frozen model), the state features
                    incl. image-level RIPU, and the c_hat regressors -> queue_scenes.json
queue_env.py        QueueEnv: budget B, MANUAL only while b > 0 (charged measured c(x)),
                    SAM2 charged c_s, reward 100 q; CostModel (c_hat, linear fit on
                    training scenes); optimum() by enumerating all 2^N MANUAL subsets
queue_dp.py         the same optimum by DP over (t, b_t), in tenths of a second
queue_policies.py   all_sam2, random, first_come, ripu_threshold (tau grid-tuned on
                    training queues), ripu_ranked, sarsa (S&B 10.1), sarsa_x (ablation)
run_queue.py        100 paired eval queues per fold and budget, full metrics -> summary.md
check_queue.py      environment and optimum checks (12/12 PASS)
```

Run order: `python -m routing.queue_features ...`, then `python -m routing.check_queue`,
then `python -m routing.run_queue --data <dir>/queue_scenes.json --out <dir> --review-s 5
--review-placeholder`. The last two flags stay until review timings exist; after that,
c_s is read from the label log.

Design choices worth knowing, all reported in `summary.md`:
- **`sarsa_x` is an extra ablation.** The main `sarsa` uses standardized features plus
  a bias only.
- **The optimum is computed twice,** by subset enumeration and by DP, and the run stops if
  the two disagree.
- **Interim folds are by session:** A fits on everything except `Brooklyn Dec 2025` and is
  scored on it, B the reverse. With T labelled, use `--eval-groups`.

**Budget rule.** Every SAM2 step is charged the review time c_s, so the budget includes
it: B = rho*N*c_bar + (1 - rho)*N*c_s (`--budget review_inclusive`, the default). That
buys rho*N hand labels plus a review of every other image. `--budget labels_only`
(B = rho*N*c_bar) lets the reviews eat the hand-labeling time: at c_s = 5 s, reviewing
a queue of 12 (60 s) costs more than all of B at rho = 0.25 (48 s).

RIPU follows the RIPU paper's (Xie et al. 2022) region-based definition: a 3x3 window, i.e. its k = 1, with
uncertainty as the mean entropy in the window. The image score is the frame mean, which
is ~0.39 x the predicted-boundary fraction.

**Status 2026-09-28 (before the cut):** env, baselines and bandit built and run. The bandit wins on its reward
sessions but not on held-out sessions (DESIGN.md 7.2), so the next step is the labeling session in
the (local) labeling plan, then a four-class env and a rerun with reward on R and test on T. No DQN yet.

For each scene in a batch, decide whether a person labels it (MANUAL), the
auto labeler does (AUTO), or nobody does (SKIP), trading SegFormer accuracy
against annotator minutes. Workstation/GPU-box code: nothing here runs on a
sensor node.

## Pipeline

```
annotator label_log + gold masks
  -> gen_auto_masks.py   what AUTO would return (SAM2.1-L, FastSAM, spectral), once per scene
  -> prep_env_data.py    env_scenes.json: features, measured manual time, auto-mask quality
  -> env.py              RoutingEnv: campaign = episode, batch = step, full SegFormer retrain per step
  -> policies.py         hand-designed baselines
  -> run_episodes.py     roll policies over identical campaigns, paired summary
finetune.py              the inner loop (train on a labeled set, score on the fixed val set)
bench_finetune.py        does a fast fine-tune rank batches like a full one? (it does not)
```

## Decisions and why

- **Full retrain every step.** Benchmarked on the 3080, single run vs single
  run, Spearman rho of batch ranking against a full retrain: full vs itself
  0.93, warm-start 150 steps 0.82, 60 steps 0.75, frozen-encoder decoder 0.71,
  any variant scored at long side 640 0.35-0.65. A 2-seed CPU run had
  suggested the fast variants were as good; 4 seeds showed that was luck.
  Results: `results/bench-finetune-{cpu,gpu}-20260927/`.
- **Score the final model, never best-of.** The score is a reward; the max over
  noisy evals on 11 val scenes is biased upward.
- **Band statistics fixed per campaign** (`Campaign.stats`), so every retrain
  normalises identically.
- **Episode = campaign, step = batch.** An episode of one batch is the special
  case `steps=1`. A batch's value depends on what is already
  labeled, which is the reason for RL over a bandit.

## Cost model: what is and is not measured

MANUAL = the scene's own `active_seconds` from a timing-valid log row
(`rl_features.timing_rows()`, 34 scenes, all `interactive_clicks`), else their
median. **AUTO = 5 s is a placeholder**; no timing-valid auto rows exist yet.
Calibrating it needs the planned labeling session.

## Known properties of the environment

- **Labeling can hurt.** Near-duplicate frames from one session pull the model
  away from the held-out val sessions (seed 0: 0.600 -> 0.462 mIoU after 3 more
  Brooklyn frames, retrain sd ~0.023). SKIP is a real option, not a formality.
- **Retrain noise is ~0.023 mIoU** = 2.3 reward points at alpha 100, against
  ~0.27 min per manual scene. At lam = 1 time barely registers; lam matters.
- Episodes resample one small labeled set; they are not independent.
- 34 gold masks were made by steering SAM2, so SAM2's auto masks look closer to
  gold than they would to an independent labeler.

## Running on the 3080

Everything lives in one project directory on the 3080 (a shared account: work
only there). `source env.sh` first: it keeps uv/HF/torch caches inside that
directory and sets `HF_HUB_OFFLINE=1`.

```
python -m routing.run_episodes --data data --out results/<name> --seeds 0 1 2 3 4
```
