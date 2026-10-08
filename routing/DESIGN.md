# Routing agent: design of the RL system

**Status as of 2026-09-27.** This describes the code in `routing/` as it
stands: `env.py` (environment), `bandit.py` (first learned policy),
`policies.py` (baselines). Items marked **Known issue** are real limitations
still in the code. Items marked **Placeholder** are values not yet measured.

## 1. The decision problem

A labeling campaign starts with a large pool of unlabeled five-band scenes and
a SegFormer-B0 that segments water. Scenes arrive in batches. For every scene
in a batch the agent picks one of three routes:

| action | what happens | annotator cost |
|---|---|---|
| `MANUAL` | a person labels the scene; the true (gold) mask enters the training set | that scene's measured labeling time |
| `AUTO` | the automatic labeler's mask (SAM2.1-L, spectrally prompted) enters the training set, right or wrong | a short review (**Placeholder: 5 s**) |
| `SKIP` | nothing enters the training set | 0 |

After the batch is routed, SegFormer is retrained on everything labeled so far
and scored on a fixed validation set. The agent wants the most segmentation
accuracy for the fewest annotator minutes.

These are three actions, `{Auto, Manual, Skip}`.
The annotator tool's route vocabulary is finer than this. `interactive_clicks`
(a person steering SAM2) is how every timed gold mask was actually made, so
`MANUAL` stands for that route and is charged its measured time.

## 2. Episode and step

- **Step = one batch** of `K = 4` candidate scenes.
- **Episode = one campaign** of `T = 4` batches, on top of a starting set
  `S0` of 4 scenes that are labeled for free.
- **Horizon is finite and undiscounted** (gamma = 1). The episode ends after
  batch `T`.

**Why the campaign, not the batch, is the episode.** Here the campaign is the episode, because a batch's value depends on what is
already labeled. For example, a third frame from a session already covered
twice is worth less than the first frame from a new session. That dependence
across batches is the reason to use RL at all. Setting `steps = 1` gives a
one-batch episode, which is then a contextual bandit.

`reset(seed)` draws a campaign from the 35-scene training pool: a random
permutation gives `S0` (4 scenes) and then the `T` batches (16 scenes), 20
scenes in all. The same seed gives every policy the same `S0` and the same
batches in the same order, so policies can be compared campaign by campaign.

## 3. State

### 3.1 The true (Markov) state

What fully determines everything that follows:

    s_t = ( L_t, B_t, Q_t, t, M_t, seed )

- `L_t`: the labeled set, a list of (scene, mask source) pairs, where the
  source is `gold` or `auto`. The source matters: the same scene labeled by
  SAM2 or by a person trains a different model.
- `B_t`: the current batch of candidates.
- `Q_t`: the batches still to come.
- `t`: the step index. `M_t`: annotator minutes spent so far.
- `seed`: the campaign seed, which is also the retrain seed.

SegFormer is **retrained from scratch** on `L_t` every step, not fine-tuned
from the previous step's weights. So the model, and its validation score, is a
function of `(L_t, seed)` alone:

    theta_t = Train(L_t, seed)      mIoU_t = Score(theta_t, V)

where `V` is the fixed validation set. With `test_groups` set, the val split is
divided by session: `V` (reward) keeps the other sessions, and a test set of
the named sessions scores every retrained model but never enters the reward.
Standard split since 2026-09-27: test = Summer 2025 + UFO006
(5 scenes), reward = Brooklyn Bridge Park + Onondaga (6 scenes). Retraining from scratch is what makes
`L_t` a sufficient statistic for the model: no history of how the set was
built leaks in. GPU nondeterminism makes `Train` slightly stochastic; retrains
of the same set vary by about 0.023 mIoU (sd over 4 seeds, measured).

### 3.2 What the agent observes

The agent does not see `L_t`, `Q_t` or the validation labels. It sees a
summary, so formally this is a POMDP, and the observation is designed to carry
what the reward depends on. For each candidate `i` in `B_t`, 17 numbers:

| group | features | computed from |
|---|---|---|
| scene appearance (8) | NDWI positive fraction and mean, NIR dark fraction at 0.25, thermal Otsu separability, Sobel edge density, gray histogram entropy, NIR mean, thermal std | the TIFF only (`rl_features.scene_features`) |
| current model on this scene (3) | mean predictive entropy, fraction of pixels with max softmax below 0.6, predicted water fraction | a forward pass of `theta_t` |
| auto-mask trust (2) | IoU between `theta_t`'s prediction and the SAM2 mask; SAM2 mask water fraction | label-free: the two automatic opinions compared with each other |
| cost (1) | *predicted* minutes to label it by hand (the median for now) | the cost model; the actual time is charged but never observed |
| session redundancy (3) | share of `L_t` from this scene's capture session; count of `L_t` from it; other candidates in `B_t` from it | capture-session names, known without labels |

Plus 4 campaign-level numbers shared by all candidates: progress `t/T`,
`|L_t|`, current `mIoU_t`, and minutes spent `M_t`.

Why these, briefly:
- **Uncertainty** stands in for how much a scene would teach the model (the
  RIPU idea, at the whole-image level).
- **Model-versus-SAM2 IoU** stands in for whether the SAM2 mask can be trusted.
  The gold mask is never used, since at decision time it doesn't exist.
- **Session redundancy** was added after measuring its effect. In one campaign,
  adding 3 near-duplicate Brooklyn frames to a 4-session starting set dropped
  validation mIoU from 0.600 to 0.462, six times the retrain noise.

`M_t` makes the cost part of the state Markov. `mIoU_t` is observed because
real campaigns would know it: the validation set is labeled once, up front.

## 4. Transition

Given actions `a_1..a_K` for the batch `B_t`:

    L_{t+1} = L_t  +  { (i, gold) : a_i = MANUAL }  +  { (i, auto) : a_i = AUTO }
    M_{t+1} = M_t  +  sum_i c(i, a_i)
    B_{t+1} = next batch of Q_t          (fixed at reset, unknown to the agent)
    theta_{t+1} = Train(L_{t+1}, seed),  mIoU_{t+1} = Score(theta_{t+1}, V)

Skipped scenes leave the campaign for good. They don't come back later.

The retrain is the training package's full recipe: 40 epochs over `L`,
AdamW, cosine schedule, cross-entropy + Lovasz loss, 512-pixel crops with
random rescale, fp16 on the RTX 3080, about 10 s per retrain. Band
normalization statistics are fixed for the whole campaign, so every retrain
sees identically normalized inputs. The score is always the **final** model's,
never the best of several evaluations. The score is the reward, and the best
of noisy evaluations on 11 validation scenes would be biased upward.

A cheaper retrain was benchmarked and rejected. Ranking agreement with a full
retrain, one run against one run: full vs itself 0.93, warm-start 150 steps 0.82,
warm-start 60 steps 0.75, frozen encoder 0.71, any variant scored at lower
resolution 0.35 to 0.65 (`results/bench-finetune-*`).

## 5. Reward and penalty

Per step:

    r_t = alpha * ( mIoU_{t+1} - mIoU_t )  -  lam * sum_i c(i, a_i) / 60

- **The accuracy term** is `alpha = 100`, so one reward unit is one mIoU point on
  the validation set. It can be negative. Training on a wrong auto mask, or
  on redundant frames that pull the model toward one session, lowers mIoU and is
  penalized with no special-case term.
- **The time penalty** is `lam` reward units per annotator minute (default 1).
  `c(i, MANUAL)` is the scene's measured seconds, `c(i, AUTO)` is 5 s
  (**Placeholder**), and `c(i, SKIP) = 0`.

The return telescopes:

    G = sum_t r_t = alpha * ( mIoU_final - mIoU_S0 ) - lam * total minutes

so the agent is scored on the accuracy it added and the time it spent, not on
how the gains were spread across steps.

**A bad-accept penalty is implicit here.** An earlier draft
had an explicit term for accepting a bad auto mask: `-eta * u * (1 - Q)`, written
eta, not gamma, since gamma is the discount. The environment has no such term,
because its effect is measured directly. A bad auto mask in `L` lowers the
retrained model's validation mIoU, and the accuracy term charges it.

**`lam` is not calibrated yet, and it matters.** Retrain noise (sd 0.023
mIoU) is about 2.3 reward units per step, while a hand-labeled scene costs
about 0.27 units at `lam = 1`. At that setting annotator time barely registers
next to accuracy noise. Choosing `lam`, that is, what a minute of annotator
time is worth in mIoU, is still open. Report results for more than one `lam`.

## 6. Known issues and placeholders

1. ~~The per-scene manual cost is visible before the decision.~~ **Fixed
   2026-09-27.** The agent now observes and decides on
   `CostModel.predicted()` (the median manual time), and the environment
   charges the actual measured time. Runs before this fix (`baselines20`,
   `bandit-20260927`) had the leak. Its effect was small at `lam = 1`.
2. **The AUTO cost of 5 s is a placeholder.** No valid timing for accepting a
   SAM2 mask exists yet. It needs the planned labeling session.
3. **Class weights and normalization see the pool's gold masks.**
   `Campaign.stats` is measured over the whole 35-scene pool, including its
   water fraction for loss class weights. This is a small leak of label
   information that a real campaign would have to estimate.
4. **SAM2's masks look better than they are.** 34 of the gold masks were made
   by steering SAM2 with clicks, so SAM2's automatic masks share their
   boundaries (median IoU against gold 0.84).
5. **Small data.** 35 pool scenes and 11 validation scenes from 4 held-out
   sessions. Campaigns reuse the same scenes in new combinations, so a
   learned policy generalizes across campaign compositions, not to unseen
   scenes. Campaigns are not independent samples.
6. **Retrain noise** of about 0.023 mIoU per step makes single-campaign returns
   noisy. Compare policies on the same campaigns (paired) and over many of them.

## 7. Policies

### 7.1 Baselines (`policies.py`)

`all_manual`, `all_auto`, `all_skip`, `random`, plus two label-free rules:
- **Disagreement(tau):** `MANUAL` if IoU(model, SAM2) is below tau, else `AUTO`.
- **Uncertainty(frac):** the most uncertain `frac` of the batch goes to
  `MANUAL`, the rest to `AUTO`, or to `SKIP` in the `+skip` variant. This is
  the RIPU-style greedy rule at image level.

Result over 20 campaigns (`results/baselines20-20260927/`), paired against
`all_manual`: `uncert0.5+skip` +3.8 +/- 5.3 return, better on 15 of 20
campaigns. Every other heuristic, including `all_auto` and `random`, is within
noise of `all_manual`. `all_skip` is worst.

### 7.2 Contextual bandit (`bandit.py`)

The reward arrives per batch, so credit is assigned to scenes by assuming the
batch's gain is a sum of per-scene contributions:

    alpha * dmIoU_t  ~  sum_i  w_{a_i} . phi(x_i, g_t)        (SKIP contributes 0)

`phi` is the standardized 17 candidate features, the 4 campaign features and a
bias. There is one weight vector `w_a` for each of `AUTO` and `MANUAL`.
- **Learning:** Bayesian linear regression on the batch sums, with the noise
  sd set to 3.3 from the measured retrain spread.
- **Deciding:** each scene takes `argmax_a  w_a . phi_i - lam * c(i, a)/60`, with
  `SKIP = 0`. The time term is not learned; the cost model gives it directly.
- **Exploration:** Thompson sampling during training, drawing `w` from the
  posterior. Greedy on the posterior mean for evaluation.
- **Protocol:** 5 uniform-random campaigns fit the feature scaling and seed the
  posterior. It then trains on campaigns 100 to 159 and is evaluated on
  campaigns 0 to 19, the same ones the ladder uses, never seen in training.

**Result (2026-09-27, `results/bandit-20260927/`, 240 posterior updates),
greedy on the 20 evaluation campaigns:**

| compared with | paired return difference | better on | final mIoU | minutes |
|---|---|---|---|---|
| `uncert0.5+skip` (best heuristic) | +6.7 +/- 7.4 | 16 / 20 | 0.635 vs 0.579 | 1.8 vs 2.4 |
| `all_manual` | +10.5 +/- 8.8 | 18 / 20 | 0.635 vs 0.561 | 1.8 vs 4.4 |

Its action mix is 29% `MANUAL`, 23% `AUTO` and 47% `SKIP`. It skips the scenes
SAM2 already gets right (median SAM2 IoU 0.96), which are the ones the model
already handles. The heuristic sends half to `MANUAL` and never uses `AUTO`.
The largest weights are on **scene-appearance** features (NIR dark
fraction, NIR mean, NDWI positive fraction, thermal), not on model uncertainty
or session redundancy.

**Read that result carefully.** Appearance features largely identify the kind
of scene and its session. The reward is measured on one fixed 11-scene
validation set from 4 sessions, and evaluation campaigns reuse the same 35
pool scenes. So the bandit has largely learned which kinds of pool scenes help
on *these* validation sessions. That is legitimate within this environment, but
the margin is probably optimistic for a new node or new sessions. Two checks
are needed before it is claimed:
1. Score the final models on a **test set** of scenes that never enter the
   reward.
2. Evaluate with **held-out sessions**: train the bandit on campaigns drawn from
   some sessions, and evaluate on campaigns from others.

**Result of those checks (2026-09-27, `results/split-*`, `results/loso-*`): the
bandit's advantage does not carry over to sessions outside the reward.**
Paired against `uncert0.5+skip` over 20 campaigns at `lam = 1` (mean +/- standard error):

| policy | reward-set return diff | better on | TEST gain diff | better on |
|---|---|---|---|---|
| bandit | +5.2 +/- 1.6 | 16/20 | **-6.0 +/- 1.6** | 3/20 |
| bandit, Kaitlyn held out in training | +7.4 +/- 2.5 | 13/20 | -2.5 +/- 2.0 | 8/20 |
| bandit, a second session held out in training | +9.8 +/- 3.0 | 15/20 | -0.4 +/- 2.7 | 10/20 |
| all_manual | -0.9 +/- 1.2 | 8/20 | -2.8 +/- 1.5 | 6/20 |

- On the 6 reward scenes the bandit wins clearly, even with a pool session
  held out of training.
- On the 5 test scenes it is *worse* than the heuristic. Test gains are small
  for every policy: even `all_manual` gains -0.5 +/- 3.1 points on test.
- **Across 220 campaigns, gain on the reward scenes and gain on the test scenes
  correlate at only 0.15** (Pearson; Spearman 0.14).

So the reward, as defined on 6 scenes from 2 sessions, measures fit to those 2
sessions. A policy that maximizes it learns which pool scenes look like them.
**The limit is the reward's validation set, not the policy class.** A stronger
learner (DQN) on this reward would overfit it further. The fix is a larger and
more session-diverse reward set: the planned evaluation set
(at least 25 masks; 11 exist).

**Swapped-roles check (2026-09-28, `results/swap-*`)**, with reward = Summer 2025
+ UFO006 (5 scenes) and test = Brooklyn Bridge Park + Onondaga
(6 scenes):

| | original split | swapped split |
|---|---|---|
| reward gain vs test gain, per campaign (Pearson, 180 campaigns) | 0.12 | 0.17 |
| policy ranking, reward vs test (Spearman over 9 policies) | **-0.75** | **-0.75** |
| bandit vs heuristic, reward set | +5.2 +/- 1.6 (16/20) | +9.2 +/- 2.8 (15/20) |
| bandit vs heuristic, test set | -6.0 +/- 1.6 (3/20) | -0.7 +/- 2.1 (9/20) |

- **The weak transfer is symmetric,** so the first pairing was not unlucky.
- **The policy ranking inverts in both directions.** Routing that helps one pair
  of sessions hurts the other. In the swapped run, the policies that label the
  most (`all_manual`, `uncert0.5`, `disagree<0.5`) gain most on the test
  sessions (+10 to +11 points). They were near the bottom on the reward sessions.
- **The bandit wins on whichever pair it is rewarded on, and never on the other.**

Conclusion: the objective depends on which sessions define it. A reward set of
2 sessions cannot stand for the network. The reward set must span every
session the model is meant to serve, and results should be reported per
session. Testing generalization then needs labeled scenes from sessions that
are in neither set.

 at `lam = 5` the bandit beats the
heuristic on the reward set by +12.3 +/- 1.6 (19/20), by labeling less
(1.3 min). At `lam = 20` time dominates: `all_skip` scores +38.3 over the
heuristic and the bandit only +2.7 more, labeling 0.6 min per campaign. On the
test set the ordering at every `lam` is the same as above.

It values each batch by its immediate gain only, with no lookahead to later
batches. Its additive credit assignment also can't represent interactions
within a batch, except through the batch-redundancy feature. It is the bandit
rung of the bandit-versus-MDP question. A DQN over the same state,
valuing `L_t` for future batches, is the next rung. Whether it beats the
bandit answers whether the sequential structure matters in practice.

## 8. Where things live

| what | where |
|---|---|
| environment, policies, bandit, inner loop | `photo_processing/routing/` |
| auto masks, per-scene env table | `routing/results/auto_masks-20260927/` |
| benchmarks, baseline and bandit results | `routing/results/` (copied back from the 3080; kept locally, not in the repo) |
| runs | a project directory on the 3080, `source env.sh` first |
| retrain memo (score + fp16 weights per labeled set) | `data/retrain_cache.jsonl`, `data/retrain_ckpt/` on the 3080 |
