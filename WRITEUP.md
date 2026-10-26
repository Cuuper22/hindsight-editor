# Hindsight Editor: write-up

> **Round 3 update (retry).** A bug in linear correction (ridge bias) and a transformer code path that
> silently used the propagated-only delta confounded rounds 1–2. With both fixed and every method trained
> by the same Adam update plus a frozen-body control (RESULTS.md, "Round 3"): local hindsight reaches 97.4%
> on MNIST, tying random feedback (97.7%); propagating hindsight across layers is below the control in every
> variant, including EMA slow copies; and on CIFAR and enwik8 every local-feedback signal (hindsight and DFA)
> is worse than training only the output layer. The distinctive claim, that an answer-aware forward pass
> gives deep layers better credit than random feedback, is not supported. The earlier editors were mostly
> training the output layer.

**Question.** Can a small learned network (the *editor*), fed only a gradient-free *hindsight delta*
(the difference between a normal forward pass and a second pass that is shown the answer), train a
network as well as backprop + Adam, at matched FLOPs?

**Answer from this run: no, at any scale tested.** The editor trains small MLPs to within 2.4-3.6
points of backprop accuracy once it is meta-trained, but at matched FLOPs its held-out loss is 2.4-3x
Adam's on MNIST and 2x on CIFAR-10, it cannot train a 10.9M-parameter character transformer below the
unigram baseline, it diverges on the ~163M FineWeb-Edu model, and it does not improve itself when
allowed to self-edit. Every number is in [RESULTS.md](RESULTS.md) (mean ± std over 3 seeds unless
stated; E4 is 1 seed).

## Gap to backprop at each rung

| rung | best Hindsight Editor | backprop + Adam | gate |
|---|---|---|---|
| E0 plumbing (MNIST MLP) | delta aligned with -grad: cos +0.14 / +0.22 / +0.70 (layers 0/1/logits) | n/a | **pass** (weakly, in hidden layers) |
| E1 imitation, editor-only training (MNIST) | 87.9% test acc (untrained editor); imitation of Adam makes it worse (69.5%) | 98.4% | **fail** (-10.5 pts) |
| E2 meta-trained, MNIST MLP | 96.0% (FTP-style sequential delta) / 94.8% (parallel); loss at 0.21 TFLOP 0.268 vs 0.096 | 98.4% | **fail** (2.8x loss at matched FLOPs) |
| E2 meta-trained, CIFAR-10 CNN | 44.0%; loss at 16.7 TFLOP 1.76 vs 0.88 | 74.0% | **fail** |
| E3 char transformer, enwik8 (10.9M) | 3.51 nats = unigram (5.07 bpc) | 1.114 nats (1.61 bpc) | **fail** (does not learn) |
| E4 ~163M GPT, FineWeb-Edu, 98M tokens | diverged: 1,076-1,654 nats from step 500 on (run lost at 4,600/7,700) | 4.316 nats (AdamW, lr 6e-4) | reported: fails |
| E5 self-editing (MNIST) | self-edited editor 76.1 ± 23.9% vs frozen 95.3% | n/a | **fail** (unstable, no self-improvement) |

**Central measurement (handoff 1d).** On MNIST, with shared initial networks, Adam's held-out loss
drop per update over the first 25 updates is 0.098 vs 0.041 for the best meta-trained editor (~40%),
and Adam has the lower held-out loss at every matched FLOP budget from 0.01 TFLOP to 1.26 TFLOP. The
editor costs 0.18 GFLOP per update vs Adam's 0.21 (two forward passes + editor vs forward + backward
+ optimizer), so the cheaper update never compensates for its lower quality. **(b) never matches (a).**

## What worked

- **The hindsight delta carries real error information**, once three things are fixed:
  1. *Linear correction.* The raw parallel delta encodes the answer y, not the error y - p, so it keeps
     pushing even when the network is right (raw rule: 15% on MNIST, diverges). Regressing each layer's
     delta on the injected code per batch and subtracting the part the blind pass already predicts,
     d~ = d - (p - 1/K)M, makes it vanish as p -> y. This is new, needs no gradient, and keeps the
     passes parallel.
  2. *The answer injected into the logits must be the answer itself* (identity output channel); a
     learned or random map there rotates the error and drives the logits the wrong way.
  3. *Per-example error weighting* from the blind pass's own loss (forward-only).
  With these, the delta's weight-level alignment with -grad on a partly trained MNIST net is
  +0.11 / +0.13 / +0.64 (layers 0/1/logits), similar to FTP-style sequential targets.
- **A randomized range finder** (project the exact weighted update onto its r dominant output
  directions) keeps a rank-16 edit's alignment intact; random-sign batch sketches destroyed it.
- **Meta-training helps substantially on MLPs**: 87.9% (untrained) -> 90.0% -> 94.8% -> 96.0% with
  longer unrolls (16), warm-started trajectories, rank 16, a slow meta learning rate, divergence
  resets and skip-on-nonfinite. Once meta-trained, the hindsight signal beats random feedback by ~1
  point (but random feedback is ~45% cheaper per update, so at matched FLOPs they are close), and the
  sequential (FTP-style) sighted pass beats the parallel one by ~1.2 points.

## What failed, and why

- **Phase 0 imitation is ill-posed for a low-rank editor.** Adam's update is element-wise normalised;
  a rank-16 edit cannot express it, and matching its size concentrates all the energy into 16
  directions (edits 3-7x too large, network blows up). Imitating the gradient direction alone is
  stable but adds nothing over the untrained editor.
- **Propagating hindsight through layers is unstable.** A layer trained toward its own amplified
  response to the injected answer forms a Hebbian loop (dW proportional to W da a^T): any propagation weight
  alpha > 0 diverges, even for a meta-trained editor. So the working editor uses only the local
  injection at each layer, which makes it close to a learned form of direct feedback alignment.
- **Width breaks the editor.** On a synthetic bigram language the same untrained editor learns
  almost as well as Adam with a tiny GPT (d=64: 1.06 vs 0.92 nats), still works with context 256, is
  mostly broken at depth 6, and fails at width 384 (stays at unigram). The likely cause is that a
  rank-r edit captures a shrinking share of the useful update as layers widen. This is why E3
  (d=384) and E4 (d=768) do not learn; the step size was a second, fixed problem (default ~10x too
  small for transformers).
- **Gradient-free self-editing is noise-driven.** Using the realised held-out improvement of each
  edit as the editor's own training signal makes its step size random-walk (up to divergence, or down
  to a frozen network); it costs 76% more FLOPs and never beats the frozen editor.
- **CNNs**: the MLP meta-training recipe does not transfer (44% -> 33% with the round-2 recipe), and
  conv DFA diverged at every learning rate tried.

## Honest caveats

- E4 is a single seed and only 98M tokens; it is reported as-is ("report whatever happens"), with no
  meta-training at that scale. The editor's VM was lost at step 4,600 of 7,700 (no error logged); it had
  diverged by step 500, so it was not resumed.
- E3 wave A ran with a bug (6 of 7 transformer hindsight channels untrained); it was found, fixed,
  unit-tested, and all later E3 runs use the fixed code. MLP/CNN rungs were unaffected.
- Wall-clock comparisons are confounded: many jobs share one GPU, and the editor path is unfused
  Python, so wall-clock plots are shown but not used for conclusions. FLOPs are analytic, counted
  identically for every method (fwd, bwd = 2x fwd, optimizer 12 flops/param, editor and hint/readout costs).
- Each rung got at most one or two focused fixes before moving on, as the handoff specifies; the
  fixes are listed in RESULTS.md with what they changed.
- Compute: about 88 Colab units by the ledger (of ~371), including ~13 lost to launch failures on an
  unreliable network before the launcher was hardened. Ledger: [runs/ledger.csv](runs/ledger.csv).

## What would be worth trying next

- A width-robust edit: block-diagonal or per-head low-rank edits, or a rank that scales with width,
  to test whether the width failure is rank starvation.
- Meta-training with less biased long-horizon estimators (PES / truncated ES) instead of 16-step
  backprop through the unroll, and meta-training across widths.
- The sequential (FTP-style) pass was the best signal on MNIST; a hybrid with linear correction is
  untested.
