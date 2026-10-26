# Hindsight Editor: results

Living document; every number links to a run directory under `runs/` (mean ± std over seeds).
Compute ledger: [runs/ledger.csv](runs/ledger.csv).

## E0: plumbing (MNIST MLP 784-256-256-10, 3 seeds, local RTX 3060)

Cosine between the hindsight delta d_l and the backprop direction -dL/dz_l, per example on 2,000
held-out images, along a backprop (Adam) training trajectory. Full table: [runs/e0_table.md](runs/e0_table.md).

| variant | step | layer 0 | layer 1 | logits |
|---|---|---|---|---|
| hindsight, beta=0.1 | 1000 | +0.142 ± 0.003 | +0.223 ± 0.009 | +0.701 ± 0.009 |
| hindsight, beta=0.1 | 3000 | +0.052 ± 0.002 | +0.159 ± 0.009 | +0.550 ± 0.013 |
| hindsight, beta=0.5 | 3000 | +0.031 ± 0.005 | +0.058 ± 0.007 | +0.461 ± 0.005 |
| FTP-style sequential | 3000 | +0.080 ± 0.002 | +0.126 ± 0.002 | +0.645 ± 0.015 |
| no readout loss | 3000 | +0.001 ± 0.008 | +0.001 ± 0.008 | -0.010 ± 0.038 |
| random feedback (DFA) | 3000 | +0.003 ± 0.011 | -0.005 ± 0.019 | +0.028 ± 0.111 |

**Gate: PASS** (mean cosine > 0 at every layer at every checkpoint after step 0; hidden-layer alignment is weak).

## E1: imitation / editor-only training (MNIST MLP, 6,000 steps, bs 128, 3 seeds, G4)

Baselines ([runs/e1base_table.md](runs/e1base_table.md)) and editor runs ([runs/e1ed_table.md](runs/e1ed_table.md)):

| method | test acc | val loss | GFLOPs/run | notes |
|---|---|---|---|---|
| Adam backprop (lr 1e-3, cosine) | **98.41 ± 0.02%** | 0.078 ± 0.001 | 1258 | 3e-3 ties on acc; 3e-4 best val loss 0.077 |
| DFA raw rule (lr 3e-3) | 96.93 ± 0.11% | 0.128 ± 0.016 | 826 | forward-only baseline |
| FTP-style sequential, raw rule (lr 1e-4) | 73.56 ± 0.79% | 0.931 ± 0.012 | 1270 | higher lr diverges |
| parallel hindsight, raw rule (lr 3e-5) | 15.41 ± 3.94% | 3.42 ± 0.52 | 1270 | diverges at every lr |
| **Editor (untrained), lin.-corrected hindsight, alpha=0** | **87.90 ± 0.43%** | 0.430 ± 0.004 | 1019 | best Hindsight Editor |
| Editor, alpha=0.5 | 85.73 ± 0.72% | 0.504 ± 0.011 | 1019 | |
| Editor, alpha=1 (full propagation) | 35.7 ± 34.4% | diverged in 2/3 | 1011 | |
| Editor, rank 1 / 4 / 16 (alpha=0.5) | 37.5 / 77.3 / 87.2% | | 957 / 984 / 1089 | rank matters a lot |
| Editor, no readout loss | 18.8 ± 5.0% | diverges | 1019 | channel training is essential |
| Editor, no per-example loss input | 13.1 ± 6.0% | diverges | 1019 | error weighting is essential |
| Editor, without a_l features | 85.73 ± 0.72% | | 971 | identical to with: moot for an untrained editor |
| Editor, plain parallel hindsight (no correction) | 46.7 ± 22.0% | diverges | 1003 | |
| Editor, FTP-style sequential delta | 88.17 ± 0.51% | 0.408 ± 0.008 | 995 | |
| Editor, random-feedback delta | 88.14 ± 0.30% | 0.403 ± 0.014 | 550 | |
| Editor + Phase 0 imitation (gradient direction) | 69.5 ± 13.7% | 1.34 ± 0.82 | 1019 | imitation hurts |

**Gate: FAIL** (best editor 87.9% vs backprop 98.4%: a 10.5-point gap; also 9 points below raw DFA).

Diagnosis and fixes (all logged in the session; details in the write-up):
1. The raw parallel delta never shrinks when the net is right (it encodes y, not y - p): fixed by
   *linear correction* (regress d on the injected code per batch, subtract the part the blind pass
   already predicts: d~ = d - (p - 1/K) M ~ (y - p) M), plus per-example loss weighting.
2. A random/learned map at the logits rotated the error: the output-layer channel is now the identity.
3. Random-sign rank-r sketches drown the signal in cross-example noise: replaced by a randomized
   range finder (project the exact weighted update onto its r dominant directions).
4. Propagated deltas create a Hebbian feedback loop (a layer trained toward its own amplified
   response): damped with alpha; alpha=0 is stable, alpha=1 diverges.
5. Phase 0 imitation of Adam (or of the gradient direction) is ill-posed for a rank-r editor and
   makes it worse; Phase 1 meta-training is the remaining lever (E2).

## E2: Phase 1 meta-training ("cut the cord")

The editor is trained only on held-out loss after K of its own edits (meta-gradient through the
unroll); the main network never receives a gradient (asserted in code and in `tests/`). Each
meta-trained editor is then **frozen** and used alone to train fresh networks.

### MLP on MNIST (G4; per-variant tables: [runs/e2mlp_table.md](runs/e2mlp_table.md))

| editor | test acc (frozen eval, 2 meta seeds x 3 nets) | val loss | notes |
|---|---|---|---|
| untrained (E1 reference) | 87.90 ± 0.43% (3 nets) | 0.430 | |
| round 1, best of 6 variants: alpha=0, K=8 unroll, warm-started trajectories | 90.03 ± 1.30% | 0.369 | other 5 variants unstable (>=1 eval net diverged) |
| round 1 best, re-evaluated on Adam's initial nets | 90.97 ± 0.14% / 88.94 ± 0.15% (meta seed 0 / 1) | 0.322 / 0.383 | |
| **round 2: K=16, rank 16, meta lr 1e-4, 6000 meta steps** | **94.84 ± 0.44%** | 0.194 ± 0.013 | gap to backprop: 3.6 points |
| Adam backprop (reference) | 98.41 ± 0.02% | 0.078 | |

### Central measurement, MLP (shared initial networks; [runs/e2mlp_central.md](runs/e2mlp_central.md))

| method | loss after 500 updates | loss drop / update, first 500 | loss at 0.21 TFLOP | loss at 0.42 TFLOP | GFLOP / update |
|---|---|---|---|---|---|
| Adam (lr 1e-3) | **0.127 ± 0.003** | **0.00549 ± 0.00047** | **0.096 ± 0.005** | **0.088 ± 0.007** | 0.21 |
| DFA raw rule | 0.748 ± 0.373 | 0.00424 ± 0.00108 | 0.269 ± 0.048 | 0.170 ± 0.033 | 0.14 |
| Editor, untrained | 1.264 ± 0.045 | 0.00321 ± 0.00039 | 0.715 ± 0.034 | 0.546 ± 0.006 | 0.17 |
| Editor, meta round 1 | 0.486 ± 0.018 | 0.00477 ± 0.00044 | 0.383 ± 0.010 | 0.342 ± 0.007 | 0.17 |
| Editor, meta round 2 | 0.379 ± 0.027 | 0.00498 ± 0.00044 | 0.295 ± 0.022 | 0.243 ± 0.019 | 0.18 |

Averaged over the first 500 updates, the meta-trained editor's held-out loss drop per update is 91% of
Adam's, but that average is dominated by the large initial drop both methods make and hides the gap.
Measured finely (evaluation every 25 updates, same initial networks, [runs/early_central.md](runs/early_central.md)):

| method | loss after 25 / 50 / 100 / 200 / 600 updates | drop per update, first 25 | loss at 0.0105 / 0.021 / 0.042 / 0.084 TFLOP |
|---|---|---|---|
| Adam (lr 1e-3, constant) | **0.410 / 0.322 / 0.230 / 0.182 / 0.122** | **0.098 ± 0.010** | **0.322 / 0.230 / 0.182 / 0.153** |
| Editor, meta round 2 (hindsight-lc) | 1.851 / 1.136 / 0.684 / 0.491 / 0.336 | 0.041 ± 0.008 | 1.040 / 0.636 / 0.465 / 0.364 |
| Editor, meta, FTP-seq delta | 1.967 / 1.143 / 0.672 / 0.489 / 0.334 | 0.036 ± 0.008 | 1.030 / 0.621 / 0.462 / 0.365 |

Early on, the editor's loss drop per update is ~40% of Adam's, and Adam has the lower held-out loss
at every matched FLOP budget measured (from 0.01 TFLOP up). An apparent early crossing in the
coarse plot was an interpolation artifact of Adam's 500-step evaluation spacing.
**MLP gate (match Adam at matched FLOPs): FAIL.**
Wall-clock per step is not comparable across methods here (many jobs share one GPU), so it is not
used for conclusions.

### CNN on CIFAR-10 (8,000 steps, bs 128, 3 seeds; [runs/e2cnn_table.md](runs/e2cnn_table.md))

| method | test acc | val loss | notes |
|---|---|---|---|
| Adam lr 1e-3 | **74.01 ± 0.23%** | 1.579 ± 0.051 | overfits (no augmentation); lr 3e-4 has best val loss 1.010 at 71.3% |
| DFA raw rule, lr 1e-3 / 1e-4 / 3e-5 | 10.0% / 14.7 ± 4.2% / 28.2 ± 9.8% | diverged at every lr | conv DFA (no tuning beyond this sweep) |
| Editor untrained | 16.5 ± 2.5% | diverged | |
| Editor meta-trained, round-1 recipe (K=8, rank 8, 2000 meta steps) | **43.96 ± 1.69%** | 1.605 ± 0.036 | best backprop-free CNN result |
| Editor meta-trained, round-2 recipe (K=16, rank 16, 4000 meta steps) | 32.87 ± 6.89% | 1.907 ± 0.318 | the MLP recipe does not transfer |

CNN central measurement ([runs/e2cnn_central.md](runs/e2cnn_central.md)): at matched FLOPs (16.7 TFLOP) held-out
loss is 1.76 ± 0.01 (editor, round 1) vs 0.88 ± 0.03 (Adam lr 1e-3). **CNN gate: FAIL.**

### MLP ablations on the meta-trained editor (round-2 recipe, 1 meta seed x 3 nets; [runs/e2c_meta_table.md](runs/e2c_meta_table.md), [runs/e2mlp_central_ablations.md](runs/e2mlp_central_ablations.md))

| delta / editor input | test acc | loss at 0.21 TFLOP | loss at 0.42 TFLOP | GFLOP / update |
|---|---|---|---|---|
| parallel hindsight + linear correction (main method; 2 meta seeds) | 94.84 ± 0.44% | 0.295 ± 0.022 | 0.243 ± 0.019 | 0.182 |
| FTP-style **sequential** sighted pass (hint = y - p) | **96.00 ± 0.08%** | **0.268 ± 0.004** | **0.208 ± 0.003** | 0.179 |
| editor without a_l features | 95.14 ± 0.07% | 0.275 ± 0.007 | 0.224 ± 0.005 | 0.174 |
| random-feedback delta (DFA signal, same learned editor) | 93.87 ± 0.13% | 0.288 ± 0.002 | 0.237 ± 0.004 | 0.103 |
| propagation alpha = 0.5 | 16.8 ± 2.4% | diverged | diverged | 0.182 |
| Adam (reference) | 98.41 ± 0.02% | 0.096 ± 0.005 | 0.088 ± 0.007 | 0.21 |

Reading: once the editor is meta-trained, the hindsight signal beats random feedback by ~1 point
of accuracy (but random feedback is ~45% cheaper per update, so at matched FLOPs they are close);
the sequential (FTP-style) sighted pass beats the parallel one; the a_l features do not help; and
propagating hindsight through layers (alpha > 0) is unstable even for a meta-trained editor.

**E2 gate (match or beat Adam at matched FLOPs on loss drop per update): FAIL** at both scales.
Best gap: MLP held-out loss 0.268 vs 0.096 at 0.21 TFLOP (2.8x); CNN 1.76 vs 0.88 (2x).

## E3: attention (character-level transformer on enwik8)

Model: pre-LN decoder, d=384, 6 layers, 6 heads, context 256, 10.9M parameters, bs 32, byte vocabulary.
Future-token hindsight: discounted summary of the next k=8 tokens through a width-w bottleneck
(optionally VIB-penalised), injected into the residual stream before every block; the head gets the
exact next-token error; deltas linearly corrected (LM version). Table: [runs/e3a_table.md](runs/e3a_table.md).

### Wave A: Adam baseline and the untrained editor (bottleneck ablation)

| method | steps reached | val loss (nats) | bpc | finished | n |
|---|---|---|---|---|---|
| adam_lr1e-3 | 10000 | 1.064 ± 0.003 | 1.535 | yes | 3 |
| adam_lr3e-3 | 10000 | 2.156 ± 0.096 | 3.110 | yes | 3 |
| adam_lr3e-4 | 10000 | 1.117 ± 0.002 | 1.611 | yes | 3 |
| ed_w16_novib | 6500 | 3.511 ± 0.000 | 5.065 | no (stopped) | 2 |
| ed_w16_vib | 6500/7000 | 4.953 ± 1.290 | 7.145 | no (stopped) | 3 |
| ed_w4_novib | 6000 | 3.505 ± 0.003 | 5.056 | no (stopped) | 3 |
| ed_w4_vib | 6500 | 3.515 ± 0.005 | 5.071 | no (stopped) | 3 |
| ed_w64_vib | 6500 | 3.533 ± 0.026 | 5.097 | no (stopped) | 3 |

Unigram reference (val): 3.511 nats = 5.065 bpc.

Adam reaches 1.535 bpc after 10k steps. The **untrained** editor learns nothing beyond the
unigram distribution under any bottleneck (width 4/16/64, with or without VIB): every variant sits
at 3.51 nats (some VIB runs diverge above it). The wave was stopped early (after 1.6 h) once this was
clear, to spend the budget on a meta-trained LM editor instead; "steps reached" is where each run stopped.

**Caveat (bug found afterwards):** in these runs the GPT sighted pass cut gradients *after* each
block's injection instead of before, so only the first of the 7 hindsight channels was trained by
its readout; the other 6 injected fixed random codes. The editor rows above therefore test the
untrained editor with a mostly untrained hindsight channel. The bug is fixed (with a unit test that
every channel receives readout gradient) and E3 results after this point use the fixed code. The
MLP/CNN rungs were not affected (their detach comes after the injected state is recorded).


### Waves B-D: meta-trained editor, step-size fix, bottleneck ablation (fixed channel code)

Wave B meta-trained the editor on small (d=128, 2 layers), and full-size GPTs and applied it frozen to
the 10.9M model: all stayed at the unigram level (3.508-3.556 nats at 6k steps vs Adam 1.133).
Diagnosis on a synthetic bigram language (generated in code, [analysis/diag/lm_markov.py](analysis/diag/lm_markov.py)):
with a tiny GPT every rule learns far below unigram if the step size is right (unigram 3.82, bigram
optimum ~0.90: Adam 0.92, raw hindsight rule 1.15, untrained editor 1.06 at lr_init 1e-2 vs 3.28 at
1e-3). The editor's default step size (tuned on the MLP) was ~10x too small for transformers.
Waves C/D reran E3 with the larger step size ([runs/e3c_table.md](runs/e3c_table.md), [runs/e3d_table.md](runs/e3d_table.md)):

| method (10.9M GPT, enwik8, 6,000 steps, 3 seeds) | val loss (nats) | bpc | vs unigram |
|---|---|---|---|
| Adam, lr 1e-3 (6k-step schedule) | **1.114 ± 0.003** | **1.607** | -2.397 |
| Editor, untrained, lr_init 1e-2 (w=16, no VIB) | 3.528 ± 0.013 | 5.090 | +0.016 |
| Editor, meta-trained on a small GPT (lr_init 1e-2 start) | 3.533 ± 0.026 | 5.097 | +0.021 |
| Editor, untrained, lr_init 3e-2 | 84.1 ± 29.9 | diverged | |
| Raw hindsight rule (linearly corrected), lr 1e-3 | diverged (3/3) | | |

Bottleneck ablation (untrained editor, lr_init 1e-2, 3 seeds each):

| bottleneck width | no info penalty | with VIB (KL 0.01) |
|---|---|---|
| 4 | 3.581 ± 0.079 | 7.079 ± 5.029 (unstable) |
| 16 | 3.643 ± 0.175 | 3.640 ± 0.159 |
| 64 | 3.513 ± 0.009 | 3.599 ± 0.122 |

Unigram (val) = 3.511 nats = 5.066 bpc.

**E3 gate (learns at all, val loss well below unigram): FAIL.** No Hindsight Editor variant gets
meaningfully below the unigram loss on enwik8; the gap to backprop is the full 2.4 nats Adam gains
over unigram. The bottleneck width/VIB ablation shows no setting that learns (narrow + VIB is
additionally unstable), so it cannot rank bottlenecks by quality.


## E5: self-editing (Phase 2) on the best E2 setup (MNIST MLP)

The editor edits its own control weights (step-size head, momentum head, per-example weighting
head) with no gradient: each step it perturbs them by a random +-eps direction, applies its edit to
the network, and uses the realised held-out improvement r = log L_before - log L_after (two extra
forward passes on a held-out batch) as the hindsight signal, moving its weights by
eta * (r - baseline) / eps along the tried direction. eps = 0.01. Table: [runs/e5_table.md](runs/e5_table.md).

| editor (6,000 steps, Adam's 3 initial nets) | test acc | val loss | GFLOPs/run |
|---|---|---|---|
| meta-trained (E2 round 2, meta seed 0), **frozen** | **95.25 ± 0.12%** | 0.182 ± 0.002 | 1089 |
| meta-trained, self-editing, eta = 0.01 | 76.1 ± 23.9% | 81 ± 76 (unstable) | 1915 |
| meta-trained, self-editing, eta = 0.05 | 10.5 ± 0.6% | diverged | 1915 |
| untrained, frozen | 91.75 ± 0.18% | 0.317 ± 0.005 | 1089 |
| untrained, self-editing, eta = 0.01 | 89.15 ± 1.62% | 0.964 ± 0.823 | 1915 |

**E5 result: not stable, and the editor does not improve itself.** The per-step reward from one
held-out batch is dominated by noise, so the self-edited step size random-walks: upward in most runs
(1e-3 -> 2-5e-3, or to 4e-2 at eta = 0.05, causing divergence) or collapsing (to 5e-5 with momentum
0.998, freezing learning). Self-editing also costs 76% more FLOPs per run (two extra forward passes per step).
