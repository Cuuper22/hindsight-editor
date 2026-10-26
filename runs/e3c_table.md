| method | val loss (nats) | bpc | below unigram (nats) | GFLOPs/run | steps | n |
|---|---|---|---|---|---|---|
| adam_lr1e-3 | 1.114 ± 0.003 | 1.607 | +2.397 | 3508700 | 6000 | 3 |
| ed_lr1e-2 | 3.528 ± 0.013 | 5.090 | -0.016 | 2720337 | 6000 | 3 |
| ed_lr3e-2 | 84.135 ± 29.948 | 121.381 | -80.623 | 2720337 | 6000 | 3 |
| meta_small (frozen meta editor) | 3.533 ± 0.026 | 5.097 | -0.021 | 2720337 | 6000 | 3 |
| raw_lr1e-3 | diverged | | | | 6000 | 3 |

Unigram (val): 3.511 nats = 5.066 bpc.
