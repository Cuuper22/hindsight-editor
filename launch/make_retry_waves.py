"""Retry waves (round 3): Adam on hindsight pseudo-gradients (HAdam), controls, propagation/EMA, editor rank.
python launch/make_retry_waves.py  -> experiments/waves/r_*.json"""
import json

def wave(name, gpu, par, hours, jobs, prefetch):
    json.dump({"wave": name, "gpu": gpu, "max_parallel": par, "max_hours": hours, "prefetch": prefetch, "jobs": jobs},
              open(f"experiments/waves/{name}.json", "w"), indent=1)
    print(name, gpu, len(jobs), "jobs")

def job(rid, config, ov):
    return {"run_id": rid, "config": config, "overrides": {**ov, "run_id": rid}}

def tag(lr):
    return f"{lr:.0e}".replace("e-0", "e-")

import sys
ONLY_LM = "--lm" in sys.argv
# ---------------- MNIST MLP: signal quality under one optimizer -------------------------
SRC = {
    "bp": {"delta_source": "backprop"},
    "dfa": {"delta_source": "dfa"},
    "frozen": {"delta_source": "dfa", "freeze_body": True},
    "hlc_p0": {"delta_source": "hindsight_lc", "prop": 0.0},
    "hlc_p05": {"delta_source": "hindsight_lc", "prop": 0.5},
    "hlc_p1": {"delta_source": "hindsight_lc", "prop": 1.0},
    "hlc_p1_ema99": {"delta_source": "hindsight_lc", "prop": 1.0, "ema": 0.99},
    "hlc_p1_ema995": {"delta_source": "hindsight_lc", "prop": 1.0, "ema": 0.995},
    "hlc_p05_ema99": {"delta_source": "hindsight_lc", "prop": 0.5, "ema": 0.99},
    "hlc_p1_copy100": {"delta_source": "hindsight_lc", "prop": 1.0, "copy_every": 100},
    "ftp": {"delta_source": "ftp_seq", "prop": 1.0},
}
base = {"method": "hadam", "gate": True, "beta": 1.0, "betas": [0.9, 0.999], "wd": 0.0}
jobs = [job(f"r_mnist_{k}_lr{tag(lr)}_s{s}", "experiments/e1_mnist.json", {**base, **v, "lr": lr, "seed": s})
        for k, v in SRC.items() for lr in (3e-4, 1e-3, 3e-3) for s in (0, 1, 2)]
# learned editor, untrained: full rank vs rank 16 (with the ridge fix)
for k, ed in {"ed_full_p0": {"rank": 0}, "ed_r16_p0": {"rank": 16}}.items():
    for s in (0, 1, 2):
        jobs.append(job(f"r_mnist_{k}_s{s}", "experiments/e1_mnist.json",
                        {"method": "editor", "delta_source": "hindsight_lc", "gate": True, "beta": 1.0, "prop": 0.0,
                         "editor": ed, "seed": s}))
for s in (0, 1, 2):
    jobs.append(job(f"r_mnist_ed_full_dfa_s{s}", "experiments/e1_mnist.json",
                    {"method": "editor", "delta_source": "dfa", "gate": True, "beta": 1.0, "editor": {"rank": 0}, "seed": s}))
if not ONLY_LM: wave("r_mnist", "G4", 24, 2.0, jobs, ["mnist"])

# ---------------- meta-training: full-rank editor, round-2 recipe ------------------------
meta = []
for k, ed in {"full": {"rank": 0, "lr_init": 1e-3}, "r16": {"rank": 16, "lr_init": 1e-3}}.items():
    for s in (0, 1):
        meta.append(job(f"r_meta_{k}_s{s}", "experiments/e2_meta_mnist.json",
                        {"prop": 0.0, "unroll": 16, "inner_steps": 2000, "adam_warm_max": 1500, "meta_lr": 1e-4,
                         "meta_steps": 6000, "editor": ed, "seed": s,
                         "eval": {"seeds": [0, 1, 2], "overrides": {"task": "mnist", "net": "mlp", "steps": 6000,
                                  "eval_every": 500, "log_every": 100, "ckpt_every": 2000}}}))
if not ONLY_LM: wave("r_meta", "A100", 4, 3.0, meta, ["mnist"])

# ---------------- CIFAR-10 CNN ---------------------------------------------------------
CS = {k: SRC[k] for k in ("bp", "dfa", "frozen", "hlc_p0")}
cj = [job(f"r_cifar_{k}_lr{tag(lr)}_s{s}", "experiments/e2_cnn.json", {**base, **v, "lr": lr, "seed": s})
      for k, v in CS.items() for lr in (3e-4, 1e-3) for s in (0, 1)]
if not ONLY_LM: wave("r_cifar", "G4", 16, 2.0, cj, ["cifar10"])

# ---------------- enwik8 char GPT (10.9M) -----------------------------------------------
lb = {"method": "hadam", "vib": False, "betas": [0.9, 0.999], "wd": 0.0, "steps": 6000, "warmup": 200}
LS = {
    "dfa": ({"delta_source": "dfa"}, (1e-4, 3e-4, 1e-3)),
    "frozen": ({"delta_source": "dfa", "freeze_body": True}, (1e-3,)),
    "hlc_a0": ({"delta_source": "hindsight_lc", "prop": 0.0}, (1e-4, 3e-4, 1e-3)),
    "hlc_a0_ema": ({"delta_source": "hindsight_lc", "prop": 0.0, "ema": 0.99}, (3e-4, 1e-3)),
    "hlc_a1": ({"delta_source": "hindsight_lc", "prop": 1.0}, (3e-4,)),
    "hlc_a0_body01": ({"delta_source": "hindsight_lc", "prop": 0.0, "body_lr_mult": 0.1}, (1e-3,)),
    "dfa_body01": ({"delta_source": "dfa", "body_lr_mult": 0.1}, (1e-3,)),
}
lj = [job(f"r_lm_{k}_lr{tag(lr)}_s{s}", "experiments/e3_enwik8.json", {**lb, **v, "lr": lr, "seed": s})
      for k, (v, lrs) in LS.items() for lr in lrs for s in (0, 1)]
lj.append(job("r_lm_bp_lr1e-3_s0", "experiments/e3_enwik8.json", {**lb, "delta_source": "backprop", "lr": 1e-3, "seed": 0}))
for k, ed in {"ed_full": {"rank": 0}, "ed_r64": {"rank": 64}, "ed_r16": {"rank": 16}}.items():
    for s in ((0, 1) if k == "ed_full" else (0,)):
        lj.append(job(f"r_lm_{k}_a0_s{s}", "experiments/e3_enwik8.json",
                      {"method": "editor", "vib": False, "delta_source": "hindsight_lc", "prop": 0.0, "steps": 6000,
                       "editor": {**ed, "lr_init": 1e-2, "max_rel": 0.05, "use_id": False}, "seed": s}))
n = 5
for i in range(n):
    wave(f"r_lm{i}", "A100", 8, 3.0, lj[i::n], ["enwik8"])
