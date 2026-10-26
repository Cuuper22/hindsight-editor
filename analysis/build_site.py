"""Build the story page (index.html) from site/template.html + figure data extracted from runs/.

python analysis/build_site.py

Every number a figure draws comes from the run logs here, except the few diagnostics marked
HAND (copied from analysis/diag/* console output, recorded in RESULTS.md / the session).
"""
import glob, json, os, re
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = lambda *p: os.path.join(ROOT, *p)


def evals(f):
    ev = [json.loads(l) for l in open(f) if '"event": "eval"' in l]
    return [e for e in ev if e.get("val_loss") is not None and e.get("flops") is not None]


def curve(pattern, key="flops", scale=1e12, maxpts=60):
    """Mean held-out loss vs FLOPs (or step) over the runs matching pattern, on the shared grid."""
    runs = [evals(f) for f in sorted(glob.glob(R(pattern)))]
    runs = [r for r in runs if len(r) > 1]
    if not runs:
        return None
    n = min(len(r) for r in runs)
    xs = np.mean([[e[key] for e in r[:n]] for r in runs], 0) / scale
    ys = np.array([[e["val_loss"] for e in r[:n]] for r in runs], float)
    ys = np.where(np.isfinite(ys), ys, np.nan)
    step = max(1, n // maxpts)
    idx = list(range(0, n, step)) + ([n - 1] if (n - 1) % step else [])
    return {"x": [round(float(xs[i]), 5) for i in idx], "y": [round(float(np.nanmean(ys[:, i])), 4) for i in idx],
            "lo": [round(float(np.nanmin(ys[:, i])), 4) for i in idx], "hi": [round(float(np.nanmax(ys[:, i])), 4) for i in idx],
            "n": len(runs)}


def summ(pattern, key):
    v = [json.load(open(f)).get(key) for f in glob.glob(R(pattern))]
    v = [x for x in v if x is not None and np.isfinite(x)]
    return {"m": round(float(np.mean(v)), 4), "s": round(float(np.std(v)), 4), "n": len(v)} if v else None


D = {}
# E0: cosine(d_l, -dL/dz_l) per layer, per variant and probe step
e0 = {}
for f in glob.glob(R("runs/e0_mnist_s*/metrics.jsonl")):
    for l in open(f):
        r = json.loads(l)
        if r.get("event") == "cos":
            e0.setdefault(r["variant"], {}).setdefault(r["step"], []).append([r[f"cos_l{i}"] for i in range(3)])
D["e0"] = {v: {str(s): [round(float(x), 3) for x in np.mean(c, 0)] for s, c in sorted(st.items())} for v, st in e0.items()}

# E1/E2/E5 accuracies
acc = lambda p: summ(p + "/summary.json", "test_acc")
D["mnist_acc"] = {
    "Adam (backprop)": acc("runs/e1base_adam_lr1e-3_s*"), "DFA raw rule": acc("runs/e1base_dfa_lr3e-3_s*"),
    "FTP-style raw rule": acc("runs/e1base_ftp_lr1e-4_s*"), "Parallel hindsight, raw rule": acc("runs/e1base_hraw_lr3e-5_s*"),
    "Editor, untrained": acc("runs/e1ed_lc_p0_s*"), "Editor + imitation (Phase 0)": acc("runs/e1ed_lc_p05_imitation_s*"),
    "Editor, meta round 1": summ("runs/e2mlp_p0_warm_k8_s*/eval_s*/summary.json", "test_acc"),
    "Editor, meta round 2": summ("runs/e2mlp_r2_s*/eval_s*/summary.json", "test_acc"),
    "Editor, meta, FTP-style pass": summ("runs/e2c_mlp_ftpseq_s0/eval_s*/summary.json", "test_acc"),
}
D["ablate"] = {
    "Parallel hindsight + correction": summ("runs/e2mlp_r2_s*/eval_s*/summary.json", "test_acc"),
    "Sequential (FTP-style) pass": summ("runs/e2c_mlp_ftpseq_s0/eval_s*/summary.json", "test_acc"),
    "No a_l inputs to the editor": summ("runs/e2c_mlp_noa_s0/eval_s*/summary.json", "test_acc"),
    "Random feedback instead": summ("runs/e2c_mlp_randfb_s0/eval_s*/summary.json", "test_acc"),
    "Propagation alpha = 0.5": summ("runs/e2c_mlp_p05_s0/eval_s*/summary.json", "test_acc"),
}
D["rank"] = {k: acc(f"runs/e1ed_lc_p05{s}_s*") for k, s in (("1", "_r1"), ("4", "_r4"), ("8", ""), ("16", "_r16"))}
D["cnn_acc"] = {
    "Adam lr 1e-3": acc("runs/e2cnn_adam_lr1e-3_s*"), "Adam lr 3e-4": acc("runs/e2cnn_adam_lr3e-4_s*"),
    "Editor, meta round 1": summ("runs/e2cnn_meta_s0/eval_s*/summary.json", "test_acc"),
    "Editor, meta round 2": summ("runs/e2c_cnn_r2_s0/eval_s*/summary.json", "test_acc"),
    "Editor, untrained": acc("runs/e2cnn_edinit_s*"),
}
# central measurement curves (MNIST): early (eval every 25) and full runs
D["mlp_early"] = {"Adam": curve("runs/early_adam_s*/metrics.jsonl"),
                  "Editor (meta, parallel)": curve("runs/early_edr2_s*/metrics.jsonl"),
                  "Editor (meta, FTP-style)": curve("runs/early_edftp_s*/metrics.jsonl")}
D["mlp_full"] = {"Adam": curve("runs/e1base_adam_lr1e-3_s*/metrics.jsonl"),
                 "DFA raw rule": curve("runs/e1base_dfa_lr3e-3_s*/metrics.jsonl"),
                 "Editor (untrained)": curve("runs/e1ed_lc_p0_s*/metrics.jsonl"),
                 "Editor (meta, parallel)": curve("runs/e2mlp_r2_s*/eval_s*/metrics.jsonl"),
                 "Editor (meta, FTP-style)": curve("runs/e2c_mlp_ftpseq_s0/eval_s*/metrics.jsonl")}
D["cnn_full"] = {"Adam": curve("runs/e2cnn_adam_lr1e-3_s*/metrics.jsonl"),
                 "Editor (meta round 1)": curve("runs/e2cnn_meta_s0/eval_s*/metrics.jsonl")}
# E3
D["e3"] = {"Adam": curve("runs/e3c_adam_lr1e-3_s*/metrics.jsonl"),
           "Editor": curve("runs/e3c_ed_lr1e-2_s*/metrics.jsonl"),
           "Editor (meta-trained)": curve("runs/e3c_meta_small_s0/eval_s*/metrics.jsonl")}
D["e3_unigram"] = 3.511
grid = {}
for w in (4, 16, 64):
    for vib in ("novib", "vib"):
        grid[f"{w}_{vib}"] = summ(f"runs/e3d_w{w}_{vib}_s*/summary.json", "val_loss")
D["e3_grid"] = grid
D["e3_adam"] = summ("runs/e3c_adam_lr1e-3_s*/summary.json", "val_loss")
# HAND: synthetic bigram language bisection (analysis/diag/lm_markov.py console output)
D["bisect"] = {"unigram": 3.823, "bigram": 0.898, "adam": 0.914,
               "rows": [["tiny  d=64, 2 layers, ctx 32", 1.056], ["longer context (256)", 1.055],
                        ["deeper (6 layers)", 3.201], ["wider (d=384)", 4.204], ["full E3 size", 3.989]]}
# HAND: weight-level alignment with -grad on an Adam-warmed MNIST net (analysis/diag/edit_alignment.py)
D["align"] = [["raw parallel delta", [0.006, 0.011, -0.002]], ["+ per-example error weight", [0.098, 0.211, 0.651]],
              ["+ rank-16 range finder", [0.099, 0.210, 0.649]], ["random-sign rank-8 sketch", [0.028, 0.016, -0.116]],
              ["sequential (FTP-style) delta", [0.114, 0.240, 0.802]]]
# E4
D["e4"] = {"AdamW lr 6e-4": curve("runs/e4_adamw_lr6e-4_s0/metrics.jsonl"),
           "AdamW lr 1.8e-3": curve("runs/e4_adamw_lr1.8e-3_s0/metrics.jsonl"),
           "Editor": curve("runs/e4_editor_s1fix/metrics.jsonl")}
D["e4_unigram"] = summ("runs/e4_adamw_lr6e-4_s0/summary.json", "unigram_nats")
# E5: self-edited step size over time
lrs = {}
for f in sorted(glob.glob(R("runs/e5_*self*_s*/metrics.jsonl"))):
    name = os.path.basename(os.path.dirname(f))
    pts = [(r["step"], r["self_lr_w"]) for r in map(json.loads, open(f)) if r.get("event") == "train" and "self_lr_w" in r]
    lrs[name] = {"x": [p[0] for p in pts][::2], "y": [round(p[1], 6) for p in pts][::2]}
D["e5_lr"] = lrs
D["e5_acc"] = {"meta, frozen": acc("runs/e5_meta_frozen_s*"), "meta, self-edit eta 0.01": acc("runs/e5_meta_self_eta1e-2_s*"),
               "meta, self-edit eta 0.05": acc("runs/e5_meta_self_eta5e-2_s*"), "untrained, frozen": acc("runs/e5_untrained_frozen_s*"),
               "untrained, self-edit eta 0.01": acc("runs/e5_untrained_self_eta1e-2_s*")}

tpl = open(R("site", "template.html"), encoding="utf-8").read()
css = open(R("site", "_cuperos.css"), encoding="utf-8").read()       # shared CuperOS stylesheet (from chicken-genes)
out = tpl.replace("/*__CSS__*/", css).replace("/*__DATA__*/null", json.dumps(D, separators=(",", ":")))
open(R("index.html"), "w", encoding="utf-8", newline="\n").write(out)
print(f"index.html: {len(out) / 1024:.0f} KB; data {len(json.dumps(D)) / 1024:.0f} KB;",
      "missing:", [k for k, v in D.items() if v is None or (isinstance(v, dict) and any(x is None for x in v.values()))])
