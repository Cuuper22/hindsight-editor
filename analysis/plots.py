"""Held-out loss vs FLOPs and vs wall-clock for every rung (handoff: plot against FLOPs and
wall-clock, not steps). Mean over seeds, shaded = min..max. Writes analysis/plots/*.png.

python analysis/plots.py
"""
import glob, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(__file__), "plots")
os.makedirs(OUT, exist_ok=True)
COLORS = ["#2a6fdb", "#e0612b", "#2e9e5b", "#9b4fd1", "#c9a227", "#d6337a", "#5b6770"]


def load(pattern):
    runs = []
    for f in sorted(glob.glob(pattern)):
        ev = [json.loads(l) for l in open(f) if '"event": "eval"' in l]
        ev = [e for e in ev if e.get("flops") is not None and e.get("val_loss") is not None]
        if len(ev) > 1:
            runs.append({k: np.array([e.get(k) or 0.0 for e in ev], float) for k in ("flops", "wall", "val_loss", "step")})
    return runs


def band(ax, runs, key, label, color, scale):
    if not runs:
        return
    n = min(len(r[key]) for r in runs)
    x = np.mean([r[key][:n] for r in runs], 0) / scale
    y = np.array([r["val_loss"][:n] for r in runs])
    y = np.where(np.isfinite(y), y, np.nan)
    ax.plot(x, np.nanmean(y, 0), color=color, label=f"{label} (n={len(runs)})", lw=1.8)
    if len(runs) > 1:
        ax.fill_between(x, np.nanmin(y, 0), np.nanmax(y, 0), color=color, alpha=0.15, lw=0)


def figure(name, title, series, ylim=None, hline=None):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, key, scale, xl in ((axes[0], "flops", 1e12, "TFLOPs (fwd + bwd/editor + optimizer)"),
                               (axes[1], "wall", 60.0, "wall-clock (min; jobs share a GPU)")):
        for (label, pattern), c in zip(series, COLORS):
            band(ax, load(pattern), key, label, c, scale)
        if hline:
            ax.axhline(hline[1], color="k", ls=":", lw=1, label=hline[0])
        ax.set_xlabel(xl); ax.set_ylabel("held-out loss (nats)")
        if ylim:
            ax.set_ylim(*ylim)
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, f"{name}.png"), dpi=130)
    plt.close(fig)
    print("wrote", name)


figure("e2_mlp", "E1/E2: MNIST MLP", [
    ("Adam (backprop)", "runs/e1base_adam_lr1e-3_s*/metrics.jsonl"),
    ("DFA raw rule", "runs/e1base_dfa_lr3e-3_s*/metrics.jsonl"),
    ("Editor untrained", "runs/e1ed_lc_p0_s*/metrics.jsonl"),
    ("Editor meta round 2", "runs/e2mlp_r2_s*/eval_s*/metrics.jsonl"),
    ("Editor meta, FTP-seq delta", "runs/e2c_mlp_ftpseq_s0/eval_s*/metrics.jsonl"),
    ("Editor meta, random feedback", "runs/e2c_mlp_randfb_s0/eval_s*/metrics.jsonl"),
], ylim=(0, 1.5))
figure("e2_cnn", "E2: CIFAR-10 CNN", [
    ("Adam lr 1e-3", "runs/e2cnn_adam_lr1e-3_s*/metrics.jsonl"),
    ("Adam lr 3e-4", "runs/e2cnn_adam_lr3e-4_s*/metrics.jsonl"),
    ("Editor meta round 1", "runs/e2cnn_meta_s0/eval_s*/metrics.jsonl"),
    ("Editor meta round 2", "runs/e2c_cnn_r2_s0/eval_s*/metrics.jsonl"),
], ylim=(0.5, 2.6))
figure("e3_enwik8", "E3: char transformer on enwik8", [
    ("Adam lr 1e-3", "runs/e3c_adam_lr1e-3_s*/metrics.jsonl"),
    ("Editor lr_init 1e-2", "runs/e3c_ed_lr1e-2_s*/metrics.jsonl"),
    ("Editor meta (small GPT)", "runs/e3c_meta_small_s0/eval_s*/metrics.jsonl"),
    ("Editor w64 no VIB", "runs/e3d_w64_novib_s*/metrics.jsonl"),
], ylim=(0.9, 4.5), hline=("unigram", 3.511))
figure("e4_fineweb", "E4: GPT (163M) on FineWeb-Edu", [
    ("AdamW lr 6e-4", "runs/e4_adamw_lr6e-4_s*/metrics.jsonl"),
    ("AdamW lr 1.8e-3", "runs/e4_adamw_lr1.8e-3_s*/metrics.jsonl"),
    ("Editor", "runs/e4_editor_s1fix/metrics.jsonl"),
])
figure("e5_selfedit", "E5: self-editing (MNIST MLP)", [
    ("meta editor, frozen", "runs/e5_meta_frozen_s*/metrics.jsonl"),
    ("meta editor, self-edit eta 0.01", "runs/e5_meta_self_eta1e-2_s*/metrics.jsonl"),
    ("untrained, frozen", "runs/e5_untrained_frozen_s*/metrics.jsonl"),
    ("untrained, self-edit eta 0.01", "runs/e5_untrained_self_eta1e-2_s*/metrics.jsonl"),
], ylim=(0, 1.5))
