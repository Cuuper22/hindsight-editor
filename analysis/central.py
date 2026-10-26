"""Central measurement (handoff 1d): held-out loss drop per update, and at matched FLOPs.

python analysis/central.py NAME=GLOB [NAME=GLOB ...] [--ref NAME] [--steps 250,1000,6000]

Each GLOB matches the metrics.jsonl files of one method (one per seed). Methods should share
initial networks (same seeds) so that loss after N updates is a like-for-like comparison.
Reports mean +- std over runs:
  * held-out loss after N updates, and the implied drop per update (L0 - L_N) / N,
  * held-out loss each method reaches at the reference method's FLOP budgets (matched FLOPs),
  * wall-clock per step and peak memory.
"""
import glob, json, sys
import numpy as np

argv = sys.argv[1:]
ref = argv[argv.index("--ref") + 1] if "--ref" in argv else None
steps_arg = argv[argv.index("--steps") + 1] if "--steps" in argv else None
specs = [a for a in argv if "=" in a and not a.startswith("--")]
curves = {}
for a in specs:
    name, pat = a.split("=", 1)
    runs = []
    for f in sorted(glob.glob(pat)):
        ev = [json.loads(l) for l in open(f) if '"event": "eval"' in l]
        ev = [e for e in ev if e.get("flops") is not None and e.get("val_loss") is not None]
        if ev:
            runs.append({"step": np.array([e["step"] for e in ev], float),
                         "flops": np.array([e["flops"] for e in ev], float),
                         "loss": np.array([e["val_loss"] for e in ev], float),
                         "wall": ev[-1].get("wall", np.nan), "mem": max(e.get("peak_mem_gb", 0) or 0 for e in ev)})
    curves[name] = runs
ref = ref or specs[0].split("=", 1)[0]
R = curves[ref]


def at(run, key, x):
    """Held-out loss at x (steps or flops), linear interpolation; nan beyond the run."""
    xs = run[key]
    return float(np.interp(x, xs, run["loss"])) if x <= xs[-1] else np.nan


def ms(v, k=4):
    v = np.array(v, float)
    return "n/a" if np.all(~np.isfinite(v)) else f"{np.nanmean(v):.{k}f} ± {np.nanstd(v):.{k}f}"


T = int(min(r["step"][-1] for r in R))
pts = [int(x) for x in steps_arg.split(",")] if steps_arg else [p for p in (100, 250, 500, 1000, 2000, T) if p <= T]
print(f"### Held-out loss after N updates (shared initial networks)\n")
print("| method | " + " | ".join(f"N={p}" for p in pts) + " | drop/update, first " + str(pts[0]) +
      " | sec/step | peak GB | n |")
print("|---|" + "---|" * (len(pts) + 4))
for name, runs in curves.items():
    cells = [ms([at(r, "step", p) for r in runs]) for p in pts]
    drop = ms([(r["loss"][0] - at(r, "step", pts[0])) / pts[0] for r in runs], 5)
    sps = [r["wall"] / r["step"][-1] for r in runs]
    print(f"| {name} | " + " | ".join(cells) + f" | {drop} | {ms(sps, 4)} | "
          f"{np.mean([r['mem'] for r in runs]):.2f} | {len(runs)} |")

budgets = [np.mean([np.interp(p, r["step"], r["flops"]) for r in R]) for p in pts[1:]]
print(f"\n### Held-out loss at matched FLOPs (budgets = {ref} after " + ", ".join(map(str, pts[1:])) + " updates)\n")
print("| method | " + " | ".join(f"{b / 1e12:.3g} TFLOP" for b in budgets) + " | GFLOP per update |")
print("|---|" + "---|" * (len(budgets) + 1))
for name, runs in curves.items():
    cells = [ms([at(r, "flops", b) for r in runs]) for b in budgets]
    per = np.mean([r["flops"][-1] / r["step"][-1] for r in runs]) / 1e9
    print(f"| {name} | " + " | ".join(cells) + f" | {per:.3g} |")
