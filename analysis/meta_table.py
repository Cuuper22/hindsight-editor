"""Meta-trained editors: frozen-eval results per variant (meta seeds x eval seeds), plus meta-training trace.

python analysis/meta_table.py WAVE   -> markdown table; runs/WAVE_<label>_s<metaseed>/eval_s<seed>/summary.json
"""
import glob, json, os, re, sys
from collections import defaultdict
import numpy as np

wave = sys.argv[1]
acc, loss, flops, wall, trace = defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list), defaultdict(list)
for f in glob.glob(f"runs/{wave}_*_s*/eval_s*/summary.json"):
    run = f.replace("\\", "/").split("/")[1]
    label = re.match(rf"{re.escape(wave)}_(.+)_s\d+$", run).group(1)
    r = json.load(open(f))
    acc[label].append(r.get("test_acc", np.nan)); loss[label].append(r["val_loss"])
    flops[label].append(r["flops"]); wall[label].append(r["wall"])
for f in glob.glob(f"runs/{wave}_*_s*/metrics.jsonl"):
    run = f.replace("\\", "/").split("/")[1]
    label = re.match(rf"{re.escape(wave)}_(.+)_s\d+$", run).group(1)
    ms = [json.loads(l) for l in open(f) if '"event": "meta"' in l]
    if ms:
        tail = ms[-max(1, len(ms) // 5):]                          # last 20% of meta-training
        trace[label].append((np.mean([m["pre"] for m in tail]), np.mean([m["post"] for m in tail]),
                             tail[-1]["skipped"], max(m["traj"] for m in ms)))
fmt = lambda xs, k=4: f"{np.nanmean(xs):.{k}f} ± {np.nanstd(xs):.{k}f}" if np.isfinite(np.nanmean(xs)) else "diverged"
print("| variant | test acc (frozen eval) | val loss | GFLOPs/run | n | meta: pre -> post loss (last 20%) | trajectories | skipped |")
print("|---|---|---|---|---|---|---|---|")
for label in sorted(acc):
    t = trace.get(label, [])
    tr = " / ".join(f"{a:.3f}->{b:.3f}" for a, b, _, _ in t)
    print(f"| {label} | {fmt(acc[label])} | {fmt(loss[label], 3)} | {np.mean(flops[label]) / 1e9:.0f} | {len(acc[label])} | {tr} | "
          f"{'/'.join(str(x[3]) for x in t)} | {'/'.join(str(x[2]) for x in t)} |")
