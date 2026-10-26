"""Results table for a wave: python analysis/table.py WAVE [--metric test_acc|val_loss]

Groups runs named WAVE_<label>_s<seed>, reports mean +- std over seeds of the final
metrics, FLOPs, wall-clock and peak memory, and marks the best LR per method family
(label prefix before "_lr"). Never fills in missing runs: n shows how many seeds finished.
"""
import glob, json, os, re, sys
from collections import defaultdict
import numpy as np

wave = sys.argv[1]
metric = sys.argv[sys.argv.index("--metric") + 1] if "--metric" in sys.argv else "test_acc"
higher = metric.endswith("acc")
groups = defaultdict(list)
for f in glob.glob(f"runs/{wave}_*_s*/summary.json"):
    label = re.match(rf"{re.escape(wave)}_(.+)_s\d+$", os.path.basename(os.path.dirname(f))).group(1)
    r = json.load(open(f))
    if "flops" in r:                                               # skip meta-run summaries
        groups[label].append(r)


def ms(xs, fmt="{:.4f}"):
    xs = [x for x in xs if x is not None and np.isfinite(x)]
    if not xs:
        return "diverged", None
    return (fmt + " ± " + fmt).format(np.mean(xs), np.std(xs)), float(np.mean(xs))


rows, best = [], {}
for label, rs in sorted(groups.items()):
    fam = label.split("_lr")[0]
    cell, m = ms([r.get(metric) for r in rs])
    rows.append((fam, label, rs, cell, m))
    if m is not None and (fam not in best or (m > best[fam][1] if higher else m < best[fam][1])):
        best[fam] = (label, m)

print(f"| method | {metric} | val loss | GFLOPs/run | wall s | peak GB | n |")
print("|---|---|---|---|---|---|---|")
for fam, label, rs, cell, m in rows:
    star = " **(best lr)**" if best.get(fam, (None,))[0] == label else ""
    vl, _ = ms([r.get("val_loss") for r in rs])
    print(f"| {label}{star} | {cell} | {vl} | {np.mean([r['flops'] for r in rs]) / 1e9:.1f} | "
          f"{np.mean([r['wall'] for r in rs]):.1f} | {np.mean([r['peak_mem_gb'] for r in rs]):.2f} | {len(rs)} |")
