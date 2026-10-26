"""Aggregate E0 cosine(d_l, -backprop error) over seeds: mean +- std per variant/step/layer."""
import glob, json, sys
from collections import defaultdict
import numpy as np

rows = defaultdict(list)
acc = defaultdict(list)
for f in glob.glob(sys.argv[1] if len(sys.argv) > 1 else "runs/e0_mnist_s*/metrics.jsonl"):
    for line in open(f):
        r = json.loads(line)
        if r.get("event") == "cos":
            rows[(r["variant"], r["step"])].append([r[f"cos_l{i}"] for i in range(3)])
        if r.get("event") == "eval":
            acc[r["step"]].append(r["val_acc"])
print("| variant | step | cos l0 | cos l1 | cos l2 (logits) | n |")
print("|---|---|---|---|---|---|")
for (v, s), xs in sorted(rows.items(), key=lambda k: (k[0][0], k[0][1])):
    a = np.array(xs)
    cells = " | ".join(f"{m:+.3f} ± {d:.3f}" for m, d in zip(a.mean(0), a.std(0)))
    print(f"| {v} | {s} | {cells} | {len(xs)} |")
print("\nbackprop-trained net val acc by step:", {s: round(float(np.mean(v)), 4) for s, v in sorted(acc.items())})
