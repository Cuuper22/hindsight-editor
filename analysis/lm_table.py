"""LM results for a wave: val loss (nats), bits per character/token, FLOPs, gap to unigram.

python analysis/lm_table.py WAVE [--bpc]   (runs WAVE_<label>_s<seed>/summary.json, and meta eval sub-runs)
"""
import glob, json, os, re, sys
from collections import defaultdict
import numpy as np

wave, bpc = sys.argv[1], "--bpc" in sys.argv
G = defaultdict(list)
for f in glob.glob(f"runs/{wave}_*/summary.json") + glob.glob(f"runs/{wave}_*/eval_s*/summary.json"):
    parts = f.replace("\\", "/").split("/")
    label = re.match(rf"{re.escape(wave)}_(.+)_s\d+$", parts[1]).group(1) + (" (frozen meta editor)" if "eval_s" in f else "")
    r = json.load(open(f))
    if "flops" in r:
        G[label].append(r)
uni = np.mean([r["unigram_nats"] for rs in G.values() for r in rs if "unigram_nats" in r])
unit = "bpc" if bpc else "bits/token"
print(f"| method | val loss (nats) | {unit} | below unigram (nats) | GFLOPs/run | steps | n |")
print("|---|---|---|---|---|---|---|")
for lab in sorted(G):
    v = np.array([r["val_loss"] for r in G[lab]], float)
    fin = v[np.isfinite(v)]
    cell = f"{fin.mean():.3f} ± {fin.std():.3f}" if len(fin) else "diverged"
    extra = "" if len(fin) == len(v) else f" ({len(v) - len(fin)} diverged)"
    print(f"| {lab} | {cell}{extra} | {fin.mean() / np.log(2):.3f} | {uni - fin.mean():+.3f} | "
          f"{np.mean([r['flops'] for r in G[lab]]) / 1e9:.0f} | {G[lab][0]['steps']} | {len(v)} |" if len(fin) else
          f"| {lab} | diverged | | | | {G[lab][0]['steps']} | {len(v)} |")
print(f"\nUnigram (val): {uni:.3f} nats = {uni / np.log(2):.3f} {unit}.")
