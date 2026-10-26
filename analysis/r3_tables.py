"""Round-3 (retry) tables: python analysis/r3_tables.py [prefix ...]  -> markdown to stdout.
Groups runs/r_* by run id minus the seed; uses summary.json when present, else the last eval."""
import collections, glob, json, os, re, statistics as S, sys

prefixes = sys.argv[1:] or ["r_mnist", "r_cifar", "r_lm", "r_meta"]
for pre in prefixes:
    G = collections.defaultdict(list)
    for d in sorted(glob.glob(f"runs/{pre}_*")):
        if not os.path.isdir(d):
            continue
        rid = os.path.basename(d)
        evals = []
        mf = os.path.join(d, "metrics.jsonl")
        if os.path.exists(mf):
            for line in open(mf):
                try:
                    e = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if e.get("event") == "eval":
                    evals.append(e)
        sm = os.path.join(d, "summary.json")
        s = json.load(open(sm)) if os.path.exists(sm) else {}
        if pre == "r_meta":
            for k, r in (s.get("eval") or {}).items():
                G[rid].append({"acc": r.get("test_acc"), "loss": r.get("val_loss"), "step": r.get("steps"), "done": True})
            continue
        last = evals[-1] if evals else {}
        G[re.sub(r"_s\d+$", "", rid)].append({"acc": s.get("test_acc", last.get("val_acc")),
                                              "loss": s.get("val_loss", last.get("val_loss")),
                                              "step": last.get("step"), "done": bool(s)})
    if not G:
        continue
    print(f"\n### {pre}\n\n| run | n (done) | step | acc % | val loss |\n|---|---|---|---|---|")
    for k in sorted(G):
        v = G[k]
        acc = [x["acc"] * 100 for x in v if x["acc"] is not None]
        loss = [x["loss"] for x in v if x["loss"] is not None]
        f = lambda a: (f"{S.mean(a):.2f} ± {S.stdev(a):.2f}" if len(a) > 1 else f"{a[0]:.2f}") if a else ""
        print(f"| {k} | {len(v)} ({sum(x['done'] for x in v)}) | {max((x['step'] or 0) for x in v)} | {f(acc)} | {f(loss)} |")
