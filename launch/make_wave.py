"""Build a wave file from a grid: python launch/make_wave.py NAME GPU HOURS PAR CONFIG 'json grid' [seeds]

grid = {"label": {"overrides"...}, ...}; each label x seed becomes one job run_id = NAME_label_sS.
"""
import json, sys

name, gpu, hours, par, config, grid = sys.argv[1:7]
seeds = [int(s) for s in (sys.argv[7] if len(sys.argv) > 7 else "0,1,2").split(",")]
jobs = [{"run_id": f"{name}_{label}_s{s}", "config": config, "overrides": {**ov, "seed": s, "run_id": f"{name}_{label}_s{s}"}}
        for label, ov in json.loads(grid).items() for s in seeds]
prefetch = [t for t in ("mnist", "cifar10", "enwik8") if t in open(config).read()]
json.dump({"wave": name, "gpu": gpu, "max_parallel": int(par), "max_hours": float(hours), "prefetch": prefetch,
           "jobs": jobs}, open(f"experiments/waves/{name}.json", "w"), indent=1)
print(f"experiments/waves/{name}.json: {len(jobs)} jobs")
