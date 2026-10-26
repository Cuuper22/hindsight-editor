"""Runs ON the Colab VM, in its own session (survives the CLI/kernel disconnecting).

Runs a wave's jobs (several processes packed on one GPU) under /content/runs, and
pushes them to a private Hugging Face dataset repo every --sync-min minutes (one
commit per wave). On start it pulls any earlier state of its runs from the repo,
so a re-launched wave resumes from the last synced checkpoints. Crashed jobs are
re-launched (resuming). When everything is finished -- or the wall-clock cap is
hit -- it does a final push and unassigns the VM so no compute units burn idle.

  python supervisor.py --wave W --jobs jobs.json --repo user/hindsight-runs
"""
import argparse, json, os, signal, subprocess, sys, threading, time

ap = argparse.ArgumentParser()
ap.add_argument("--wave", required=True)
ap.add_argument("--jobs", required=True)
ap.add_argument("--repo", required=True)
ap.add_argument("--root", default="/content/runs")
ap.add_argument("--code", default="/content/hs")
ap.add_argument("--max-parallel", type=int, default=4)
ap.add_argument("--max-hours", type=float, default=3.0)
ap.add_argument("--sync-min", type=float, default=5.0)
ap.add_argument("--retries", type=int, default=2)
ap.add_argument("--no-unassign", action="store_true")
ap.add_argument("--prefetch", default="[]", help="datasets to download before starting jobs")
args = ap.parse_args()

from huggingface_hub import HfApi, snapshot_download

api = HfApi()
jobs = json.load(open(args.jobs))
rids = [j["run_id"] for j in jobs]
wave_dir = os.path.join(args.root, "_waves", args.wave)
os.makedirs(wave_dir, exist_ok=True)
t_start = time.time()
state = {r: {"state": "pending", "tries": 0} for r in rids}
running = {}
sync_lock = threading.Lock()


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def write_status(final=None):
    gpu = ""
    try:
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total",
                              "--format=csv,noheader"], capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:
        pass
    st = {"wave": args.wave, "start": t_start, "now": time.time(), "hours": (time.time() - t_start) / 3600,
          "gpu": gpu, "final": final, "jobs": state}
    tmp = os.path.join(wave_dir, "status.json.tmp")
    json.dump(st, open(tmp, "w"), indent=1)
    os.replace(tmp, os.path.join(wave_dir, "status.json"))


def sync(tries=1):
    """Push this wave's run dirs + status to the repo (unchanged files are skipped)."""
    for i in range(tries):
        try:
            with sync_lock:
                api.upload_folder(repo_id=args.repo, repo_type="dataset", folder_path=args.root, path_in_repo="runs",
                                  allow_patterns=[f"{r}/**" for r in rids] + [f"_waves/{args.wave}/**"],
                                  ignore_patterns=["**/*.tmp", "**/__pycache__/**"],
                                  commit_message=f"{args.wave} sync {time.strftime('%H:%M:%S')}")
            return True
        except Exception as e:
            log("sync failed:", repr(e)[:300])
            time.sleep(min(60, 10 * (i + 1)))
    return False


def syncer():
    while True:
        time.sleep(60 * args.sync_min)
        write_status()
        sync()


def restore():
    try:
        extra = [x for j in jobs for x in j.get("restore", [])]   # other runs a job reads (e.g. an editor.pt)
        snapshot_download(args.repo, repo_type="dataset", local_dir=os.path.dirname(args.root),
                          allow_patterns=[f"runs/{r}/*" for r in rids] + [f"runs/{x}/editor.pt" for x in extra])
        for r in rids:
            if os.path.exists(os.path.join(args.root, r, "DONE")):
                state[r]["state"] = "done"
        log("restored from repo;", sum(s["state"] == "done" for s in state.values()), "runs already done")
    except Exception as e:
        log("restore skipped:", repr(e)[:300])


def start(job):
    rid = job["run_id"]
    out = os.path.join(args.root, rid)
    os.makedirs(out, exist_ok=True)
    cmd = [sys.executable, "-m", "hindsight.run", os.path.join(args.code, job["config"]), "--out", out]
    cmd += [f"{k}={json.dumps(v)}" for k, v in job.get("overrides", {}).items()]
    env = {**os.environ, "HS_DATA": "/content/data", "PYTHONUNBUFFERED": "1"}
    f = open(os.path.join(out, "stdout.log"), "a")
    p = subprocess.Popen(cmd, cwd=args.code, env=env, stdout=f, stderr=subprocess.STDOUT)
    state[rid].update(state="running", tries=state[rid]["tries"] + 1, pid=p.pid, started=time.time())
    running[rid] = (p, f)
    log("start", rid, "try", state[rid]["tries"])


final = "done"
try:
    restore()
    threading.Thread(target=syncer, daemon=True).start()
    sys.path.insert(0, args.code)
    os.environ["HS_DATA"] = "/content/data"
    import hindsight.data as D
    for name in json.loads(args.prefetch):                          # once, before jobs race for it
        t = time.time()
        {"mnist": D.mnist, "cifar10": D.cifar10, "enwik8": D.enwik8}[name]()
        log("prefetched", name, f"{time.time() - t:.0f}s")
    while True:
        for rid, (p, f) in list(running.items()):
            rc = p.poll()
            if rc is None:
                continue
            f.close(); del running[rid]
            if os.path.exists(os.path.join(args.root, rid, "DONE")):
                state[rid].update(state="done", rc=rc, ended=time.time())
            elif state[rid]["tries"] <= args.retries:
                state[rid].update(state="pending", rc=rc)              # resumes from its checkpoint
            else:
                state[rid].update(state="failed", rc=rc, ended=time.time())
            log("exit", rid, rc, state[rid]["state"])
        pending = [j for j in jobs if state[j["run_id"]]["state"] == "pending"]
        while pending and len(running) < args.max_parallel:
            start(pending.pop(0))
        if not running and not pending:
            break
        if time.time() - t_start > args.max_hours * 3600:
            final = "deadline"
            log("deadline reached; stopping jobs (they keep their last checkpoint)")
            for rid, (p, f) in running.items():
                p.send_signal(signal.SIGTERM)
            time.sleep(30)
            for rid, (p, f) in running.items():
                p.kill(); state[rid]["state"] = "deadline"
            break
        write_status()
        time.sleep(20)
except Exception as e:                                              # never leave the VM running on a bug
    final = f"supervisor_error: {e!r}"
    log(final)
write_status(final)
log("finished:", final)
log("final sync:", "ok" if sync(tries=6) else "FAILED")
if not args.no_unassign:
    from google.colab import runtime
    log("unassigning VM")
    runtime.unassign()
