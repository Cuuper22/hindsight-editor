"""Colab CLI wrapper: one *wave* = one runtime running several jobs under launch/supervisor.py.

  python launch/launch.py up      experiments/waves/e1.json   # provision, start, detach
  python launch/launch.py status  e1                          # wave status (read from the HF repo)
  python launch/launch.py collect e1                          # pull run dirs (minus checkpoints) into runs/
  python launch/launch.py close   e1                          # record the finished wave in runs/ledger.csv
  python launch/launch.py stop    e1                          # emergency stop (supervisor normally self-releases)

After `up` returns, nothing depends on this machine: jobs sync to a private HF
dataset repo and the VM unassigns itself when the wave ends or hits max_hours.
Status/collect read from that repo, so they work after the VM is gone.

Wave file: {"wave": "e1", "gpu": "G4", "max_parallel": 8, "max_hours": 2,
            "prefetch": ["mnist"], "jobs": [{"run_id": .., "config": .., "overrides": {..}}]}
"""
import csv, datetime, json, os, re, subprocess, sys, tarfile, tempfile, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COLAB = "colab"
COLAB_PY = os.path.expanduser("~/AppData/Roaming/uv/tools/google-colab-cli/Scripts/python.exe")
REPO = os.environ.get("HS_REPO", "cuuper22/hindsight-runs")
HF_TOKEN_FILE = os.path.expanduser("~/.cache/huggingface/token")
STATE = os.path.join(ROOT, "runs", "_colab")
LEDGER = os.path.join(ROOT, "runs", "ledger.csv")
ENV = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1", "MSYS_NO_PATHCONV": "1"}
sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # CLI errors contain box-drawing chars
sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def _kill_tree(pid):
    if os.name == "nt":
        subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)], capture_output=True, timeout=30)
    else:
        os.kill(pid, 9)


def _run(args, timeout, input):
    """subprocess.run with a hard deadline: communicate() runs in a helper thread and the caller
    waits on that thread, so nothing a child does (open pipes, hung sockets) can block past
    `timeout`; on expiry the whole process tree is killed (colab.exe is a launcher + Python child)."""
    import threading
    p = subprocess.Popen(args, stdin=subprocess.PIPE if input is not None else subprocess.DEVNULL,
                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="utf-8",
                         errors="replace", env=ENV)
    box = {}
    th = threading.Thread(target=lambda: box.update(zip(("out", "err"), p.communicate(input=input))), daemon=True)
    th.start(); th.join(timeout)
    if th.is_alive():
        try:
            _kill_tree(p.pid)
        except Exception:
            pass
        raise subprocess.TimeoutExpired(args, timeout)
    return subprocess.CompletedProcess(args, p.returncode, box.get("out", ""), box.get("err", ""))


def sh(args, retries=5, timeout=900, check=True, input=None):
    for i in range(retries):
        try:
            r = _run(args, timeout, input)
            if r.returncode == 0 or not check:
                return r
            err = r.stderr[-800:] + r.stdout[-400:]
        except subprocess.TimeoutExpired:
            err = f"timeout after {timeout}s"
        print(f"  retry {i + 1}/{retries}: {' '.join(args[:4])} ... {err.strip()[-300:]}", flush=True)
        time.sleep(10 * (i + 1))
    raise RuntimeError(f"failed: {args[:6]}")


def cfgpath(wave):
    os.makedirs(STATE, exist_ok=True)
    return os.path.join(STATE, f"{wave}.json")


def colab(wave, *args, **kw):
    return sh([COLAB, "--config", cfgpath(wave), *args], **kw)


def usage():
    out = sh([COLAB, "usage"]).stdout
    bal = re.search(r"balance:\s*([\d.]+)", out)
    rate = re.search(r"rate:\s*([\d.]+)", out)
    return float(bal.group(1)) if bal else None, float(rate.group(1)) if rate else None


def ledger(row):
    new = not os.path.exists(LEDGER)
    with open(LEDGER, "a", newline="") as f:
        w = csv.DictWriter(f, ["wave", "runtime", "session", "start", "stop", "hours", "rate", "units",
                               "balance_before", "jobs", "note"])
        if new:
            w.writeheader()
        w.writerow(row)


def iso(t):
    return datetime.datetime.fromtimestamp(t).isoformat(timespec="seconds")


def bundle(wave, jobs):
    """Code + jobs.json in one tarball; pushed to the HF repo (reliable on flaky links)."""
    os.makedirs(STATE, exist_ok=True)
    jobs_local = os.path.join(STATE, f"{wave}.jobs.json")
    json.dump(jobs, open(jobs_local, "w"))
    path = os.path.join(STATE, f"{wave}.tgz")
    with tarfile.open(path, "w:gz") as tf:
        for d in ("hindsight", "experiments", "launch", "analysis"):
            tf.add(os.path.join(ROOT, d), arcname=d, filter=lambda t: None if "__pycache__" in t.name else t)
        tf.add(jobs_local, arcname="jobs.json")
    return path


def hf_put(local, remote, tries=6):
    from huggingface_hub import HfApi
    for i in range(tries):
        try:
            HfApi().upload_file(path_or_fileobj=local, path_in_repo=remote, repo_id=REPO, repo_type="dataset",
                                commit_message=f"launch bundle {remote}")
            return
        except Exception as e:
            print(f"  hf upload retry {i + 1}: {repr(e)[:200]}", flush=True)
            time.sleep(10 * (i + 1))
    raise RuntimeError("bundle upload failed")


def endpoints():
    r = sh([COLAB_PY, os.path.join(ROOT, "launch", "assignments.py"), "list"], retries=3, timeout=120, check=False)
    return set(r.stdout.split()) if r.returncode == 0 else None


def release_new(before, keep=None):
    """Release VMs that appeared since `before` (e.g. duplicates from a retried `colab new`)."""
    now = endpoints()
    if before is None or now is None:
        return
    stray = sorted(now - before - ({keep} if keep else set()))
    if stray:
        print(f"  releasing stray VMs: {stray}", flush=True)
        sh([COLAB_PY, os.path.join(ROOT, "launch", "assignments.py"), "unassign", *stray], retries=3, timeout=120,
           check=False)


def scrub(secret):
    """Redact a secret from every file the colab CLI writes locally (history logs and colab.log)."""
    root = os.path.expanduser("~/.config/colab-cli")
    for d, _, files in os.walk(root):
        for f in files:
            fp = os.path.join(d, f)
            try:
                txt = open(fp, encoding="utf-8", errors="surrogateescape").read()
                if secret in txt:
                    open(fp, "w", encoding="utf-8", errors="surrogateescape").write(txt.replace(secret, "<redacted>"))
            except OSError:
                pass


def send_token(wave, sess):
    """Write the HF token on the VM via `exec` (the file-upload API hangs on flaky links), then
    redact it from the colab CLI's local history log, which records exec code verbatim."""
    tok = open(HF_TOKEN_FILE).read().strip()
    code = "\n".join(["import os",
                      f"open('/content/.hf_token', 'w').write({tok!r})",
                      "print('token ok')"])
    out = ""
    try:
        out = colab(wave, "exec", "-s", sess, input=code, timeout=120).stdout
    finally:
        scrub(tok)
    if "token ok" not in out:
        raise RuntimeError("token not written")


def up(wave_file):
    import threading
    W = json.load(open(wave_file))
    wave, gpu = W["wave"], W.get("gpu", "G4")
    sess = f"hs-{wave}-{int(time.time()) % 100000}".replace("_", "-")[:40]    # fresh name every attempt
    for f in (cfgpath(wave), cfgpath(wave) + ".lock"):                        # no stale session records
        if os.path.exists(f):
            os.remove(f)
    # everything slow or flaky happens BEFORE a VM exists (nothing bills while we upload)
    tgz = bundle(wave, W["jobs"])
    hf_put(tgz, f"launch/{wave}.tgz")
    bal0, rate0 = usage()
    print(f"[{wave}] bundle on HF; balance before: {bal0}", flush=True)
    t0 = time.time()
    before = endpoints()
    try:
        colab(wave, "new", "-s", sess, *(["--gpu", gpu] if gpu != "CPU" else []), retries=3, timeout=600)
    except Exception:
        release_new(before)
        raise
    mine = json.load(open(cfgpath(wave)))[sess]["endpoint"]
    release_new(before, keep=mine)
    started = threading.Event()

    def watchdog():                                                # never leave a VM billing on a stuck launch
        if not started.wait(60 * W.get("launch_deadline_min", 15)):
            print(f"[{wave}] launch deadline passed; stopping VM", flush=True)
            sh([COLAB, "--config", cfgpath(wave), "stop", "-s", sess], check=False, retries=3, timeout=120)
            release_new(before)
            ledger({"wave": wave, "runtime": gpu, "session": sess, "start": iso(t0), "stop": iso(time.time()),
                    "hours": round((time.time() - t0) / 3600, 3), "rate": 8.9 if gpu == "G4" else None,
                    "balance_before": bal0, "note": "launch deadline: watchdog stopped VM"})
            os._exit(4)
    threading.Thread(target=watchdog, daemon=True).start()
    try:
        send_token(wave, sess)
        boot = BOOT.format(wave=wave, repo=REPO, par=W.get("max_parallel", 4), hours=W.get("max_hours", 3),
                           sync=W.get("sync_min", 5), prefetch=json.dumps(W.get("prefetch", [])),
                           pip=" ".join(W.get("pip", [])))
        out = colab(wave, "exec", "-s", sess, input=boot, timeout=1800, retries=1).stdout
        print(out[-600:], flush=True)
        pid = re.search(r"SUPERVISOR_PID (\d+)", out).group(1)
        started.set()
    except Exception:
        started.set()
        print(f"[{wave}] launch failed; stopping runtime", flush=True)
        colab(wave, "stop", "-s", sess, check=False, timeout=120)
        release_new(before)
        ledger({"wave": wave, "runtime": gpu, "session": sess, "start": iso(t0), "stop": iso(time.time()),
                "hours": round((time.time() - t0) / 3600, 3), "balance_before": bal0, "note": "launch failed"})
        raise
    _, rate = usage()
    rate = round((rate or 0) - (rate0 or 0), 2)                     # this wave's own rate (others may be running)
    json.dump({"wave": wave, "session": sess, "endpoint": mine, "gpu": gpu, "start": t0, "rate": rate, "pid": pid,
               "balance_before": bal0, "jobs": [j["run_id"] for j in W["jobs"]]},
              open(os.path.join(STATE, f"{wave}.meta.json"), "w"))
    # keep the kernel busy until the supervisor exits; this local process may die freely
    hold = f"import os,time\nwhile os.path.exists('/proc/{pid}'): time.sleep(30)\nprint('supervisor exited')"
    p = subprocess.Popen([COLAB, "--config", cfgpath(wave), "exec", "-s", sess], stdin=subprocess.PIPE,
                         stdout=open(os.path.join(STATE, f"{wave}.hold.log"), "a"), stderr=subprocess.STDOUT,
                         env=ENV, creationflags=getattr(subprocess, "DETACHED_PROCESS", 0))
    p.stdin.write(hold.encode()); p.stdin.close()
    print(f"[{wave}] running: session={sess} rate={rate}/hr pid={pid}", flush=True)


BOOT = r"""
import json, os, shutil, subprocess, sys, tarfile
os.makedirs(os.path.expanduser("~/.cache/huggingface"), exist_ok=True)
tok = os.path.expanduser("~/.cache/huggingface/token")
shutil.move("/content/.hf_token", tok); os.chmod(tok, 0o600)
from huggingface_hub import HfApi, hf_hub_download
print("hf user:", HfApi().whoami()["name"])
b = hf_hub_download("{repo}", "launch/{wave}.tgz", repo_type="dataset", local_dir="/content/_bundle")
os.makedirs("/content/hs", exist_ok=True)
tarfile.open(b).extractall("/content/hs")
shutil.copy("/content/hs/jobs.json", "/content/jobs.json")
if "{pip}".strip():
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *"{pip}".split()], check=False)
wd = "/content/runs/_waves/{wave}"
os.makedirs(wd, exist_ok=True)
p = subprocess.Popen([sys.executable, "/content/hs/launch/supervisor.py", "--wave", "{wave}",
                      "--jobs", "/content/jobs.json", "--repo", "{repo}", "--max-parallel", "{par}",
                      "--max-hours", "{hours}", "--sync-min", "{sync}", "--prefetch", '{prefetch}'], cwd="/content/hs",
                     start_new_session=True,
                     stdout=open(wd + "/supervisor.log", "a"), stderr=subprocess.STDOUT, env=os.environ)
print("SUPERVISOR_PID", p.pid)
"""


def meta(wave):
    return json.load(open(os.path.join(STATE, f"{wave}.meta.json")))


def _hf_get(path, local_dir):
    from huggingface_hub import hf_hub_download
    return hf_hub_download(REPO, path, repo_type="dataset", local_dir=local_dir, force_download=True)


def status(wave):
    try:
        st = json.load(open(_hf_get(f"runs/_waves/{wave}/status.json", STATE)))
    except Exception as e:
        print(f"[{wave}] no status in repo yet ({type(e).__name__}); first sync is a few minutes after start")
        return None
    counts = {}
    for j in st["jobs"].values():
        counts[j["state"]] = counts.get(j["state"], 0) + 1
    M = meta(wave)
    eps = endpoints()
    alive = None if eps is None else (M.get("endpoint") in eps if M.get("endpoint") else bool(eps))
    print(f"[{wave}] {st['hours']:.2f}h (synced {time.time() - st['now']:.0f}s ago) gpu[{st['gpu']}] "
          f"final={st['final']} {counts} VM={'unknown' if alive is None else 'alive' if alive else 'released'}")
    return st


def collect(wave, with_ckpt=False):
    from huggingface_hub import snapshot_download
    rids = meta(wave)["jobs"]
    snapshot_download(REPO, repo_type="dataset", local_dir=ROOT,
                      allow_patterns=[f"runs/{r}/**" for r in rids] + [f"runs/_waves/{wave}/*"],
                      ignore_patterns=None if with_ckpt else ["*ckpt.pt"])
    print(f"[{wave}] collected {len(rids)} runs into runs/")


def close(wave, note=""):
    """Record the wave in the ledger, using the supervisor's last status as the stop time."""
    M = meta(wave)
    st = status(wave)
    stop = st["now"] if st else time.time()
    hours = (stop - M["start"]) / 3600
    ledger({"wave": wave, "runtime": M["gpu"], "session": M["session"], "start": iso(M["start"]), "stop": iso(stop),
            "hours": round(hours, 3), "rate": M["rate"], "units": round(hours * (M["rate"] or 0), 2),
            "balance_before": M["balance_before"], "jobs": len(M["jobs"]), "note": note or (st or {}).get("final")})


if __name__ == "__main__":
    cmd, arg = sys.argv[1], sys.argv[2]
    if cmd == "up":
        up(arg)
    elif cmd == "status":
        status(arg)
    elif cmd == "collect":
        collect(arg, with_ckpt="--ckpt" in sys.argv)
    elif cmd == "close":
        close(arg, " ".join(sys.argv[3:]))
    elif cmd == "stop":
        colab(arg, "stop", "-s", meta(arg)["session"], check=False)
        close(arg, "manual stop")
