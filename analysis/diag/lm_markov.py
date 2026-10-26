"""Does any update rule learn a synthetic bigram language below unigram? (no dataset download)

A sparse random Markov chain over 64 tokens: unigram entropy ~4.1 nats, bigram ~1.4 nats.
Tiny GPT (d=64, 2 layers, ctx 32). Compares Adam, raw hindsight rule, untrained editor.
"""
import json, math, sys, torch
from hindsight import run as R

V, N = 64, 400_000
g = torch.Generator().manual_seed(0)
T = torch.zeros(V, V)
for i in range(V):                                                  # each token has 3 likely successors
    T[i, torch.randperm(V, generator=g)[:3]] = torch.tensor([0.6, 0.3, 0.1])
seq = torch.empty(N, dtype=torch.long); seq[0] = 0
for t in range(1, N):
    seq[t] = torch.multinomial(T[seq[t - 1]], 1, generator=g)


class Markov:
    vocab = V
    def __init__(self, cfg): self.ctx = cfg["ctx"]
    def batch(self, split, bs, gen):
        lo, hi = (0, int(0.9 * N)) if split == "train" else (int(0.9 * N), N)
        i = lo + torch.randint(hi - lo - self.ctx - 1, (bs,), generator=gen)
        s = seq[i[:, None] + torch.arange(self.ctx + 1)].to(R.DEV); return s[:, :-1], s[:, 1:]
    def unigram_nats(self):
        p = torch.bincount(seq, minlength=V).double() / N; return float(-(p * p.clamp_min(1e-12).log()).sum())


R.make_data = lambda cfg: Markov(cfg); R.TextData = Markov
base = json.load(open("experiments/e3_enwik8.json"))
size = sys.argv[1] if len(sys.argv) > 1 else "tiny"
arch = {"tiny": dict(d=64, n_layer=2, n_head=2, ctx=32), "e3": dict(d=384, n_layer=6, n_head=6, ctx=256),
        "wide": dict(d=384, n_layer=2, n_head=6, ctx=32), "deep": dict(d=64, n_layer=6, n_head=2, ctx=32),
        "long": dict(d=64, n_layer=2, n_head=2, ctx=256)}[size]
base.update(vocab=V, bs=32, steps=1500, eval_every=500, eval_batches=10, eval_bs=32, log_every=500, warmup=50,
            bottleneck=16, vib=False, **arch)
print("unigram", round(Markov(base).unigram_nats(), 3), "nats; bigram ~", round(float(-(T * T.clamp_min(1e-12).log()).sum(1).mean()), 3))
runs = {
        "editor untrained lr1e-2": {"method": "editor", "editor": {"rank": 16, "lr_init": 1e-2, "use_id": False, "max_rel": 0.05}}}
for name, ov in runs.items():
    cfg = {**base, **ov, "run_id": "markov"}
    out = f"runs/_diag_markov_{name.replace(' ', '_')}"
    import shutil; shutil.rmtree(out, ignore_errors=True)
    import io, contextlib
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            R.run_train(cfg, R.Logger(out, cfg), out)
    except Exception as e:
        print(f"{name:26s} crashed: {repr(e)[:80]}")
    ev = [json.loads(l) for l in open(f"{out}/metrics.jsonl") if '"event": "eval"' in l]
    print(f"{name:26s}", " ".join(f"{e['step']}:{e['val_loss']:.3f}" for e in ev))
