"""Frozen-editor trajectory: weight/activation norms, bias norms, row weights, loss."""
import json, sys, torch
from hindsight import run as R
from hindsight.hindsight import Hindsight, loss_fn
from hindsight.editor import apply_edits, factors
from hindsight.nets import build
src = sys.argv[1] if len(sys.argv) > 1 else "hindsight"
cfg = json.load(open("experiments/e1_mnist.json")); cfg["delta_source"] = src
data = R.make_data(cfg); net = build(cfg)
P = net.init(R.gen_for(0, 0, 99), R.DEV)
hs = Hindsight(net, cfg, n_classes=10).to(R.DEV); hopt = torch.optim.Adam(hs.parameters(), 1e-3)
ed = R.make_editor(cfg, net); ed.requires_grad_(False)
states = {}
for t in range(201):
    x, y = data.batch("train", 128, R.gen_for(0, t))
    rec, out, D, aux, _ = hs.deltas(P, x, y)
    hopt.zero_grad(); aux.backward(); hopt.step()
    rl = R.row_loss(out, y)
    if t % 25 == 0:
        dW0, db0 = ed.edit(2, net.sites[2], rec.a["l2"], D["l2"], P["l2.w"], {}, 3, rowloss=rl)
        print(t, f"loss {loss_fn(out, y).item():.2f}",
              " ".join(f"|{k}| {v.norm():.2f}" for k, v in P.items()),
              " ".join(f"a{s.name} {rec.a[s.name].float().pow(2).mean().sqrt():.2f}" for s in net.sites),
              " ".join(f"d{s.name} {D[s.name].float().pow(2).mean().sqrt():.3f}" for s in net.sites),
              f"rowloss mean {rl.mean():.2f}")
    with torch.no_grad():
        P = apply_edits(ed, net, P, rec, D, states, detach=True, rowloss=rl)
