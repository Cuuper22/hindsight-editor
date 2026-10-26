"""Trace Phase-0 imitation: per-site cos(edit, Adam label), edit/label RMS, learned lr, val loss."""
import json, torch, torch.nn.functional as F
from hindsight import run as R
from hindsight.hindsight import Hindsight, backprop_error, loss_fn
from hindsight.editor import apply_edits
from hindsight.nets import build
cfg = json.load(open("experiments/e1_mnist.json"))
data = R.make_data(cfg); net = build(cfg)
P = net.init(R.gen_for(0, 0, 99), R.DEV)
hs = Hindsight(net, cfg, n_classes=10).to(R.DEV); hopt = torch.optim.Adam(hs.parameters(), 1e-3)
ed = R.make_editor(cfg, net); eopt = torch.optim.Adam(ed.parameters(), 1e-3)
states, shadow = {}, {}
xv, yv = data.batch("val", 2000, R.gen_for(0, 0, 5))
for t in range(301):
    x, y = data.batch("train", 128, R.gen_for(0, t))
    rec, out, D, aux, _ = hs.deltas(P, x, y)
    hopt.zero_grad(); aux.backward(); hopt.step()
    bp = backprop_error(net, P, x, y); G = R.site_grads(net, P, rec.a, bp)
    lab = {k: R.adam_label(shadow.setdefault(k, {}), g, 1e-3) for k, g in G.items()}
    newP = apply_edits(ed, net, P, rec, D, states, rowloss=R.row_loss(out, y))
    il, info = 0.0, []
    for s in net.sites:
        e, l = (newP[s.w] - P[s.w]).flatten(), lab[s.w].flatten()
        cos = F.cosine_similarity(e, l, 0)
        il = il + (1 - cos) + 0.5 * (e.pow(2).mean().log() - l.pow(2).mean().log()).pow(2)
        info.append(f"{s.name} cos {cos.item():+.2f} e {e.pow(2).mean().sqrt().item():.1e} l {l.pow(2).mean().sqrt().item():.1e}")
    eopt.zero_grad(); il.backward(); eopt.step()
    R.detach_states(states)
    P = {k: v.detach() for k, v in newP.items()}
    if t % 50 == 0:
        with torch.no_grad():
            vl = loss_fn(net.forward(P, xv), yv).item()
        lr = ed.lr_head.bias.data.exp().tolist()
        print(t, f"val {vl:.3f}", " | ".join(info), f"lr_b {lr[0]:.1e}")
