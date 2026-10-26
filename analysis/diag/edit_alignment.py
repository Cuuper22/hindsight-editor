"""Where is the signal lost? cos(-grad, raw D^T A), cos(-grad, editor sketch), edit RMS vs Adam step RMS."""
import json, sys, torch, torch.nn.functional as F
from hindsight import run as R
from hindsight.hindsight import Hindsight, backprop_error, loss_fn
from hindsight.editor import Editor, factors
from hindsight.nets import build
cfg = json.load(open("experiments/e1_mnist.json"))
data = R.make_data(cfg); net = build(cfg)
P = net.init(R.gen_for(0, 0, 99), R.DEV)
warm = int(sys.argv[1]) if len(sys.argv) > 1 else 0
if warm:
    P = R.adam_warm(net, P, data, {**cfg, "warm_lr": 1e-3}, warm, 0)
for src in ("hindsight", "hindsight_lc", "ftp_seq", "dfa"):
    hs = Hindsight(net, {**cfg, "delta_source": src}, n_classes=10).to(R.DEV)
    hopt = torch.optim.Adam(hs.parameters(), 1e-3)
    for t in range(300):                                    # let the channel learn first
        x, y = data.batch("train", 128, R.gen_for(1, t))
        _, _, _, aux, _ = hs.deltas(P, x, y)
        if aux.requires_grad:
            hopt.zero_grad(); aux.backward(); hopt.step()
    x, y = data.batch("train", 128, R.gen_for(2, 0))
    rec, out, D, _, _ = hs.deltas(P, x, y)
    bp = backprop_error(net, P, x, y)
    G = R.site_grads(net, P, rec.a, bp)
    ed = Editor(len(net.sites), rank=8).to(R.DEV)
    rl = R.row_loss(out, y)
    row = []
    for i, s in enumerate(net.sites):
        A, Dm = factors(s, rec.a[s.name], D[s.name])
        raw = (Dm.float().t() @ A.float())
        err = 1 - out.detach().float().softmax(-1).gather(1, y[:, None]).squeeze(1)   # 1 - p_y, forward only
        rawW = ((Dm.float() * err.repeat_interleave(Dm.shape[0] // len(err))[:, None]).t() @ A.float())
        g = -G[s.w].reshape(raw.shape)
        dW, _ = ed.edit(i, s, rec.a[s.name], D[s.name], P[s.w], {}, len(net.sites), rowloss=rl)
        c = lambda u: F.cosine_similarity(u.flatten().float(), g.flatten(), 0).item()
        row.append(f"{s.name}: raw {c(raw):+.3f} err-weighted {c(rawW):+.3f} sketch {c(dW.detach()):+.3f} |edit| {dW.detach().pow(2).mean().sqrt():.1e}")
    print(f"warm={warm} {src:9s}", " | ".join(row))
