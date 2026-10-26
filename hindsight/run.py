"""Single entry point: python -m hindsight.run CONFIG.json [--out DIR] [key=value ...]

modes
  e0     : backprop-trained net; track cosine(d_l, -dL/dz_l) per layer for each delta source
  train  : train a fresh net with method in {adam, raw, editor}; editor may learn online
           (editor_train = imitation) or be frozen (editor_ckpt)
  meta   : Phase-1 meta-training of the editor (held-out loss after its edits)

Every run writes OUT/metrics.jsonl (one JSON per log event), OUT/summary.json at the
end, and OUT/ckpt.pt periodically (resumed automatically if present).
"""
import argparse, json, math, os, sys, time
import torch
import torch.nn.functional as F

from . import flops
from .data import ImageData, TextData
from .editor import Editor, apply_edits, factors
from .hindsight import Hindsight, loss_fn, backprop_error, per_example_cos
from .nets import Rec, build

DEV = "cuda" if torch.cuda.is_available() else "cpu"


# ---------------------------------------------------------------------------------
class Logger:
    def __init__(self, out, cfg):
        os.makedirs(out, exist_ok=True)
        self.out, self.f = out, open(os.path.join(out, "metrics.jsonl"), "a")
        self.wb = None
        if cfg.get("wandb") and os.environ.get("WANDB_API_KEY"):
            try:
                import wandb
                self.wb = wandb.init(project="hindsight-editor", name=cfg.get("run_id"), config=cfg,
                                     resume="allow", id=cfg.get("run_id"), settings=wandb.Settings(init_timeout=30))
            except Exception as e:                                   # never block on W&B
                print("wandb disabled:", e, flush=True)

    def log(self, **kw):
        kw["time"] = time.time()
        self.f.write(json.dumps(kw) + "\n"); self.f.flush()
        if self.wb is not None:
            try:
                self.wb.log({k: v for k, v in kw.items() if isinstance(v, (int, float))})
            except Exception:
                self.wb = None
        print(json.dumps({k: (round(v, 5) if isinstance(v, float) else v) for k, v in kw.items()}), flush=True)


def make_data(cfg):
    if cfg["task"] in ("mnist", "cifar10"):
        return ImageData(cfg["task"], DEV, seed=cfg.get("data_seed", 0))
    return TextData(cfg["task"], DEV, cfg["ctx"], cfg.get("n_train_tokens"))


def autocast():
    return torch.autocast("cuda", dtype=torch.bfloat16, enabled=DEV == "cuda")


def gen_for(seed, step, salt=0):
    return torch.Generator().manual_seed(seed * 1_000_003 + step * 7919 + salt)


@torch.no_grad()
def evaluate(net, P, data, cfg, split="val", max_batches=None):
    """Mean loss (and accuracy for classifiers) over a fixed held-out set."""
    tot, correct, n = 0.0, 0, 0
    with flops.paused(), autocast():
        if isinstance(data, ImageData):
            for x, y in data.iterate(split, 2000):
                out = net.forward(P, x)
                tot += F.cross_entropy(out.float(), y, reduction="sum").item()
                correct += (out.argmax(-1) == y).sum().item(); n += len(y)
            return tot / n, correct / n
        nb = max_batches or cfg.get("eval_batches", 20)
        for b in range(nb):
            x, y = data.batch(split, cfg.get("eval_bs", cfg["bs"]), gen_for(12345, b, 1))
            tot += loss_fn(net.forward(P, x), y).item()
        return tot / nb, None


def peak_mem():
    return torch.cuda.max_memory_allocated() / 2 ** 30 if DEV == "cuda" else 0.0


def sync():
    if DEV == "cuda":
        torch.cuda.synchronize()


# ---------------------------------------------------------------------------------
# E0: cosine(d_l, -backprop error) along a backprop trajectory
def run_e0(cfg, log):
    data = make_data(cfg)
    net = build(cfg)
    seed = cfg["seed"]
    torch.manual_seed(seed)
    P = net.init(gen_for(seed, 0, 99), DEV)
    for v in P.values():
        v.requires_grad_(True)
    opt = torch.optim.Adam(P.values(), lr=cfg.get("lr", 1e-3))
    variants = cfg.get("variants", [{"name": "hindsight", "delta_source": "hindsight"}])
    hs = {}
    for v in variants:
        h = Hindsight(net, {**cfg, **v}, n_classes=data.n_classes).to(DEV)
        hs[v["name"]] = (h, torch.optim.Adam(h.parameters(), lr=cfg.get("chan_lr", 1e-3)))
    probe_at = set(cfg.get("probe_at", [0, 50, 200, 500, 1000, 2000, 3000]))
    xv, yv = data.batch("val", 2000, gen_for(seed, 0, 5))
    for t in range(cfg["steps"] + 1):
        x, y = data.batch("train", cfg["bs"], gen_for(seed, t))
        # channels learn online from their own readout loss (main weights untouched)
        for name, (h, hopt) in hs.items():
            Pd = {k: v.detach() for k, v in P.items()}
            _, _, _, aux, _ = h.deltas(Pd, x, y)
            if aux.requires_grad:
                hopt.zero_grad(); aux.backward(); hopt.step()
        if t in probe_at:
            Pd = {k: v.detach() for k, v in P.items()}
            bp = backprop_error(net, Pd, xv, yv)
            for name, (h, _) in hs.items():
                with torch.no_grad():
                    _, _, D, _, info = h.deltas(Pd, xv, yv)
                cos = {s.name: per_example_cos(D[s.name], -bp[s.name]) for s in net.sites}
                log.log(event="cos", step=t, variant=name,
                        **{f"cos_{k}": c.mean().item() for k, c in cos.items()},
                        **{f"cos_{k}_frac_pos": (c > 0).float().mean().item() for k, c in cos.items()},
                        **({f"readout_{i}": r.item() for i, r in enumerate(info.get("readout", []))}))
            vl, va = evaluate(net, Pd, data, cfg)
            log.log(event="eval", step=t, val_loss=vl, val_acc=va)
        if t == cfg["steps"]:
            break
        out = net.forward(P, x)
        opt.zero_grad(); loss_fn(out, y).backward(); opt.step()


# ---------------------------------------------------------------------------------
# raw hindsight rule: dW = lr * D^T A / N with momentum 0.9 -- SGD on the local regression
# loss 1/2||W a - z_sighted||^2 (FTP-like). Not normalised: the delta's magnitude is the error signal.
class RawRule:
    def __init__(self, lr, mom=0.9, max_rows=4096):
        self.lr, self.mom, self.max_rows, self.m = lr, mom, max_rows, {}

    @torch.no_grad()
    def edits(self, net, P, rec, D):
        for s in net.sites:
            A, Dm = factors(s, rec.a[s.name], D[s.name])
            N = Dm.shape[0] / (Dm.shape[0] // rec.z[s.name].shape[0])  # per-example mean (conv/LM: sum over positions)
            if s.kind == "embed":
                dW = torch.zeros_like(P[s.w], dtype=torch.float32).index_add_(0, A, Dm.float()) / N
            else:
                dW = (Dm.float().t() @ A.float()).reshape(P[s.w].shape) / N
                flops.add("editor", 2 * Dm.numel() * A.shape[1])
            m = self.m.get(s.name)
            self.m[s.name] = m = dW if m is None else self.mom * m + dW
            P[s.w] += self.lr * m.to(P[s.w].dtype)
            if s.b is not None:
                mb = self.m.get(s.name + ".b")
                g = Dm.float().sum(0) / N
                self.m[s.name + ".b"] = mb = g if mb is None else self.mom * mb + g
                P[s.b] += self.lr * mb.to(P[s.b].dtype)


def adam_label(state, g, lr, b1=0.9, b2=0.999, eps=1e-8):
    """Shadow Adam step (the Phase-0 imitation target) for gradient g."""
    t = state["t"] = state.get("t", 0) + 1
    m = state["m"] = b1 * state.get("m", torch.zeros_like(g)) + (1 - b1) * g
    v = state["v"] = b2 * state.get("v", torch.zeros_like(g)) + (1 - b2) * g * g
    return -lr * (m / (1 - b1 ** t)) / ((v / (1 - b2 ** t)).sqrt() + eps)


def site_grads(net, P, rec_a, bp):
    """dL/dW from backprop errors and recorded inputs (labels only)."""
    G = {}
    for s in net.sites:
        A, Dm = factors(s, rec_a[s.name], bp[s.name])
        if s.kind == "embed":
            G[s.w] = torch.zeros_like(P[s.w], dtype=torch.float32).index_add_(0, A, Dm.float())
        else:
            G[s.w] = (Dm.float().t() @ A.float()).reshape(P[s.w].shape)
        if s.b is not None:
            G[s.b] = Dm.float().sum(0)
    return G


def make_editor(cfg, net):
    e = cfg.get("editor", {})
    return Editor(len(net.sites), rank=e.get("rank", 8), hidden=e.get("hidden", 8),
                  max_rows=e.get("max_rows", 4096), use_a=e.get("use_a", True),
                  lr_init=e.get("lr_init", 3e-4), row_loss=e.get("row_loss", True),
                  max_rel=e.get("max_rel", 0.02), use_id=e.get("use_id", True)).to(DEV)


def row_loss(logits, y):
    """Per-example (per-token) blind loss: forward-pass only, fed to the editor."""
    return F.cross_entropy(logits.detach().float().reshape(-1, logits.shape[-1]), y.reshape(-1),
                           reduction="none").reshape(y.shape)


def detach_states(states):
    for st in states.values():
        for k, v in list(st.items()):
            if torch.is_tensor(v):
                st[k] = v.detach()


# ---------------------------------------------------------------------------------
def run_train(cfg, log, out):
    data = make_data(cfg)
    net = build(cfg)
    seed, method = cfg["seed"], cfg["method"]
    torch.manual_seed(seed)
    P = net.init(gen_for(seed, 0, 99), DEV)
    n_params = sum(v.numel() for v in P.values())
    hs = hopt = editor = eopt = opt = rule = None
    if method == "adam":
        for v in P.values():
            v.requires_grad_(True)
        opt = torch.optim.AdamW(P.values(), lr=cfg["lr"], weight_decay=cfg.get("wd", 0.0),
                                betas=tuple(cfg.get("betas", (0.9, 0.999))))
    else:
        hs = Hindsight(net, cfg, n_classes=getattr(data, "n_classes", None),
                       vocab=None if isinstance(data, ImageData) else net.vocab).to(DEV)  # model vocab (may be padded)
        hopt = torch.optim.Adam(hs.parameters(), lr=cfg.get("chan_lr", 1e-3))
        if method == "raw":
            rule = RawRule(cfg["lr"], cfg.get("mom", 0.9))
        else:
            editor = make_editor(cfg, net)
            if cfg.get("editor_ckpt"):
                editor.load_state_dict(torch.load(cfg["editor_ckpt"], map_location=DEV)["editor"])
            if cfg.get("editor_train") == "imitation":
                eopt = torch.optim.Adam(editor.parameters(), lr=cfg.get("meta_lr", 1e-3))
            else:
                editor.requires_grad_(False)
    states, shadow, se_state = {}, {}, {"baseline": 0.0, "log": {}}
    sched = cfg.get("sched", "const")
    t0, flop_tot, wall = 0, {}, 0.0
    ck = os.path.join(out, "ckpt.pt")
    if os.path.exists(ck):
        s = torch.load(ck, map_location=DEV, weights_only=False)
        for k in P:
            P[k].data.copy_(s["P"][k])
        t0, flop_tot, wall, states, shadow = s["t"], s["flops"], s["wall"], s["states"], s["shadow"]
        for obj, key in ((opt, "opt"), (hs, "hs"), (hopt, "hopt"), (editor, "editor"), (eopt, "eopt")):
            if obj is not None and s.get(key) is not None:
                obj.load_state_dict(s[key])
        if rule is not None:
            rule.m = s["rule_m"]
        log.log(event="resume", step=t0)
    else:
        vl, va = evaluate(net, P, data, cfg)
        log.log(event="eval", step=0, val_loss=vl, val_acc=va, flops=0.0, wall=0.0, n_params=n_params)
    if DEV == "cuda":
        torch.cuda.reset_peak_memory_stats()
    steps = cfg["steps"]
    for t in range(t0, steps):
        x, y = data.batch("train", cfg["bs"], gen_for(seed, t))
        frac = t / steps
        mult = 1.0 if sched == "const" else 0.5 * (1 + math.cos(math.pi * frac)) if sched == "cos" else \
            min(1.0, (t + 1) / cfg.get("warmup", 1)) * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * frac)))
        sync(); tic = time.time()
        info, imit = {}, None
        with flops.measure() as fm:
            if method == "adam":
                for g in opt.param_groups:
                    g["lr"] = cfg["lr"] * mult
                f0 = flops.snapshot().get("fwd", 0.0)
                with autocast():
                    out_ = net.forward(P, x)
                    loss = loss_fn(out_, y)
                fwd = flops.snapshot().get("fwd", 0.0) - f0
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
                flops.add("bwd", 2 * fwd); flops.add("opt", 12 * n_params)
            else:
                Pd = {k: v.detach() for k, v in P.items()}
                # blind pass builds no graph (detached weights); the sighted pass keeps a
                # local graph into the channel/readouts only, for their readout loss
                with autocast():
                    rec, out_, D, aux, info = hs.deltas(Pd, x, y)
                hopt.zero_grad(set_to_none=True)
                if aux.requires_grad:
                    aux.backward(); hopt.step()
                loss = loss_fn(out_, y)
                rl = row_loss(out_, y)
                if method == "raw":
                    rule.lr = cfg["lr"] * mult
                    rule.edits(net, P, rec, D)
                else:
                    if eopt is not None:                            # Phase 0: imitate Adam on backprop grads
                        with flops.paused():
                            bp = backprop_error(net, Pd, x, y)
                            G = site_grads(net, Pd, rec.a, bp)
                            labels = {k: adam_label(shadow.setdefault(k, {}), g, cfg.get("label_lr", 1e-3)) for k, g in G.items()}
                            if cfg.get("label", "adam") == "sgd":       # gradient direction, Adam's step size
                                labels = {k: -G[k] * (lab.pow(2).mean().sqrt() / G[k].pow(2).mean().add(1e-20).sqrt())
                                          for k, lab in labels.items()}
                        with torch.enable_grad():
                            newP = apply_edits(editor, net, Pd, rec, D, states, rowloss=rl)
                            il = 0.0
                            for s in net.sites:
                                for key in (s.w, s.b):
                                    if key is None:
                                        continue
                                    e, lab = (newP[key] - Pd[key]).flatten().float(), labels[key].flatten()
                                    il = il + (1 - F.cosine_similarity(e, lab, 0)) + \
                                        (e.pow(2).mean().add(1e-20).log() - lab.pow(2).mean().add(1e-20).log()).pow(2) * cfg.get("imit_mag", 0.5)
                            eopt.zero_grad(set_to_none=True); il.backward(); eopt.step()
                        imit = il.item() / len(net.sites)
                        detach_states(states)
                        with torch.no_grad():
                            for k in P:
                                P[k] = newP[k].detach()
                    elif cfg.get("self_edit"):                     # Phase 2: the editor edits itself
                        P = self_edit_step(cfg["self_edit"], editor, net, P, rec, D, states, rl, data, cfg, t, se_state)
                        info.update({k: torch.tensor(v) for k, v in se_state["log"].items()})
                    else:
                        with torch.no_grad():
                            P = apply_edits(editor, net, P, rec, D, states, detach=True, rowloss=rl)
        sync(); wall += time.time() - tic
        for k, v in fm.items():
            flop_tot[k] = flop_tot.get(k, 0.0) + v
        if (t + 1) % cfg.get("log_every", 50) == 0:
            rec_ = {"event": "train", "step": t + 1, "loss": loss.item(), "flops": sum(flop_tot.values()), "wall": wall}
            if imit is not None:
                rec_["imit_loss"] = imit
            if "readout" in info:
                rec_["readout_mean"] = info["readout"].mean().item()
            for k in ("self_reward", "self_lr_w", "self_mom"):
                if k in info:
                    rec_[k] = float(info[k])
            log.log(**rec_)
        if (t + 1) % cfg.get("eval_every", 500) == 0 or t + 1 == steps:
            vl, va = evaluate(net, P, data, cfg)
            log.log(event="eval", step=t + 1, val_loss=vl, val_acc=va, flops=sum(flop_tot.values()),
                    flops_by=dict(flop_tot), wall=wall, peak_mem_gb=peak_mem())
            if not math.isfinite(vl):
                log.log(event="diverged", step=t + 1)
                break
        if (t + 1) % cfg.get("ckpt_every", 2000) == 0:
            save = {"P": P, "t": t + 1, "flops": flop_tot, "wall": wall, "states": states, "shadow": shadow,
                    "rule_m": rule.m if rule else None}
            for obj, key in ((opt, "opt"), (hs, "hs"), (hopt, "hopt"), (editor, "editor"), (eopt, "eopt")):
                save[key] = obj.state_dict() if obj is not None else None
            torch.save(save, ck + ".tmp"); os.replace(ck + ".tmp", ck)
    final = {}
    if isinstance(data, ImageData):
        final["test_loss"], final["test_acc"] = evaluate(net, P, data, cfg, "test")
    final["val_loss"], final["val_acc"] = evaluate(net, P, data, cfg, "val")
    if not isinstance(data, ImageData):
        final["unigram_nats"] = data.unigram_nats()                # the E3 'learns at all' reference
    if editor is not None and eopt is not None:
        torch.save({"editor": editor.state_dict(), "cfg": cfg}, os.path.join(out, "editor.pt"))
    return {**final, "flops": sum(flop_tot.values()), "flops_by": flop_tot, "wall": wall,
            "peak_mem_gb": peak_mem(), "n_params": n_params, "steps": steps}


@torch.no_grad()
def self_edit_step(se, editor, net, P, rec, D, states, rl, data, cfg, t, st):
    """Phase 2 (E5): the editor edits its own control weights (step size, momentum, per-row weighting
    heads) by weight perturbation. The hindsight signal is the realised held-out improvement of its
    own edit (two extra forward passes, no gradient):
        theta += eta * (r - baseline) / eps * u,   r = log L_before - log L_after on a held-out batch,
    where u is the random +-1 direction the editor tried this step."""
    params = [editor.lr_head.weight, editor.lr_head.bias, editor.mom_head.bias, editor.score.weight, editor.score.bias]
    eps, eta = se.get("eps", 0.01), se.get("eta", 0.01)
    xv, yv = data.batch("val", cfg["bs"], gen_for(cfg["seed"], t, 29))
    with autocast():
        lb = loss_fn(net.forward(P, xv), yv).float()
    us = [torch.randint(0, 2, p.shape, device=p.device).float() * 2 - 1 for p in params]
    for p_, u in zip(params, us):
        p_.add_(eps * u)
    P = apply_edits(editor, net, P, rec, D, states, detach=True, rowloss=rl)
    with autocast():
        la = loss_fn(net.forward(P, xv), yv).float()
    r = float(lb.log() - la.log()) if torch.isfinite(la) else -1.0
    adv = r - st["baseline"]
    st["baseline"] = 0.9 * st["baseline"] + 0.1 * r
    for p_, u in zip(params, us):
        p_.add_(-eps * u + eta * max(-1.0, min(1.0, adv / eps)) * u)
    st["log"] = {"self_reward": r, "self_lr_w": float(editor.lr_head.bias[0].exp()),
                 "self_mom": float(torch.sigmoid(editor.mom_head.bias[0]))}
    return P


# ---------------------------------------------------------------------------------
# Phase 1: meta-train the editor on held-out loss after its own edits
def run_meta(cfg, log, out):
    data = make_data(cfg)
    net = build(cfg)
    seed = cfg["seed"]
    torch.manual_seed(seed)
    editor = make_editor(cfg, net)
    if cfg.get("editor_init"):
        editor.load_state_dict(torch.load(cfg["editor_init"], map_location=DEV)["editor"])
    mopt = torch.optim.Adam(editor.parameters(), lr=cfg.get("meta_lr", 1e-3))
    K, T_inner, n_meta = cfg.get("unroll", 4), cfg.get("inner_steps", 1000), cfg["meta_steps"]
    m0 = 0
    ck = os.path.join(out, "ckpt.pt")
    if os.path.exists(ck):
        s = torch.load(ck, map_location=DEV, weights_only=False)
        editor.load_state_dict(s["editor"]); mopt.load_state_dict(s["mopt"]); m0 = s["m"]
        log.log(event="resume", step=m0)
    P = hs = hopt = None
    states, t_in, traj, l_start, skipped = {}, 0, 0, None, 0
    for m in range(m0, n_meta):
        if P is None or t_in >= T_inner:
            l_start = None
            traj += 1
            P = net.init(gen_for(seed, m, 77), DEV)
            hs = Hindsight(net, cfg, n_classes=getattr(data, "n_classes", None),
                           vocab=None if isinstance(data, ImageData) else net.vocab).to(DEV)  # model vocab (may be padded)
            hopt = torch.optim.Adam(hs.parameters(), lr=cfg.get("chan_lr", 1e-3))
            states, t_in = {}, 0
            # optionally start part-way along an Adam trajectory to cover later training phases
            warm = int(torch.randint(0, cfg.get("adam_warm_max", 0) + 1, (1,), generator=gen_for(seed, m, 3)))
            if warm:
                P = adam_warm(net, P, data, cfg, warm, seed + m)
        leaves = P
        for v in leaves.values():
            assert not v.requires_grad
        Pk, meta_loss, pre = P, 0.0, None
        with autocast():
            for k in range(K):
                x, y = data.batch("train", cfg["bs"], gen_for(seed, m * K + k))
                rec, out_, D, aux, info = hs.deltas(Pk, x, y)
                if aux.requires_grad:
                    hp = [q for q in hs.parameters() if q.requires_grad]
                    gs = torch.autograd.grad(aux, hp, allow_unused=True)
                    for p_, g_ in zip(hp, gs):
                        p_.grad = g_
                    hopt.step(); hopt.zero_grad(set_to_none=True)
                Pk = apply_edits(editor, net, Pk, rec, D, states, rowloss=row_loss(out_, y))
                xv, yv = data.batch("val", cfg["bs"], gen_for(seed, m * K + k, 11))
                lv = loss_fn(net.forward(Pk, xv), yv)
                if pre is None:
                    with torch.no_grad():
                        pre = loss_fn(net.forward(P, xv), yv).item()
                meta_loss = meta_loss + lv.log() / K
        mopt.zero_grad(set_to_none=True)
        if not torch.isfinite(meta_loss):
            skipped += 1; P = None; continue                       # diverged inner net: new trajectory
        meta_loss.backward()
        # the main network's weights must never receive a gradient
        for v in leaves.values():
            assert v.grad is None and not v.requires_grad
        gn = torch.nn.utils.clip_grad_norm_(editor.parameters(), cfg.get("clip", 1.0))
        if torch.isfinite(gn):
            mopt.step()
        else:
            skipped += 1
        P = {a: b.detach() for a, b in Pk.items()}
        detach_states(states)
        t_in += K
        l_start = pre if l_start is None else l_start
        if lv.item() > cfg.get("reset_factor", 3.0) * max(l_start, 1.0):
            P = None                                               # inner net diverged: restart it
        if (m + 1) % cfg.get("log_every", 20) == 0:
            log.log(event="meta", step=m + 1, meta_loss=meta_loss.item(), pre=pre, post=lv.item(),
                    traj=traj, t_inner=t_in, grad_norm=gn.item(), skipped=skipped)
        if (m + 1) % cfg.get("ckpt_every", 200) == 0 or m + 1 == n_meta:
            torch.save({"editor": editor.state_dict(), "mopt": mopt.state_dict(), "m": m + 1}, ck + ".tmp")
            os.replace(ck + ".tmp", ck)
            torch.save({"editor": editor.state_dict(), "cfg": cfg}, os.path.join(out, "editor.pt"))
    return {"meta_steps": n_meta}


def adam_warm(net, P, data, cfg, steps, seed):
    P = {k: v.clone().requires_grad_(True) for k, v in P.items()}
    opt = torch.optim.Adam(P.values(), lr=cfg.get("warm_lr", 1e-3))
    with flops.paused():
        for t in range(steps):
            x, y = data.batch("train", cfg["bs"], gen_for(seed, t, 21))
            with autocast():
                l = loss_fn(net.forward(P, x), y)
            opt.zero_grad(); l.backward(); opt.step()
    return {k: v.detach() for k, v in P.items()}


# ---------------------------------------------------------------------------------
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--out", default=None)
    ap.add_argument("overrides", nargs="*")
    a = ap.parse_args(argv)
    cfg = json.load(open(a.config))
    for kv in a.overrides:
        k, v = kv.split("=", 1)
        try:
            v = json.loads(v)
        except json.JSONDecodeError:
            pass
        cfg[k] = v
    cfg.setdefault("run_id", os.path.splitext(os.path.basename(a.config))[0])
    out = a.out or os.path.join("runs", cfg["run_id"])
    os.makedirs(out, exist_ok=True)
    json.dump(cfg, open(os.path.join(out, "config.json"), "w"), indent=1)
    torch.backends.cuda.matmul.allow_tf32 = True
    log = Logger(out, cfg)
    log.log(event="start", device=DEV, gpu=torch.cuda.get_device_name(0) if DEV == "cuda" else "cpu", cfg=cfg)
    mode = cfg["mode"]
    res = {"e0": lambda: run_e0(cfg, log), "train": lambda: run_train(cfg, log, out),
           "meta": lambda: run_meta(cfg, log, out)}[mode]()
    if mode == "meta" and cfg.get("eval"):
        # evaluate the frozen meta-trained editor on fresh networks (one sub-run per seed)
        ev = cfg["eval"]
        res["eval"] = {}
        for s in ev.get("seeds", [100, 101, 102]):
            ecfg = {**cfg, **ev.get("overrides", {}), "mode": "train", "method": "editor", "seed": s,
                    "editor_ckpt": os.path.join(out, "editor.pt"), "editor_train": None,
                    "run_id": f"{cfg['run_id']}_eval_s{s}"}
            eout = os.path.join(out, f"eval_s{s}")
            if os.path.exists(os.path.join(eout, "summary.json")):
                res["eval"][s] = json.load(open(os.path.join(eout, "summary.json")))
                continue
            os.makedirs(eout, exist_ok=True)
            elog = Logger(eout, ecfg)
            r = run_train(ecfg, elog, eout)
            json.dump({"cfg": ecfg, **r}, open(os.path.join(eout, "summary.json"), "w"), indent=1)
            res["eval"][s] = r
    json.dump({"cfg": cfg, **(res or {})}, open(os.path.join(out, "summary.json"), "w"), indent=1)
    log.log(event="done")
    open(os.path.join(out, "DONE"), "w").write("ok")


if __name__ == "__main__":
    main()
