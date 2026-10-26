"""Functional networks whose weights live in a plain dict, so they can be edited
in place (no grad) or carried as non-leaf tensors through a meta-unroll.

Every weight matrix is an *edit site*. A forward pass can:
  * add a hindsight injection at each *injection point* (`inj(i, h) -> tensor`),
  * record per-site (input activation a, output pre-activation z),
  * return the per-point states h_i (what readouts look at).

All matmul FLOPs go through `flops.add` so methods are costed identically.
"""
import math
import torch
import torch.nn.functional as F

from . import flops


def _lin(x, w, b=None, tag="fwd"):
    flops.add(tag, 2 * x[..., 0].numel() * w.shape[0] * w.shape[1])
    return F.linear(x, w, b)


def _conv(x, w, b, pad, tag="fwd"):
    out = F.conv2d(x, w, b, padding=pad)
    flops.add(tag, 2 * out.numel() * w.shape[1] * w.shape[2] * w.shape[3])
    return out


class Site:
    def __init__(self, name, w, b=None, kind="linear", pad=0):
        self.name, self.w, self.b, self.kind, self.pad = name, w, b, kind, pad


class Rec:
    """What one forward pass exposes: per-site (a, z) and per-point states."""

    def __init__(self):
        self.a, self.z, self.h = {}, {}, []


def _add_inj(inj, i, h):
    return h if inj is None else h + inj(i, h)


class MLP:
    def __init__(self, in_dim=784, hidden=(256, 256), n_out=10, act="relu"):
        self.dims = [in_dim, *hidden, n_out]
        self.act = {"relu": F.relu, "tanh": torch.tanh}[act]
        self.sites = [Site(f"l{i}", f"l{i}.w", f"l{i}.b") for i in range(len(self.dims) - 1)]
        self.point_dims = self.dims[1:]

    def init(self, gen, device):
        P = {}
        for i, (di, do) in enumerate(zip(self.dims[:-1], self.dims[1:])):
            P[f"l{i}.w"] = (torch.randn(do, di, generator=gen) * math.sqrt(2.0 / di)).to(device)
            P[f"l{i}.b"] = torch.zeros(do, device=device)
        return P

    def forward(self, P, x, inj=None, rec=None, detach_between=False):
        a = x.flatten(1)
        n = len(self.sites)
        for i, s in enumerate(self.sites):
            z = _add_inj(inj, i, _lin(a, P[s.w], P[s.b]))
            if rec is not None:
                rec.a[s.name], rec.z[s.name] = a, z
                rec.h.append(z)
            a = self.act(z) if i < n - 1 else z
            if detach_between:
                a = a.detach()
        return a


class CNN:
    """conv(32)-pool-conv(64)-pool-conv(128)-pool-fc(256)-fc(10) for 32x32 inputs."""

    def __init__(self, in_ch=3, chans=(32, 64, 128), fc=256, n_out=10, side=32):
        self.chans = [in_ch, *chans]
        self.flat = chans[-1] * (side // 2 ** len(chans)) ** 2
        self.sites = [Site(f"c{i}", f"c{i}.w", f"c{i}.b", "conv", 1) for i in range(len(chans))]
        self.sites += [Site("f0", "f0.w", "f0.b"), Site("f1", "f1.w", "f1.b")]
        self.fc, self.n_out = fc, n_out
        self.point_dims = [*chans, fc, n_out]

    def init(self, gen, device):
        P = {}
        for i, (ci, co) in enumerate(zip(self.chans[:-1], self.chans[1:])):
            P[f"c{i}.w"] = (torch.randn(co, ci, 3, 3, generator=gen) * math.sqrt(2.0 / (ci * 9))).to(device)
            P[f"c{i}.b"] = torch.zeros(co, device=device)
        for k, (di, do) in {"f0": (self.flat, self.fc), "f1": (self.fc, self.n_out)}.items():
            P[f"{k}.w"] = (torch.randn(do, di, generator=gen) * math.sqrt(2.0 / di)).to(device)
            P[f"{k}.b"] = torch.zeros(do, device=device)
        return P

    def forward(self, P, x, inj=None, rec=None, detach_between=False):
        a = x
        for i, s in enumerate(self.sites):
            if s.kind == "conv":
                z = _add_inj(inj, i, _conv(a, P[s.w], P[s.b], s.pad))
                nxt = F.max_pool2d(F.relu(z), 2)
            else:
                if a.dim() > 2:
                    a = a.flatten(1)
                z = _add_inj(inj, i, _lin(a, P[s.w], P[s.b]))
                nxt = F.relu(z) if i < len(self.sites) - 1 else z
            if rec is not None:
                rec.a[s.name], rec.z[s.name] = a, z
                rec.h.append(z)
            a = nxt.detach() if detach_between else nxt
        return a


class GPT:
    """Pre-LN decoder, no biases, non-affine LayerNorm (so every parameter is a matrix site)."""

    def __init__(self, vocab, ctx, d=384, n_layer=6, n_head=6):
        self.vocab, self.ctx, self.d, self.L, self.H = vocab, ctx, d, n_layer, n_head
        self.sites = [Site("emb", "emb", None, "embed")]
        for i in range(n_layer):
            self.sites += [Site(f"b{i}.{k}", f"b{i}.{k}") for k in ("qkv", "proj", "fc1", "fc2")]
        self.sites.append(Site("head", "head"))
        # injection points: residual stream entering each block, and before the head
        self.point_dims = [d] * (n_layer + 1)

    def n_params(self):
        return self.vocab * self.d * 2 + self.ctx * self.d + self.L * 12 * self.d ** 2

    def init(self, gen, device):
        d, std = self.d, 0.02
        r = lambda *s: torch.randn(*s, generator=gen) * std
        P = {"emb": r(self.vocab, d), "pos": r(self.ctx, d), "head": r(self.vocab, d)}
        for i in range(self.L):
            P[f"b{i}.qkv"] = r(3 * d, d)
            P[f"b{i}.proj"] = r(d, d) / math.sqrt(2 * self.L)
            P[f"b{i}.fc1"] = r(4 * d, d)
            P[f"b{i}.fc2"] = r(d, 4 * d) / math.sqrt(2 * self.L)
        return {k: v.to(device) for k, v in P.items()}

    def forward(self, P, idx, inj=None, rec=None, detach_between=False):
        B, T = idx.shape
        d, H = self.d, self.H
        x = P["emb"][idx] + P["pos"][:T]
        x = _add_inj(inj, 0, x)
        if rec is not None:
            rec.a["emb"], rec.z["emb"] = idx, x
        site = lambda name, a, w: self._site(rec, name, a, w)
        for i in range(self.L):
            if rec is not None:
                rec.h.append(x)
            h = F.layer_norm(x, (d,))
            qkv = site(f"b{i}.qkv", h, P[f"b{i}.qkv"])
            q, k, v = qkv.view(B, T, 3, H, d // H).permute(2, 0, 3, 1, 4)
            flops.add("fwd", 4 * B * T * T * d)
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True).transpose(1, 2).reshape(B, T, d)
            x = x + site(f"b{i}.proj", y, P[f"b{i}.proj"])
            h = F.layer_norm(x, (d,))
            u = site(f"b{i}.fc1", h, P[f"b{i}.fc1"])
            x = x + site(f"b{i}.fc2", F.gelu(u), P[f"b{i}.fc2"])
            if detach_between:                                     # cut BEFORE the injection, so the
                x = x.detach()                                     # readout at point i+1 trains C_{i+1}
            x = _add_inj(inj, i + 1, x)
        if rec is not None:
            rec.h.append(x)
        return site("head", F.layer_norm(x, (d,)), P["head"])

    @staticmethod
    def _site(rec, name, a, w):
        z = _lin(a, w)
        if rec is not None:
            rec.a[name], rec.z[name] = a, z
        return z


def build(cfg):
    kind = cfg["net"]
    if kind == "mlp":
        return MLP(hidden=tuple(cfg.get("hidden", (256, 256))), act=cfg.get("act", "relu"))
    if kind == "cnn":
        return CNN()
    if kind == "gpt":
        return GPT(cfg["vocab"], cfg["ctx"], cfg["d"], cfg["n_layer"], cfg["n_head"])
    raise ValueError(kind)
