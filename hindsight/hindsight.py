"""Hindsight channel, readouts and the hindsight delta.

Blind pass   : net(P, x)
Sighted pass : net(P, x, inj=channel(code(answer)))      (same weights, run in parallel)
Delta        : d = stopgrad(z_sighted) - z_blind          at every edit site

The channel and readouts are the only things trained by the readout loss, and
that gradient is kept local: the sighted pass sees detached weights and detaches
between layers, so the readout at point i only trains injection C_i and readout R_i.

Delta sources (for ablations/baselines):
  hindsight : parallel sighted pass, answer = centered one-hot / future summary
  ftp_seq   : sequential sighted pass, answer = onehot(y) - softmax(blind)  (FTP-style)
  dfa       : fixed random projection of (onehot(y) - p) at every point    (random feedback)
  backprop  : -dL/dz (oracle; used for imitation labels and cosine checks only)
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import flops
from .nets import Rec


def _scale_to(v, beta, dim):
    """Rescale per example so the injection has RMS = beta over its feature dim."""
    n = v.pow(2).mean(dim=dim, keepdim=True).add(1e-12).sqrt()
    return v * (beta / n)


class FutureSummary(nn.Module):
    """LM hindsight code: discounted sum of embeddings of tokens t+1..t+k, then a
    width-w bottleneck with an optional VIB (KL to N(0, I)) information penalty."""

    def __init__(self, vocab, k, width, emb=64, gamma=0.7, vib=True):
        super().__init__()
        self.k, self.vib = k, vib
        self.table = nn.Embedding(vocab, emb)
        self.register_buffer("w", gamma ** torch.arange(k, dtype=torch.float32))
        self.mu = nn.Linear(emb, width)
        self.logvar = nn.Linear(emb, width) if vib else None

    def forward(self, idx, targets):
        # targets[:, t] = idx[:, t+1]; future window of position t = targets[:, t : t+k]
        B, T = targets.shape
        e = self.table(targets)                                   # (B,T,emb)
        s = torch.zeros_like(e)
        for j in range(self.k):                                    # positions past T get nothing
            s[:, : T - j] += self.w[j] * e[:, j:]
        flops.add("hint", 2 * B * T * e.shape[-1] * (self.k + self.mu.out_features * (2 if self.vib else 1)))
        mu = self.mu(s)
        if not self.vib:
            return mu, mu.new_zeros(())
        logvar = self.logvar(s).clamp(-8, 8)
        z = mu + torch.randn_like(mu) * (0.5 * logvar).exp() if self.training else mu
        kl = 0.5 * (mu.pow(2) + logvar.exp() - 1 - logvar).sum(-1).mean()
        return z, kl

    def predicted(self, targets, p_next):
        """The code the blind pass expects: same future window, but token t+1 replaced by its
        predicted distribution p_next (B,T,V). Used by the linear correction."""
        B, T = targets.shape
        e = self.table(targets)
        s = torch.zeros_like(e)
        for j in range(1, self.k):
            s[:, : T - j] += self.w[j] * e[:, j:]
        s = s + self.w[0] * (p_next.to(e.dtype) @ self.table.weight)
        flops.add("hint", 2 * B * T * self.table.weight.numel())
        return self.mu(s)


class Hindsight(nn.Module):
    def __init__(self, net, cfg, n_classes=None, vocab=None):
        super().__init__()
        self.net, self.cfg = net, cfg
        self.source = cfg.get("delta_source", "hindsight")
        self.beta = cfg.get("beta", 0.5)
        self.lm = vocab is not None
        dims = net.point_dims
        if self.lm:
            w = cfg.get("bottleneck", 16)
            self.summary = FutureSummary(vocab, cfg.get("k_future", 8), w, vib=cfg.get("vib", True))
            self.kl_coef = cfg.get("kl_coef", 1e-2) if cfg.get("vib", True) else 0.0
            n_code, n_ans = w, vocab
        else:
            n_code = n_ans = n_classes
        self.n_ans = n_ans
        self.C = nn.ParameterList([nn.Parameter(torch.randn(n_code, d) / math.sqrt(n_code)) for d in dims])
        # readout at the last point of a classifier is the logits themselves
        last_is_logits = (not self.lm) and dims[-1] == n_ans
        # large vocabularies: readout = small d x d map + the model's own (detached) head, scored with a
        # sampled softmax over the batch's targets + random negatives, on a subset of points/positions
        self.big_vocab = self.lm and n_ans > cfg.get("full_readout_max_vocab", 4096)
        if self.big_vocab:
            # a readout at every point (each injection C_i is trained only by its own local readout),
            # each scoring a small random fraction of positions
            self.r_points = list(range(len(dims)))
            self.R = nn.ModuleList([nn.Linear(dims[i], dims[i], bias=False) for i in self.r_points])
            for r in self.R:
                nn.init.eye_(r.weight)
        else:
            self.r_points = list(range(len(dims[:-1] if last_is_logits else dims)))
            self.R = nn.ModuleList([nn.Linear(d, n_ans) for d in (dims[:-1] if last_is_logits else dims)])
        self.last_is_logits = last_is_logits
        if last_is_logits:
            # the answer injected into the logits is the answer itself (fixed identity map);
            # a learned/random map there rotates the error and pushes logits the wrong way
            self.C[-1].data = torch.eye(n_ans)
            self.C[-1].requires_grad_(False)
        self.train_channel = cfg.get("readout_loss", True)

    # ---- codes / injection -------------------------------------------------
    def code(self, y, blind_logits=None, idx=None):
        if self.lm:
            return self.summary(idx, y)
        onehot = F.one_hot(y, self.n_ans).float()
        if self.source == "ftp_seq":
            return onehot - blind_logits.detach().float().softmax(-1), onehot.new_zeros(())
        return onehot - 1.0 / self.n_ans, onehot.new_zeros(())

    def injector(self, code, C=None):
        C = self.C if C is None else C

        def inj(i, h):
            # fixed-gain linear map (C rescaled to unit RMS), so an injection shrinks with its code:
            # for ftp_seq the code is (y - p) and the delta vanishes once the net is right
            Ci = C[i]
            v = code @ (Ci * (self.beta * math.sqrt(self.n_ans) / Ci.pow(2).sum(1).mean().add(1e-12).sqrt()))
            flops.add("hint", 2 * v.numel() * code.shape[-1])
            return v[:, :, None, None] if h.dim() == 4 else v      # conv: per-channel, broadcast over space
        return inj

    # ---- readouts ----------------------------------------------------------
    def readout_loss(self, states, y, head=None):
        if self.big_vocab:
            return self._sampled_readout(states, y, head)
        losses = []
        for i, h in enumerate(states):
            if h.dim() == 4:
                h = h.mean((2, 3))
            if i < len(self.R):
                logits = self.R[i](h)
                flops.add("readout", 3 * 2 * h[..., 0].numel() * h.shape[-1] * self.n_ans)
            else:
                logits = h
            losses.append(F.cross_entropy(logits.reshape(-1, self.n_ans).float(), y.reshape(-1)))
        return torch.stack(losses)

    def _sampled_readout(self, states, y, head):
        frac, n_neg = self.cfg.get("readout_rows", 0.1), self.cfg.get("readout_negatives", 4096)
        yy = y.reshape(-1)
        n = yy.numel()
        sel = torch.randperm(n, device=yy.device)[: max(1, int(frac * n))]
        cand = torch.unique(torch.cat([yy[sel], torch.randint(self.n_ans, (n_neg,), device=yy.device)]))
        tgt = torch.searchsorted(cand, yy[sel])
        E = head.detach()[cand]                                    # (C, d): the head's current token codes
        losses = []
        for r, i in zip(self.R, self.r_points):
            h = states[i].reshape(n, -1)[sel]
            logits = r(F.layer_norm(h, h.shape[-1:])) @ E.t()
            flops.add("readout", 3 * 2 * h.shape[0] * h.shape[1] * (h.shape[1] + E.shape[0]))
            losses.append(F.cross_entropy(logits.float(), tgt))
        return torch.stack(losses)

    # ---- the hindsight delta ----------------------------------------------
    def deltas(self, P, x, y, with_backprop=False):
        """Returns (rec_blind, logits_blind, {site: d}, aux_loss, info).

        Graph from P to rec_blind is kept (so a meta-unroll can differentiate it);
        sighted states are stop-gradded w.r.t. P by construction.
        """
        net = self.net
        rb = Rec()
        blind = net.forward(P, x, rec=rb)
        info = {}
        aux = blind.new_zeros(())
        if self.source == "backprop" or with_backprop:
            bp = backprop_error(net, P, x, y)
            info["bp"] = bp
            if self.source == "backprop":
                return rb, blind, {k: -v for k, v in bp.items()}, aux, info
        if self.source == "dfa":
            # standard DFA: true error at the output, ReLU-gated random projections below it.
            # Feedback matrices are per edit site (shape from the site's output), made lazily.
            e = F.one_hot(y, self.n_ans).float() - blind.detach().float().softmax(-1)
            D = {}
            last = net.sites[-1].name
            for i, s in enumerate(net.sites):
                z = rb.z[s.name].detach()
                if s.name == last:
                    D[s.name] = e.to(z.dtype) if z.dim() == e.dim() else e
                    continue
                width = z.shape[1] if z.dim() == 4 else z.shape[-1]
                Bs = self.__dict__.setdefault("_Bs", {})          # plain dict: deterministic, not checkpointed
                if s.name not in Bs or Bs[s.name].shape[1] != width:
                    g = torch.Generator().manual_seed(1000 + i)
                    Bs[s.name] = (torch.randn(self.n_ans, width, generator=g) / math.sqrt(self.n_ans)).to(z.device)
                v = e @ Bs[s.name].to(e.dtype)
                v = v[:, :, None, None].expand_as(z) if z.dim() == 4 else v
                D[s.name] = (v * (z > 0).float()) if self.cfg.get("dfa_gate", not self.lm) else v.to(z.dtype)
            return rb, blind, D, aux, info
        code, kl = self.code(y, blind_logits=blind, idx=x)
        Pd = {k: v.detach() for k, v in P.items()}
        # slow copy: the sighted pass (and its blind reference) run on an EMA of the weights, so a
        # layer is not trained toward its own *current* amplified response (the Hebbian loop)
        tau = self.cfg.get("ema", 0.0)
        ref, Ps = rb, Pd
        if tau > 0:
            if getattr(self, "P_ema", None) is None or set(self.P_ema) != set(Pd):
                self.P_ema = {k: v.clone() for k, v in Pd.items()}
            Ps = self.P_ema
            ref = Rec()
            with torch.no_grad():
                net.forward(Ps, x, rec=ref)
        alpha = self.cfg.get("prop", 1.0)
        kw = {"ref": ref, "alpha": alpha} if self.lm else {}
        rs = Rec()
        with torch.enable_grad():
            net.forward(Ps, x, inj=self.injector(code), rec=rs, detach_between=True, **kw)
            rl = self.readout_loss(rs.h, y, head=Ps.get("head"))
        info["readout"] = rl.detach()
        if self.train_channel:
            aux = rl.sum() + (self.kl_coef * kl if self.lm else 0.0)
        info["kl"] = kl.detach()
        D = {s.name: rs.z[s.name].detach() - ref.z[s.name].detach() for s in net.sites} if tau > 0 else \
            {s.name: rs.z[s.name].detach() - rb.z[s.name] for s in net.sites}
        if tau > 0:
            with torch.no_grad():
                for k, v in Pd.items():
                    self.P_ema[k].mul_(tau).add_(v, alpha=1 - tau)
        alpha = self.cfg.get("prop", 1.0)
        if alpha != 1.0 and not self.lm:
            # d = inj + alpha * (propagated part). alpha < 1 damps the Hebbian loop in which a
            # layer is trained toward its own amplified response to the injected answer.
            with torch.no_grad():
                inj = self.injector(code)
                for i, s in enumerate(net.sites):
                    local = inj(i, rb.z[s.name]).to(D[s.name].dtype).expand_as(D[s.name])
                    D[s.name] = local + alpha * (D[s.name] - local)
        if self.source == "hindsight_lc" and not self.lm:
            D = self.linear_correct(D, code, blind)
        if self.source == "hindsight_lc" and self.lm:
            D = self.linear_correct_lm(D, code, blind, y)
        if self.cfg.get("gate", False) and not self.lm:            # ReLU nets: only active units move
            last = net.sites[-1].name
            D = {k: v if k == last else v * (rb.z[k].detach() > 0) for k, v in D.items()}
        return rb, blind, D, aux, info

    def linear_correct_lm(self, D, code, blind, y, ridge=1e-2):
        """LM version: rows are tokens; the code is the bottleneck future summary z_t. Regress each
        site's delta on z, subtract the part explained by the code the blind pass expected
        (summary with token t+1 replaced by p_t). The head gets the exact next-token error
        beta * (onehot(y) - p): the answer injected at the output is the answer itself."""
        p = blind.detach().float().softmax(-1)                      # (B,T,V)
        with torch.no_grad(), torch.autocast(p.device.type, enabled=False):
            q = self.summary.predicted(y, p).float()               # (B,T,w)
            Cz = code.detach().float().reshape(-1, code.shape[-1])
            if not (torch.isfinite(Cz).all() and torch.isfinite(q).all()):
                return D                                           # diverged net: nothing sensible to correct
            Q = q.reshape(-1, q.shape[-1])
            w = Cz.shape[1]
            G = Cz.t() @ Cz + ridge * Cz.shape[0] * torch.eye(w, device=Cz.device)
            out = {}
            for name, d in D.items():
                if name == "head":                                 # beta * (onehot(y) - p), built in place
                    h = p.mul(-self.beta).reshape(-1, self.n_ans)
                    h[torch.arange(h.shape[0], device=h.device), y.reshape(-1)] += self.beta
                    out[name] = h.reshape(d.shape).to(d.dtype)
                    continue
                Df = d.reshape(-1, d.shape[-1]).float()
                M = torch.linalg.solve(G, Cz.t() @ Df)              # (w, F)
                flops.add("hint", 4 * Df.numel() * w)
                out[name] = (Df - Q @ M).reshape(d.shape).to(d.dtype)
        return out

    def linear_correct(self, D, code, blind, ridge=1e-2):
        """Parallel hindsight with linear correction (`hindsight_lc`).

        Per batch, regress each site's delta on the injected code, d_b ~ code_b M (ridge least
        squares, K x F), then remove what the blind pass already predicts:
            d~_b = d_b - (p_b - 1/K) M  ~=  (y_b - p_b) M + nonlinear residual.
        It vanishes as p -> y, like DFA's B(y - p), but M is the network's own measured response
        to the answer. Uses only forward quantities (p from the parallel blind pass).
        """
        K = self.n_ans
        C = code.float()                                           # centered one-hot (N, K)
        q = blind.detach().float().softmax(-1) - 1.0 / K           # predicted code
        G = C.t() @ C + ridge * C.shape[0] * torch.eye(K, device=C.device)
        out = {}
        with torch.autocast(C.device.type, enabled=False):         # fp32 solve
            for name, d in D.items():
                Df = d.reshape(d.shape[0], -1).float()
                M = torch.linalg.solve(G, C.t() @ Df)              # (K, F)
                flops.add("hint", 4 * Df.numel() * K)
                out[name] = (Df - q @ M).reshape(d.shape).to(d.dtype)
        return out


def loss_fn(logits, y):
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(), y.reshape(-1))


def backprop_error(net, P, x, y):
    """dL/dz at every site output, via a real backward pass (labels/diagnostics only)."""
    Pd = {k: v.detach().requires_grad_(True) for k, v in P.items()}  # fresh leaves, discarded after
    rec = Rec()
    with torch.enable_grad(), flops.paused():
        out = net.forward(Pd, x, rec=rec)
        zs = [rec.z[s.name] for s in net.sites]
        g = torch.autograd.grad(loss_fn(out, y), zs)
    return {s.name: gi.detach() for s, gi in zip(net.sites, g)}


def per_example_cos(a, b):
    """Cosine per example (per token for 3-D LM tensors), flattening the rest."""
    if a.dim() == 3:
        a, b = a.reshape(-1, a.shape[-1]), b.reshape(-1, b.shape[-1])
    a, b = a.flatten(1).float(), b.flatten(1).float()
    return F.cosine_similarity(a, b, dim=1, eps=1e-12)
