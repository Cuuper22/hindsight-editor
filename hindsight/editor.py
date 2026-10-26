"""The editor E: hindsight factors (a, d) -> low-rank weight edit U V^T.

Per site, with rows n = examples (or examples x positions):
  A (N x in)  input activations       D (N x out)  hindsight delta at the output
  1. subsample at most `max_rows` rows   -> cost independent of batch size
  2. transform each factor elementwise with a tiny MLP shared across sites,
     conditioned on the site (layer id, relative depth, kind) and row statistics
  3. sketch the rows into r combinations S (N x r)
  4. U = D~^T S, V = A~^T S, edit = U V^T  (rank <= r), RMS-normalised
  5. momentum with learned decay, learned per-site step size

The editor never sees a gradient of the main network: D is a hindsight delta.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import flops

KINDS = ["linear", "conv", "embed", "qkv", "proj", "fc1", "fc2", "head"]


def site_kind(site):
    if site.kind != "linear":
        return site.kind
    tail = site.name.split(".")[-1]
    return tail if tail in KINDS else "linear"


def factors(site, a, d):
    """Flatten a site's (input, delta) into row matrices A (N x fan_in), D (N x out)."""
    if site.kind == "conv":
        k = 3
        A = F.unfold(a, k, padding=site.pad)                       # (B, in*k*k, HW)
        A = A.transpose(1, 2).reshape(-1, A.shape[1])
        D = d.flatten(2).transpose(1, 2).reshape(-1, d.shape[1])
        return A, D
    if site.kind == "embed":
        return a.reshape(-1), d.reshape(-1, d.shape[-1])           # A = token ids (implicit one-hot)
    return a.reshape(-1, a.shape[-1]), d.reshape(-1, d.shape[-1])


def _rms(x, dim=None, keepdim=False):
    return x.float().pow(2).mean(dim=dim, keepdim=keepdim).add(1e-12).sqrt() if dim is not None \
        else x.float().pow(2).mean().add(1e-12).sqrt()


class ElemMLP(nn.Module):
    """y = x + w2 . gelu(w1 * x + c_row), applied to every element; c_row is per-row context."""

    def __init__(self, hidden, ctx_dim):
        super().__init__()
        self.w1 = nn.Parameter(torch.randn(hidden) * 0.5)
        self.ctx = nn.Linear(ctx_dim, hidden)
        self.w2 = nn.Parameter(torch.zeros(hidden))

    def forward(self, x, ctx):
        h = F.gelu(x[..., None] * self.w1 + self.ctx(ctx)[:, None, :])
        flops.add("editor", 6 * x.numel() * self.w1.numel())
        return x + h @ self.w2


class Editor(nn.Module):
    def __init__(self, n_sites, rank=8, hidden=8, emb=16, max_rows=4096, use_a=True,
                 lr_init=3e-4, mom_init=0.9, row_loss=True, max_rel=0.05, use_id=True):
        super().__init__()
        self.max_rel, self.use_id = max_rel, use_id
        self.max_elems = 2 ** 25                                    # rows x width sampled per site
        self.rank, self.max_rows, self.use_a, self.row_loss = rank, max_rows, use_a, row_loss
        self.id_emb = nn.Embedding(max(n_sites, 64), emb)
        self.site_in = nn.Linear(1 + len(KINDS), emb)
        ctx = emb + 5                                               # site cond + row stats (+ row loss)
        self.tD = ElemMLP(hidden, ctx)
        self.tA = ElemMLP(hidden, ctx)
        self.score = nn.Linear(ctx + 2, 1)                          # per-row weight
        nn.init.zeros_(self.score.weight); nn.init.zeros_(self.score.bias)
        if row_loss:                                                # start from w_b = log(1 + loss_b)
            self.score.weight.data[0, ctx - 1] = 1.0
            self.score.bias.data.fill_(-0.5413)
        # per-site step size and momentum from (site cond, running stats)
        self.lr_head = nn.Linear(emb + 3, 2)
        nn.init.zeros_(self.lr_head.weight)
        self.lr_head.bias.data = torch.tensor([math.log(lr_init), math.log(lr_init)])
        self.mom_head = nn.Linear(emb, 1)
        nn.init.zeros_(self.mom_head.weight)
        self.mom_head.bias.data.fill_(math.log(mom_init / (1 - mom_init)))

    # ---- helpers -----------------------------------------------------------
    def cond(self, i, site, n_sites, device):
        k = torch.zeros(1 + len(KINDS), device=device)
        k[0] = i / max(n_sites - 1, 1)
        k[1 + KINDS.index(site_kind(site))] = 1.0
        # use_id=False: condition only on relative depth + layer kind, which transfer across model sizes
        return self.site_in(k) + (self.id_emb.weight[i] if self.use_id else 0.0)

    def row_weights(self, Dn, An, ctx):
        """Learned per-row weight w_b >= 0 (= 1 at init). Fed the row's blind loss via ctx,
        it can scale each example's contribution by how wrong the blind pass was."""
        feats = torch.cat([ctx, Dn.pow(2).mean(1, keepdim=True).add(1e-12).log(),
                           An.pow(2).mean(1, keepdim=True).add(1e-12).log()], 1)
        return F.softplus(self.score(feats)[:, 0] + 0.5413)

    def low_rank(self, Dw, An, A_idx, W_shape, gen=None):
        """Rank-r edit Q (Dw Q)^T An / N, Q = range of Dw^T (An Om): a randomized range finder.

        It projects the exact weighted batch update Dw^T An / N onto its r dominant output
        directions (no cross-example noise), at O(N r (in + out) + r in out) cost.
        """
        N, r = Dw.shape[0], self.rank
        n_in = W_shape[0] if A_idx is not None else An.shape[1]
        Om = torch.randn(n_in, r, generator=gen).to(Dw.device)
        AOm = Om[A_idx] if A_idx is not None else An @ Om          # (N, r)
        Y = Dw.t() @ AOm                                            # (out, r)
        Q = torch.linalg.qr(Y + 1e-6 * torch.randn_like(Y))[0]     # (out, r) orthonormal
        DQ = Dw @ Q                                                 # (N, r)
        if A_idx is not None:                                       # embedding: A is one-hot
            B = torch.zeros(W_shape[0], r, device=Dw.device, dtype=DQ.dtype).index_add_(0, A_idx, DQ)
            flops.add("editor", 4 * N * r * Dw.shape[1] + 2 * r * W_shape[0] * W_shape[1])
            return B @ Q.t() / N                                    # (vocab, d)
        B = DQ.t() @ An                                             # (r, in)
        flops.add("editor", 2 * N * r * (2 * Dw.shape[1] + 2 * An.shape[1]) + 2 * r * Dw.shape[1] * An.shape[1])
        return Q @ B / N                                            # (out, in)

    # ---- the edit ------------------------------------------------------------
    def edit(self, i, site, a, d, W, state, n_sites, gen=None, rowloss=None):
        """Return (dW, db or None). `state` is a per-site dict (momentum, EMA stats).

        rowloss: per-example (or per-token) blind-pass loss -- a forward-pass quantity,
        no gradient -- so the editor can tell which examples are already right.
        """
        A, D = factors(site, a, d)
        N = D.shape[0]
        L = None
        if rowloss is not None and self.row_loss:
            L = rowloss.reshape(-1).float()
            L = L.repeat_interleave(N // L.numel())                 # conv: one row per position
        cap = max(64, min(self.max_rows, self.max_elems // max(D.shape[1], 1)))   # wide sites (LM head)
        if N > cap:
            idx = torch.randperm(N, device=D.device, generator=gen)[:cap]
            A, D, N = A[idx], D[idx], cap
            L = L[idx] if L is not None else None
        D = D.float()
        c = self.cond(i, site, n_sites, D.device)
        rD = _rms(D, 1, True)
        if site.kind == "embed":
            rA = torch.ones_like(rD)
        else:
            A = A.float()
            rA = _rms(A, 1, True)
        stats = torch.cat([rD.log(), rA.log() if self.use_a else torch.zeros_like(rA),
                           (rD / rD.mean()).log(), (rA / rA.mean()).log() if self.use_a else torch.zeros_like(rA),
                           (L[:, None] + 1e-4).log() if L is not None else torch.zeros_like(rD)], 1)
        ctx = torch.cat([c.expand(N, -1), stats], 1)
        Dn = self.tD(D / rD, ctx)
        if site.kind == "embed":
            An = None
        else:
            An = self.tA(A / rA, ctx) if self.use_a else A / rA
        w = self.row_weights(Dn, An if An is not None else Dn.new_zeros(N, 1), ctx)
        Dw = Dn * w[:, None]
        dW = self.low_rank(Dw, An, A if site.kind == "embed" else None, W.shape, gen).reshape(W.shape)
        # not RMS-normalised: the size of the edit (via the learned row weights) is the error signal
        # running stats -> step size; momentum with learned decay
        ema = state.get("ema_rD")
        cur = rD.mean().log().detach()
        state["ema_rD"] = cur if ema is None else 0.9 * ema + 0.1 * cur
        feat = torch.stack([cur - state["ema_rD"], _rms(W).log().detach(), torch.tensor(float(state.get("t", 0)) / 1000, device=D.device)])
        lr_w, lr_b = self.lr_head(torch.cat([c, feat])).clamp(max=0.0).exp()
        mu = torch.sigmoid(self.mom_head(c))[0]
        m = state.get("m")
        m = (1 - mu) * dW if m is None else mu * m + (1 - mu) * dW
        state["m"] = m
        state["t"] = state.get("t", 0) + 1
        step = lr_w * m
        # trust region: one edit moves a weight matrix by at most `max_rel` of its RMS
        step = step * (self.max_rel * _rms(W).detach() / (_rms(step) + 1e-12)).clamp(max=1.0)
        db = None
        if site.b is not None:
            db = lr_b * Dw.mean(0)                                   # sum_b w_b d_b / N
        return step, db


def apply_edits(editor, net, P, rec, D, states, gen=None, detach=False, rowloss=None):
    """Edit every site; returns the new param dict (functional; graph kept unless detach)."""
    newP = dict(P)
    n = len(net.sites)
    dev = next(iter(P.values())).device.type
    with torch.autocast(dev, enabled=False):                        # editor math in fp32
        return _apply(editor, net, P, rec, D, states, gen, detach, rowloss, newP, n)


def _apply(editor, net, P, rec, D, states, gen, detach, rowloss, newP, n):
    for i, s in enumerate(net.sites):
        dW, db = editor.edit(i, s, rec.a[s.name], D[s.name], P[s.w], states.setdefault(s.name, {}), n, gen,
                             rowloss=rowloss)
        if detach:
            dW = dW.detach(); db = db.detach() if db is not None else None
        newP[s.w] = P[s.w] + dW.to(P[s.w].dtype)
        if db is not None:
            newP[s.b] = P[s.b] + db.to(P[s.b].dtype)
    return newP
