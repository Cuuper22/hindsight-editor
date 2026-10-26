"""CPU unit tests for the core pieces. Run: python -m pytest tests -q"""
import torch, pytest
from hindsight import run as R
from hindsight.nets import MLP, CNN, GPT, Rec
from hindsight.hindsight import Hindsight, loss_fn
from hindsight.editor import Editor, apply_edits

torch.manual_seed(0)
G = lambda: torch.Generator().manual_seed(0)


def setups():
    yield MLP(), torch.randn(16, 1, 28, 28), torch.randint(10, (16,)), dict(n_classes=10)
    yield CNN(), torch.randn(4, 3, 32, 32), torch.randint(10, (4,)), dict(n_classes=10)
    g = GPT(vocab=32, ctx=12, d=24, n_layer=2, n_head=2)
    yield g, torch.randint(32, (3, 12)), torch.randint(32, (3, 12)), dict(vocab=32)


@pytest.mark.parametrize("i", range(3))
def test_delta_shapes_and_stopgrad(i):
    net, x, y, kw = list(setups())[i]
    P = net.init(G(), "cpu")
    for v in P.values():
        v.requires_grad_(True)
    hs = Hindsight(net, {"beta": 0.5, "bottleneck": 4}, **kw)
    rec, out, D, aux, _ = hs.deltas(P, x, y)
    for s in net.sites:
        assert D[s.name].shape == rec.z[s.name].shape
    # stop-gradient: d depends on P only through the blind pass, so d(sum d)/dP == -d(sum z_blind)/dP
    s = net.sites[-1]
    gd = torch.autograd.grad(D[s.name].sum(), P[s.w], retain_graph=True)[0]
    gz = torch.autograd.grad(rec.z[s.name].sum(), P[s.w], retain_graph=True)[0]
    assert torch.allclose(gd, -gz)
    # readout loss is local: it trains the channel/readouts, never the main weights
    gs = torch.autograd.grad(aux, list(P.values()), allow_unused=True)
    assert all(g is None for g in gs)
    assert any(p.grad is None for p in hs.parameters())
    aux.backward()
    assert hs.C[0].grad is not None and hs.C[0].grad.abs().sum() > 0


@pytest.mark.parametrize("i", range(3))
def test_editor_edit_application(i):
    net, x, y, kw = list(setups())[i]
    P = net.init(G(), "cpu")
    hs = Hindsight(net, {"beta": 0.5, "bottleneck": 4}, **kw)
    ed = Editor(len(net.sites), rank=4, max_rows=64)
    rec, out, D, _, _ = hs.deltas(P, x, y)
    newP = apply_edits(ed, net, P, rec, D, {}, rowloss=loss_fn(out, y).expand(y.shape))
    for i, s in enumerate(net.sites):
        assert newP[s.w].shape == P[s.w].shape and torch.isfinite(newP[s.w]).all()
        dW, _ = ed.edit(i, s, rec.a[s.name], D[s.name], P[s.w], {}, len(net.sites))   # the edit itself
        assert dW.abs().sum() > 0
        assert torch.linalg.matrix_rank(dW.reshape(dW.shape[0], -1).double(), rtol=1e-4) <= 4  # float32 noise


class FakeImages:
    n_classes = 10
    def batch(self, split, bs, gen):
        return torch.randn(bs, 1, 28, 28, generator=gen), torch.randint(10, (bs,), generator=gen)


def test_phase1_no_grad_reaches_main_weights(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "DEV", "cpu")
    monkeypatch.setattr(R, "make_data", lambda cfg: FakeImages())
    monkeypatch.setattr(R, "ImageData", FakeImages)
    cfg = {"mode": "meta", "net": "mlp", "hidden": [32, 32], "seed": 0, "bs": 8, "meta_steps": 3,
           "unroll": 2, "inner_steps": 4, "editor": {"rank": 2, "max_rows": 8}, "log_every": 1, "ckpt_every": 100}
    log = R.Logger(str(tmp_path), cfg)
    R.run_meta(cfg, log, str(tmp_path))          # asserts inside: leaves never require/receive grad
    assert (tmp_path / "editor.pt").exists()


@pytest.mark.parametrize("i", range(3))
def test_every_trainable_channel_gets_readout_gradient(i):
    net, x, y, kw = list(setups())[i]
    P = net.init(G(), "cpu")
    hs = Hindsight(net, {"beta": 0.5, "bottleneck": 4, "vib": False}, **kw)
    _, _, _, aux, _ = hs.deltas(P, x, y)
    aux.backward()
    for j, c in enumerate(hs.C):
        if c.requires_grad:
            assert c.grad is not None and c.grad.abs().sum() > 0, f"channel {j} never trained"
