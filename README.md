# Hindsight Editor

A learned, gradient-free training rule. The network runs twice on the same input: a normal
**blind** pass, and a **sighted** pass where the answer is fed in through a small hindsight channel.
The per-layer difference (the *hindsight delta*) goes to a small learned **editor**, which turns it
into a low-rank weight edit. No backward pass touches the main network.

Results: [RESULTS.md](RESULTS.md). Compute ledger: [runs/ledger.csv](runs/ledger.csv).

## Layout

| path | what |
|---|---|
| `hindsight/nets.py` | functional MLP, CNN, GPT; every weight matrix is an *edit site* |
| `hindsight/hindsight.py` | hindsight channel, readouts, future-token bottleneck (LM), delta sources, linear correction |
| `hindsight/editor.py` | the editor: elementwise factor transforms, per-example weights, randomized range-finder low-rank edit, momentum, trust region |
| `hindsight/run.py` | entry point: `e0` (cosine check), `train` (adam / raw rule / editor), `meta` (Phase 1, optional frozen eval) |
| `hindsight/flops.py` | analytic FLOP accounting (every matmul reports here) |
| `experiments/*.json` | base configs; `experiments/waves/*.json` = the exact job lists that were run |
| `launch/` | Colab CLI wrapper (`launch.py`), on-VM supervisor, Drive/HF helpers |
| `analysis/` | tables (`table.py`, `meta_table.py`), central measurement (`central.py`), diagnostics (`diag/`) |
| `tests/` | CPU unit tests |

## Rerun

Requirements: Python 3.12, PyTorch 2.6+, `numpy`, `huggingface_hub`, `pandas`, `pillow`.

```bash
python -m pytest tests -q
```

```bash
python -m hindsight.run experiments/e0_mnist.json --out runs/e0_mnist_s0 seed=0
```

Any config key can be overridden as `key=json_value`, e.g. `method="editor" delta_source="hindsight_lc"`.

### On Colab (what produced the results)

1. Install and authenticate the Colab CLI (`colab sessions` must work), and log in to Hugging Face
   (`~/.cache/huggingface/token`, write access). Set `HS_REPO` to a private dataset repo you own
   (default `cuuper22/hindsight-runs`).
2. Launch a wave (one VM, many jobs packed on one GPU):

```bash
python launch/launch.py up experiments/waves/e1base.json
```

3. Check, collect and record it:

```bash
python launch/launch.py status e1base
```

```bash
python launch/launch.py collect e1base
```

```bash
python launch/launch.py close e1base
```

What happens on `up`: the code and job list are pushed to the HF repo *before* a VM exists; the VM
is created, receives the HF token via `exec`, pulls the bundle, and starts `launch/supervisor.py`
detached. The supervisor downloads datasets on the VM, runs the jobs (re-launching crashed ones
from their last checkpoint), pushes every run to the HF repo every few minutes, and finally
**unassigns its own VM**. Nothing depends on the launching machine staying connected. A
watchdog stops the VM if launch stalls, and VMs created during a failed launch are released.

### Tables

```bash
python analysis/table.py e1base
```

```bash
python analysis/meta_table.py e2mlp
```

```bash
python analysis/central.py "Adam=runs/e1base_adam_lr1e-3_s*/metrics.jsonl" "Editor=runs/e2mlp_bestre_m0_s*/metrics.jsonl"
```
