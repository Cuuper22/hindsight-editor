"""Datasets, loaded whole into memory (no torchvision dependency).

MNIST / CIFAR-10 return uint8 tensors; batches are normalized on the fly.
Text datasets return a flat uint8/int32 token tensor per split.
"""
import gzip, io, os, pickle, struct, tarfile, urllib.request, zipfile
import numpy as np
import torch

DATA_DIR = os.environ.get("HS_DATA", os.path.join(os.path.dirname(__file__), "..", "data"))


def _fetch(url, name):
    os.makedirs(DATA_DIR, exist_ok=True)
    path = os.path.join(DATA_DIR, name)
    for attempt in range(6):                                       # flaky networks; parallel runs
        if os.path.exists(path):
            return path
        tmp = f"{path}.{os.getpid()}.part"
        try:
            urllib.request.urlretrieve(url, tmp)
            os.replace(tmp, path)
        except Exception:
            if attempt == 5:
                raise
            import time
            time.sleep(5 * (attempt + 1))
    return path


def _idx(path):
    with gzip.open(path, "rb") as f:
        data = f.read()
    ndim = data[3]                                                  # magic = 0x00 0x00 dtype ndim
    dims = struct.unpack(">" + "I" * ndim, data[4:4 + 4 * ndim])
    return np.frombuffer(data, np.uint8, offset=4 + 4 * ndim).reshape(dims)


def mnist():
    base = "https://ossci-datasets.s3.amazonaws.com/mnist/"
    out = {}
    for split, pre in (("train", "train"), ("test", "t10k")):
        x = _idx(_fetch(base + f"{pre}-images-idx3-ubyte.gz", f"mnist-{pre}-x.gz"))
        y = _idx(_fetch(base + f"{pre}-labels-idx1-ubyte.gz", f"mnist-{pre}-y.gz"))
        out[split] = (torch.from_numpy(x.copy()).reshape(len(x), 1, 28, 28), torch.from_numpy(y.astype(np.int64)))
    return out, (0.1307,), (0.3081,)


def cifar10():
    """CIFAR-10 from the HF Hub parquet export (fast from Colab); Toronto tarball as fallback."""
    cache = os.path.join(DATA_DIR, "cifar10.npz")
    if not os.path.exists(cache):
        try:
            import pandas as pd
            from PIL import Image
            from huggingface_hub import hf_hub_download
            arrs = {}
            for split in ("train", "test"):
                f = hf_hub_download("uoft-cs/cifar10", f"plain_text/{split}/0000.parquet", repo_type="dataset",
                                    revision="refs/convert/parquet", local_dir=os.path.join(DATA_DIR, "hf_cifar"))
                df = pd.read_parquet(f)
                x = np.stack([np.asarray(Image.open(io.BytesIO(b["bytes"])).convert("RGB")) for b in df["img"]])
                arrs[f"x_{split}"] = x.transpose(0, 3, 1, 2).copy()
                arrs[f"y_{split}"] = df["label"].to_numpy().astype(np.int64)
            os.makedirs(DATA_DIR, exist_ok=True)
            np.savez(cache + ".tmp.npz", **arrs)
            os.replace(cache + ".tmp.npz", cache)
        except Exception as e:
            print("HF CIFAR-10 failed, falling back to Toronto:", repr(e)[:200], flush=True)
            return _cifar10_toronto()
    z = np.load(cache)
    out = {s_: (torch.from_numpy(z[f"x_{s_}"]), torch.from_numpy(z[f"y_{s_}"])) for s_ in ("train", "test")}
    return out, (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)


def _cifar10_toronto():
    path = _fetch("https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz", "cifar-10-python.tar.gz")
    xs, ys = {"train": [], "test": []}, {"train": [], "test": []}
    with tarfile.open(path) as tf:
        for m in tf.getmembers():
            b = os.path.basename(m.name)
            split = "train" if b.startswith("data_batch") else "test" if b == "test_batch" else None
            if split:
                d = pickle.load(tf.extractfile(m), encoding="bytes")
                xs[split].append(d[b"data"]); ys[split].extend(d[b"labels"])
    out = {s: (torch.from_numpy(np.concatenate(xs[s])).reshape(-1, 3, 32, 32), torch.tensor(ys[s])) for s in xs}
    return out, (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)


class ImageData:
    """Holds a dataset on `device`; train split is further cut into train/val (val = held-out for meta/eval)."""

    def __init__(self, name, device, n_val=5000, seed=0):
        raw, mean, std = {"mnist": mnist, "cifar10": cifar10}[name]()
        x, y = raw["train"]
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(len(x), generator=g)
        vi, ti = perm[:n_val], perm[n_val:]
        self.split = {"train": (x[ti], y[ti]), "val": (x[vi], y[vi]), "test": raw["test"]}
        self.split = {k: (a.to(device), b.to(device)) for k, (a, b) in self.split.items()}
        self.mean = torch.tensor(mean, device=device).view(1, -1, 1, 1)
        self.std = torch.tensor(std, device=device).view(1, -1, 1, 1)
        self.n_classes = 10
        self.in_shape = tuple(x.shape[1:])
        self.device = device

    def norm(self, x):
        return (x.float() / 255.0 - self.mean) / self.std

    def batch(self, split, bs, gen):
        x, y = self.split[split]
        idx = torch.randint(len(x), (bs,), generator=gen, device="cpu").to(self.device)
        return self.norm(x[idx]), y[idx]

    def iterate(self, split, bs):
        x, y = self.split[split]
        for i in range(0, len(x), bs):
            yield self.norm(x[i:i + bs]), y[i:i + bs]


def enwik8():
    """Byte-level enwik8 with the standard 90M/5M/5M split."""
    path = _fetch("http://mattmahoney.net/dc/enwik8.zip", "enwik8.zip")
    with zipfile.ZipFile(path) as z:
        data = np.frombuffer(z.read("enwik8"), dtype=np.uint8)
    n = 90_000_000
    return {"train": torch.from_numpy(data[:n].copy()), "val": torch.from_numpy(data[n:n + 5_000_000].copy()),
            "test": torch.from_numpy(data[n + 5_000_000:].copy())}, 256


def _llmc_shard(path):
    """llm.c token shard: 256 int32 header (magic 20240520, version, n_tokens), then uint16 tokens."""
    with open(path, "rb") as f:
        head = np.frombuffer(f.read(256 * 4), dtype=np.int32)
        assert head[0] == 20240520, "not an llm.c shard"
        return np.frombuffer(f.read(), dtype=np.uint16, count=int(head[2]))


def fineweb_edu(n_train_tokens, n_val_tokens=10_000_000):
    """GPT-2 tokens of FineWeb-Edu from karpathy/fineweb-edu-100B-gpt2-token-shards (100M tokens per
    shard; downloaded on the VM only). Val = first n_val_tokens of the official val shard."""
    from huggingface_hub import hf_hub_download
    repo, local = "karpathy/fineweb-edu-100B-gpt2-token-shards", os.path.join(DATA_DIR, "fwe")
    get = lambda name: _llmc_shard(hf_hub_download(repo, name, repo_type="dataset", local_dir=local))
    n_shards = -(-n_train_tokens // 100_000_000)
    train = np.concatenate([get(f"edu_fineweb_train_{i:06d}.bin") for i in range(1, n_shards + 1)])[:n_train_tokens]
    val = get("edu_fineweb_val_000000.bin")[:n_val_tokens]
    return {"train": torch.from_numpy(train.astype(np.int32)), "val": torch.from_numpy(val.astype(np.int32))}, 50257


class TextData:
    def __init__(self, name, device, ctx, n_train_tokens=None):
        splits, self.vocab = enwik8() if name == "enwik8" else fineweb_edu(n_train_tokens)
        self.split = {k: v.to(device) for k, v in splits.items()}
        self.ctx, self.device = ctx, device

    def batch(self, split, bs, gen):
        d = self.split[split]
        i = torch.randint(len(d) - self.ctx - 1, (bs,), generator=gen).to(self.device)
        off = torch.arange(self.ctx + 1, device=self.device)
        seq = d[i[:, None] + off].long()
        return seq[:, :-1], seq[:, 1:]

    def unigram_nats(self):
        counts = torch.bincount(self.split["train"].long(), minlength=self.vocab).double()
        p = counts / counts.sum()
        val = self.split["val"].long()
        return float(-(p[val].clamp_min(1e-12).log()).mean())
