"""Analytic FLOP accounting. Every matmul in nets/editor reports here by tag.

Convention: a matmul of (N x k) by (k x m) costs 2*N*k*m. Backprop of a forward
pass costs 2x that forward (grad wrt inputs + grad wrt weights), the usual 3x rule.
"""
from collections import defaultdict
from contextlib import contextmanager

_counts = defaultdict(float)
_enabled = [True]


def add(tag, n):
    if _enabled[0]:
        _counts[tag] += n


def snapshot():
    return dict(_counts)


def reset():
    _counts.clear()


@contextmanager
def paused():
    _enabled[0] = False
    try:
        yield
    finally:
        _enabled[0] = True


@contextmanager
def measure():
    """with measure() as m: ...  -> m is a dict of FLOPs by tag spent inside the block."""
    before = dict(_counts)
    out = {}
    try:
        yield out
    finally:
        for k, v in _counts.items():
            dv = v - before.get(k, 0.0)
            if dv:
                out[k] = dv
