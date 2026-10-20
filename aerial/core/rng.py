"""Named random streams and the pairing check (T0.4).

Environment streams (``comm``, ``attack``, ``sense``, ``requery``) must draw the same blocks every tick whatever the
controllers do; that is what makes naive and gated runs of one seed comparable. ``CountingGenerator`` records every
draw so ``run_pair`` can assert it.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

ENV_STREAMS = ("comm", "attack", "sense", "requery")
STREAMS = ("world",) + ENV_STREAMS + ("ctrl",)


class CountingGenerator:
    """Wraps a numpy Generator and logs (method, shape) of every draw."""

    def __init__(self, gen: np.random.Generator):
        self._gen = gen
        self.log: List[Tuple[str, Tuple]] = []

    def __getattr__(self, name):
        attr = getattr(self._gen, name)
        if not callable(attr):
            return attr

        def wrapped(*args, **kwargs):
            out = attr(*args, **kwargs)
            self.log.append((name, np.shape(out)))
            return out

        return wrapped

    def signature(self) -> int:
        return hash(tuple(self.log))


class Streams:
    def __init__(self, seed: int, n_agents: int):
        children = np.random.SeedSequence(seed).spawn(len(STREAMS))
        self.seq: Dict[str, np.random.SeedSequence] = dict(zip(STREAMS, children))
        self.world = np.random.default_rng(self.seq["world"])
        self.env: Dict[str, CountingGenerator] = {
            k: CountingGenerator(np.random.default_rng(self.seq[k])) for k in ENV_STREAMS
        }
        self.ctrl = [np.random.default_rng(s) for s in self.seq["ctrl"].spawn(n_agents)]

    def __getattr__(self, name):
        if name in ENV_STREAMS:
            return self.env[name]
        raise AttributeError(name)

    def env_signature(self) -> Dict[str, int]:
        return {k: g.signature() for k, g in self.env.items()}
