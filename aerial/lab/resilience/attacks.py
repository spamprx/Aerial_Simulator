"""Attack interface (F-local adversary model) and the simulated attack library.

The simulator routes every broadcast through an ``Attack``, so that compromised agents can report
*false* states to their neighbours while their true physical state is untouched.  New attacks plug
in via the registry:

    @attack_registry.register("constant_offset")
    class ConstantOffset(Attack): ...
"""
from __future__ import annotations

from abc import ABC
from typing import Dict, Optional, Tuple

import numpy as np

from ..registry import attack_registry


class Attack(ABC):
    def reset(self, sim) -> None:
        """Called once when the simulator is reset (access to sim.cfg, sim.compromised...)."""

    def corrupt(
        self,
        t: float,
        pos: np.ndarray,
        vel: np.ndarray,
        compromised: np.ndarray,
        rng: np.random.Generator,
        adj: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Return the (pos, vel) that each agent *broadcasts*.  Must not modify inputs.

        ``compromised`` is a boolean mask; only those rows may differ from the truth.
        The F-local model (at most F bad agents in any honest agent's neighbourhood) is a
        property of *which* agents are compromised, enforced by the experiment setup.
        ``adj`` is this step's receive graph.  It is passed only to an attack whose ``corrupt``
        declares it, so attacks written without it keep working.
        """
        return pos, vel

    def corrupt_payload(self, t: float, payload: Dict[str, np.ndarray], compromised: np.ndarray) -> Dict[str, np.ndarray]:
        """Return the extra arrays each agent broadcasts (see ``Controller.broadcast_payload``).  Must not modify
        inputs, and only the rows of compromised agents may differ from the truth."""
        return payload


@attack_registry.register("none")
class NoAttack(Attack):
    pass


@attack_registry.register("constant_offset")
class ConstantOffset(Attack):
    """Broadcast a fixed position/velocity offset from compromised agents."""

    def __init__(self, pos_offset=0.0, vel_offset=0.0):
        self.pos_offset = pos_offset
        self.vel_offset = vel_offset

    def corrupt(self, t, pos, vel, compromised, rng):
        msg_pos, msg_vel = pos.copy(), vel.copy()
        msg_pos[compromised] += np.asarray(self.pos_offset)
        msg_vel[compromised] += np.asarray(self.vel_offset)
        return msg_pos, msg_vel


@attack_registry.register("slow_drift")
class SlowDrift(Attack):
    """Broadcast an offset that grows linearly with simulation time."""

    def __init__(self, pos_rate=0.0, vel_rate=0.0):
        self.pos_rate = pos_rate
        self.vel_rate = vel_rate

    def corrupt(self, t, pos, vel, compromised, rng):
        msg_pos, msg_vel = pos.copy(), vel.copy()
        msg_pos[compromised] += t * np.asarray(self.pos_rate)
        msg_vel[compromised] += t * np.asarray(self.vel_rate)
        return msg_pos, msg_vel


@attack_registry.register("random_spoof")
class RandomSpoof(Attack):
    """Broadcast independent Gaussian position/velocity noise when compromised."""

    def __init__(self, pos_std=1.0, vel_std=1.0):
        self.pos_std, self.vel_std = float(pos_std), float(vel_std)
        if self.pos_std < 0 or self.vel_std < 0:
            raise ValueError("spoof standard deviations must be non-negative")

    def corrupt(self, t, pos, vel, compromised, rng):
        msg_pos, msg_vel = pos.copy(), vel.copy()
        count = int(compromised.sum())
        msg_pos[compromised] += rng.normal(0.0, self.pos_std, size=(count, pos.shape[1]))
        msg_vel[compromised] += rng.normal(0.0, self.vel_std, size=(count, vel.shape[1]))
        return msg_pos, msg_vel


@attack_registry.register("covert")
class Covert(Attack):
    """Report a false state that W-MSR accepts: the lie the filter cannot see.

    Each compromised agent broadcasts a position ``offset_mag`` from where it really is, kept per
    coordinate inside the range of the honest reports it hears, excluding the F most extreme on each
    side.  Honest values then lie beyond it in both directions, so it is never the value W-MSR trims.
    It works in the frame the fleet's filter uses, so under a formation it reports a place near its
    own slot.  Its velocity is held inside the honest range in the same way.

    It reports its offset in the chosen direction if that fits.  If not, it reports the middle of the
    honest values, the safest place to be, since receivers hear slightly different sets and the edge of
    its own range can be the extreme of theirs.  If the middle is closer to the truth than
    ``offset_mag``, it moves out only as far as it must.  Staying in range comes first, so a liar flying
    with a converged fleet has little room to lie, and the full offset needs one that is physically
    elsewhere, such as ``adversary_motion="hover"``.

    ``direction`` picks where the lie points: ``away_from_path`` is sideways from the honest agents'
    line of travel, ``fixed`` is along ``vector``, and ``random_per_run`` is drawn once at reset.
    One broadcast goes to everyone (D7).  ``f`` defaults to ``cfg.F``.  It also claims to know that it is
    safe, and to have caught nobody, which is what a liar hoping to pass as honest would say, and it reports
    the wrong distances, if the fleet shares them.
    """

    DIRECTIONS = ("away_from_path", "fixed", "random_per_run")

    def __init__(self, offset_mag: float = 3.0, direction: str = "away_from_path", vector=None, f: Optional[int] = None):
        if direction not in self.DIRECTIONS:
            raise ValueError(f"direction must be one of {self.DIRECTIONS}")
        if offset_mag < 0:
            raise ValueError("offset_mag must be non-negative")
        if (direction == "fixed") != (vector is not None):
            raise ValueError('give vector exactly when direction is "fixed"')
        self.offset_mag, self.direction, self.vector, self.f_param = float(offset_mag), direction, vector, f
        self.f = 1
        self.frame = None
        self.random_directions = None

    def reset(self, sim) -> None:
        self.f = sim.cfg.F if self.f_param is None else self.f_param
        self.frame = sim.controller.filter_offsets()
        if self.frame is None:
            self.frame = np.zeros((sim.n, sim.dim))
        if self.direction == "fixed":
            self.vector_unit = _unit(np.asarray(self.vector, dtype=float))
            if self.vector_unit.shape != (sim.dim,):
                raise ValueError(f"vector must have {sim.dim} coordinates")
        elif self.direction == "random_per_run":
            draws = sim.rng_attack.normal(size=(sim.n, sim.dim))
            self.random_directions = draws / np.linalg.norm(draws, axis=1, keepdims=True)

    def _pointing(self, i: int, pos: np.ndarray, vel: np.ndarray, heard: np.ndarray) -> np.ndarray:
        if self.direction == "fixed":
            return self.vector_unit
        if self.direction == "random_per_run":
            return self.random_directions[i]
        from_centre = pos[i] - pos[heard].mean(axis=0)
        heading = _unit(vel[heard].mean(axis=0))
        sideways = from_centre - (from_centre @ heading) * heading
        if np.linalg.norm(sideways) > 1e-9:
            return _unit(sideways)
        if np.linalg.norm(heading) > 0.0:  # dead on the line of travel: either side will do
            turned = np.zeros_like(heading)
            turned[:2] = -heading[1], heading[0]
            return _unit(turned) if np.linalg.norm(turned) > 0.0 else np.eye(len(heading))[0]
        return _unit(from_centre) if np.linalg.norm(from_centre) > 0.0 else np.eye(len(heading))[0]

    def corrupt(self, t, pos, vel, compromised, rng, adj=None):
        msg_pos, msg_vel = pos.copy(), vel.copy()
        for i in np.flatnonzero(compromised):
            hears = ~compromised if adj is None else adj[i] & ~compromised
            heard = np.flatnonzero(hears)
            pointing = self._pointing(i, pos, vel, heard) if len(heard) else self._fallback_pointing(pos.shape[1], i)
            true_value = pos[i] - self.frame[i]
            wanted = true_value + self.offset_mag * pointing
            if len(heard) == 0:  # nothing to stay inside of
                msg_pos[i] = wanted + self.frame[i]
                continue
            values = pos[heard] - self.frame[heard]
            low, high = _inner_range(values, self.f)
            chosen = wanted
            if np.any(wanted < low) or np.any(wanted > high):
                chosen = _least_exposed(np.clip(np.median(values, axis=0), low, high), low, high, true_value, self.offset_mag)
            msg_pos[i] = chosen + self.frame[i]
            low, high = _inner_range(vel[heard], self.f)
            msg_vel[i] = np.clip(vel[i], low, high)
        return msg_pos, msg_vel

    def corrupt_payload(self, t, payload, compromised):
        forged = dict(payload)
        if "k_safe" in payload:
            forged["k_safe"] = payload["k_safe"] | compromised
        if "known_faulty" in payload:
            forged["known_faulty"] = payload["known_faulty"] & ~compromised[:, None]
        if "ranges" in payload:
            forged["ranges"] = np.where(compromised[:, None], payload["ranges"] + self.offset_mag, payload["ranges"])
        return forged

    def _fallback_pointing(self, dim: int, i: int) -> np.ndarray:
        if self.direction == "fixed":
            return self.vector_unit
        return self.random_directions[i] if self.direction == "random_per_run" else np.eye(dim)[0]


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    return vector / norm if norm > 1e-12 else np.zeros_like(vector)


def _least_exposed(middle: np.ndarray, low: np.ndarray, high: np.ndarray, truth: np.ndarray, offset: float) -> np.ndarray:
    """The point nearest ``middle`` that is ``offset`` from ``truth``, on the way to the far corner of
    the box, or that corner if the box is too small to allow it."""
    gap = middle - truth
    if np.linalg.norm(gap) >= offset:
        return middle
    corner = np.where(np.abs(low - truth) > np.abs(high - truth), low, high)
    direction = corner - middle
    a, b = direction @ direction, direction @ gap
    if a < 1e-18:
        return middle
    along = (-b + np.sqrt(max(b ** 2 - a * (gap @ gap - offset ** 2), 0.0))) / a
    return middle + min(along, 1.0) * direction


def _inner_range(values: np.ndarray, f: int) -> Tuple[np.ndarray, np.ndarray]:
    """Per coordinate, the range left after setting aside the f lowest and f highest of ``values``.
    With too few values to set any aside, this is the median."""
    ordered = np.sort(values, axis=0)
    m = len(ordered)
    if m <= 2 * f:
        median = np.median(ordered, axis=0)
        return median, median
    return ordered[f], ordered[m - 1 - f]
