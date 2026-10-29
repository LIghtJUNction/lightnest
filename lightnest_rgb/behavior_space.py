"""Finite behavior posterior for an optional RGB research planner.

This is a deterministic structured model, not a neural execution predictor.
A ring of N Boolean outputs is restricted to at most K label transitions.
Only observed input/output pairs enter inference. No candidate is certified
outside this explicitly declared hypothesis class.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np


class InconsistentEvidence(ValueError):
    """No behavior in the declared class explains the observed answers."""


def transitions(bits: np.ndarray) -> int:
    a = np.asarray(bits)
    if a.ndim != 1 or len(a) < 2 or not np.isin(a, (0, 1)).all():
        raise ValueError("expected a one-dimensional Boolean behavior")
    return int(np.count_nonzero(a != np.roll(a, 1)))


def fit_penalty(training_tables: np.ndarray, max_transitions: int) -> float:
    """Fit one exponential-family parameter by matching training mean T.

    P(y) is proportional to exp(-penalty*T(y)) on the declared finite class.
    Ring multiplicity is 2*binomial(N,k) for even k, zero for odd k.
    The nonnegative penalty constraint is explicit; no test data is used.
    """
    rows = np.asarray(training_tables)
    if rows.ndim != 2 or len(rows) == 0 or not np.isin(rows, (0, 1)).all():
        raise ValueError("training_tables must be nonempty Boolean rows")
    n = rows.shape[1]
    _validate_class(n, max_transitions, 0.0)
    counts = np.count_nonzero(rows != np.roll(rows, 1, axis=1), axis=1)
    if np.any(counts > max_transitions):
        raise ValueError("training behaviors exceed the declared class")
    ks = np.arange(0, max_transitions + 1, 2, dtype=float)
    logs = np.array([math.log(2 * math.comb(n, int(k))) for k in ks])
    target = float(counts.mean())
    def mean(beta: float) -> float:
        z = logs - beta * ks
        p = np.exp(z - z.max())
        return float(p @ ks / p.sum())
    if target >= mean(0.0):
        return 0.0
    lo, hi = 0.0, 8.0
    for _ in range(64):
        mid = (lo + hi) / 2
        if mean(mid) > target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def _validate_class(n: int, cap: int, penalty: float) -> None:
    if not isinstance(n, int) or not 2 <= n <= 256:
        raise ValueError("domain size must be an integer in [2,256]")
    if not isinstance(cap, int) or not 0 <= cap <= min(n, 32):
        raise ValueError("transition cap must be an integer in [0,min(N,32)]")
    if not math.isfinite(penalty) or not 0 <= penalty <= 8:
        raise ValueError("penalty must be finite and in [0,8]")


@dataclass(frozen=True)
class BehaviorPosterior:
    probability_one: np.ndarray
    possible: np.ndarray  # [input, label]; structural support, not confidence
    log2_behavior_count: float
    observation_count: int

    @property
    def conditional_unique(self) -> bool:
        """Unique only inside the declared class, never a global proof."""
        return bool(np.all(self.possible.sum(axis=1) == 1))

    def choose_query(self, costs: np.ndarray | None = None) -> int | None:
        """Maximize binary entropy / positive query cost; deterministic ties.

        For unit costs the numerator is exact one-step mutual information.
        The cost ratio is a greedy heuristic, not a global optimality claim.
        Observed and structurally fixed bits cannot be selected.
        """
        eligible = self.possible.all(axis=1)
        if not eligible.any():
            return None
        c = np.ones(len(eligible)) if costs is None else np.asarray(costs, float)
        if c.shape != eligible.shape or not np.isfinite(c).all() or (c <= 0).any():
            raise ValueError("costs must be finite positive values per input")
        p = self.probability_one
        h = np.zeros_like(p)
        interior = (p > 0) & (p < 1)
        h[interior] = -p[interior]*np.log2(p[interior]) - (1-p[interior])*np.log2(1-p[interior])
        score = np.where(eligible, h/c, -np.inf)
        return int(np.argmax(score))


class BehaviorSpace:
    def __init__(self, size: int = 256, max_transitions: int = 12,
                 penalty: float = 0.0):
        _validate_class(size, max_transitions, penalty)
        self.size, self.cap, self.penalty = size, max_transitions, float(penalty)
        i = np.arange(self.cap + 1)
        total = i[:, None] + i[None, :]
        self._support_weights = (total <= self.cap).astype(float)
        self._weights = np.exp(-self.penalty * total) * self._support_weights

    def _known(self, evidence: np.ndarray) -> np.ndarray:
        x = np.asarray(evidence)
        if x.shape != (self.size,) or not np.isin(x, (-1, 0, 1)).all():
            raise ValueError("evidence must contain -1/0/1, one entry per input")
        return x.astype(np.int8, copy=True)

    def posterior(self, evidence: np.ndarray) -> BehaviorPosterior:
        known = self._known(evidence)
        n, k = self.size, self.cap
        mass = np.zeros((n, 2), dtype=float)
        counts = np.zeros((n, 2), dtype=float)
        for start in (0, 1):
            if known[0] not in (-1, start):
                continue
            forward = np.zeros((n, 2, k + 1), dtype=float)
            forward[0, start, 0] = 1
            for t in range(1, n):
                for bit in (0, 1):
                    if known[t] in (-1, bit):
                        forward[t, bit] = forward[t-1, bit]
                        forward[t, bit, 1:] += forward[t-1, 1-bit, :-1]
            backward = np.zeros_like(forward)
            for bit in (0, 1):
                closing = int(bit != start)
                if closing <= k:
                    backward[-1, bit, closing] = 1
            for t in range(n-2, -1, -1):
                for bit in (0, 1):
                    if known[t+1] in (-1, bit):
                        backward[t, bit] += backward[t+1, bit]
                    if known[t+1] in (-1, 1-bit):
                        backward[t, bit, 1:] += backward[t+1, 1-bit, :-1]
            # Positive-only arithmetic: zero support means no feasible path.
            weighted_tail = backward @ self._weights.T
            uniform_tail = backward @ self._support_weights.T
            mass += np.sum(forward * weighted_tail, axis=2)
            counts += np.sum(forward * uniform_tail, axis=2)
        if counts[0].sum() == 0:
            raise InconsistentEvidence("no behavior satisfies evidence and transition cap")
        p = mass[:, 1] / mass.sum(axis=1)
        return BehaviorPosterior(p, counts > 0, math.log2(float(counts[0].sum())),
                                 int(np.count_nonzero(known >= 0)))

    def decode(self, evidence: np.ndarray) -> np.ndarray:
        """Minimum-transition consistent behavior; fixed, non-oracle ties.

        For positive penalty this is a MAP behavior. At zero penalty it is
        an additional fixed simplicity tie-break. Active/random studies use
        the identical decoder. No target values or target metadata are read.
        """
        known = self._known(evidence)
        n, inf = self.size, self.size + 1
        best = None
        best_cost = inf
        for start in (0, 1):
            if known[0] not in (-1, start):
                continue
            cost = np.array([inf, inf], dtype=int)
            cost[start] = 0
            parents = np.zeros((n, 2), dtype=np.int8)
            for t in range(1, n):
                nxt = np.full(2, inf, dtype=int)
                for bit in (0, 1):
                    if known[t] in (-1, bit):
                        candidates = cost + np.array([bit != 0, bit != 1])
                        prev = int(candidates.argmin())
                        nxt[bit], parents[t, bit] = candidates[prev], prev
                cost = nxt
            ends = cost + np.array([start != 0, start != 1])
            end = int(ends.argmin())
            value = int(ends[end])
            if value <= self.cap and value < best_cost:
                out = np.empty(n, dtype=np.uint8)
                out[-1] = end
                for t in range(n-1, 0, -1):
                    out[t-1] = parents[t, out[t]]
                best, best_cost = out, value
        if best is None:
            raise InconsistentEvidence("no behavior satisfies evidence and transition cap")
        return best
