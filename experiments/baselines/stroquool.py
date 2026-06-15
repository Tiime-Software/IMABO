"""
Baseline tree-based optimizers: StoSOO, HOO-T, Stroquool, Sequool.

These are deterministic / stochastic hierarchical search algorithms that operate
on the unit hypercube [0,1]^d.  They are exposed as generator coroutines that
yield the next point to evaluate and accept the observed reward via .send().
TimedOptimizer wraps any such generator into the standard suggest/observe interface.
"""

import random
import warnings
from collections import defaultdict
from collections.abc import Callable, Generator
from dataclasses import dataclass
from math import ceil, floor, log, sqrt

import numpy as np


@dataclass(frozen=True)
class Node:
    k: tuple[int, ...]  # numerators
    n: tuple[int, ...]  # denominator exponents
    depth: int          # tree depth (splitting dimension = depth % D)

    @staticmethod
    def root(dim: int) -> "Node":
        return Node(k=(1,) * dim, n=(1,) * dim, depth=0)

    def to_point(self) -> np.ndarray:
        return np.array([self.k[i] / (1 << self.n[i]) for i in range(len(self.k))])

    def children(self) -> tuple["Node", "Node"]:
        d = self.depth % len(self.k)
        kL, nL = list(self.k), list(self.n)
        kR, nR = list(self.k), list(self.n)
        kL[d] = 2 * self.k[d] - 1
        nL[d] = self.n[d] + 1
        kR[d] = 2 * self.k[d] + 1
        nR[d] = self.n[d] + 1
        return (
            Node(tuple(kL), tuple(nL), self.depth + 1),
            Node(tuple(kR), tuple(nR), self.depth + 1),
        )

    def parent(self) -> "Node":
        if self.depth == 0:
            raise ValueError("Root has no parent.")
        d = (self.depth - 1) % len(self.k)
        kP, nP = list(self.k), list(self.n)
        kP[d] = (self.k[d] - 1) // 4 * 2 + 1
        nP[d] = self.n[d] - 1
        return Node(tuple(kP), tuple(nP), self.depth - 1)

    def __repr__(self) -> str:
        return f"Node({self.to_point()}, depth={self.depth})"


@dataclass
class Eval:
    sum: float
    cnt: int


def sequool(n: int, dim: int):
    def n_bound(h_max: int):
        return 2 * sum(floor(h_max / h) for h in range(1, h_max + 1))

    def find_h_max(n: int):
        a, b = 0, n
        while b - a > 1:
            c = (a + b) // 2
            if n_bound(c) < n:
                a = c
            else:
                b = c
        return a

    h_max = find_h_max(n)
    evals: list[dict[Node, float]] = [{} for _ in range(h_max + 2)]

    def open_cell(node: Node):
        for child in node.children():
            evals[child.depth][child] = yield child.to_point()

    def best_x():
        _, ans = max((v, k) for e in evals for k, v in e.items())
        return ans.to_point()

    yield best_x

    root = Node.root(dim)
    yield from open_cell(root)
    for h in range(1, h_max + 1):
        top_cells = sorted((v, k) for k, v in evals[h].items())[-floor(h_max / h):]
        for _, cell in top_cells:
            yield from open_cell(cell)

    return best_x()


def stroquool(n: int, dim: int):
    def n_bound(h_max):
        return 2 * (
            h_max
            + sum(
                floor(h_max / (h * m))
                for h in range(1, h_max + 1)
                for m in range(1, h_max // h)
            )
        ) + (floor(log(h_max)) + 1) * (h_max // 2)

    def find_h_max(n: int):
        a, b = 0, n
        while b - a > 1:
            c = (a + b) // 2
            if n_bound(c) < n:
                a = c
            else:
                b = c
        return a

    h_max = find_h_max(n)
    evals: list[dict[Node, Eval]] = [
        defaultdict(lambda: Eval(0, 0)) for _ in range(h_max + 1)
    ]
    opened_cells: set[Node] = set()

    def evaluate(node: Node):
        e = evals[node.depth][node]
        e.cnt += 1
        e.sum += yield node.to_point()

    def open_cell(node: Node):
        opened_cells.add(node)
        for child in node.children():
            yield from evaluate(child)

    root = Node.root(dim)
    for _ in range(h_max):
        yield from open_cell(root)

    for h in range(1, h_max + 1):
        for m in range(1, h_max // h):
            cell, _ = max(
                (
                    (x, e)
                    for x, e in evals[h].items()
                    if x not in opened_cells and e.cnt >= floor(h_max / (h * m))
                ),
                key=lambda xe: xe[1].sum / xe[1].cnt,
                default=(None, None),
            )
            if cell is None:
                continue
            for _ in range(floor(h_max / (h * m))):
                yield from open_cell(cell)

    candidates: list[tuple[float, Node]] = []
    for p in range(floor(log(h_max)) + 1):
        cand, _ = max(
            ((x, e) for ev in evals for x, e in ev.items() if e.cnt >= 2**p),
            key=lambda xe: xe[1].sum / xe[1].cnt,
        )
        s = 0
        for _ in range(h_max // 2):
            s += yield cand.to_point()
        candidates.append((s, cand))
    _, ans = max(candidates, key=lambda x: x[0])
    return ans.to_point()


def stosoo(n, dim: int, k=None, h_max=None, delta=None):
    if k is None:
        k = round(n / (log(n) ** 3))
    if h_max is None:
        h_max = round(sqrt(n / k))
    if delta is None:
        delta = 1 / sqrt(n)

    evals = [defaultdict(lambda: Eval(0, 0)) for _ in range(h_max + 1)]
    root = Node.root(dim)
    evals[0][root] = Eval(0, 0)

    def evaluate(node: Node):
        e = evals[node.depth][node]
        e.cnt += 1
        e.sum += yield node.to_point()

    def expand(node: Node):
        for child in node.children():
            evals[child.depth][child]

    def depth():
        return max(h for h in range(h_max) if evals[h])

    def is_leaf(node):
        child, *_ = node.children()
        return child not in evals[child.depth]

    def leaves(h):
        return [l for l in evals[h] if is_leaf(l)]

    def get_eval(node):
        return evals[node.depth][node]

    def b_val(node):
        e = get_eval(node)
        if e.cnt == 0:
            return float("inf")
        return e.sum / e.cnt + sqrt(log(n * k / delta) / (2 * e.cnt))

    def depth_without_leaves():
        return max(
            h for h in range(h_max) if any(not is_leaf(node) for node in evals[h])
        )

    def best_x():
        best_avg = float("-inf")
        best_node = None
        for k_, v in evals[depth_without_leaves()].items():
            if v.cnt > 0:
                avg = v.sum / v.cnt
                if avg >= best_avg:
                    best_avg = avg
                    best_node = k_
        return best_node.to_point()

    yield best_x

    while True:
        b_max = -float("inf")
        for h in range(min(depth(), h_max) + 1):
            best_leaf = max(leaves(h), key=b_val, default=None)
            if best_leaf is None:
                continue
            b_best = b_val(best_leaf)
            if b_best > b_max:
                if get_eval(best_leaf).cnt < k:
                    yield from evaluate(best_leaf)
                else:
                    expand(best_leaf)
                    b_max = b_best


def hoo_t(n: int, dim: int, nu1, rho):
    """HOO-T algorithm (fixed horizon variant)."""
    root = Node.root(dim)
    tree: dict[Node, Eval] = defaultdict(lambda: Eval(0, 0))
    b_vals: dict[Node, float] = defaultdict(lambda: float("inf"))

    def u_val(node: Node):
        e = tree[node]
        return e.sum / e.cnt + sqrt(2 * log(n) / e.cnt) + nu1 * rho**node.depth

    def best_x():
        return max(
            (node for node in tree if tree[node].cnt > 0),
            key=lambda node: b_vals[node],
        ).to_point()

    yield best_x

    max_depth = ceil((log(n) / 2 - log(1 / nu1)) / log(1 / rho))

    for _ in range(n):
        node = root
        path = [root]
        while node in tree and node.depth < max_depth:
            c1, c2 = node.children()
            b1 = b_vals[c1]
            b2 = b_vals[c2]
            node = c1 if b1 > b2 or (b1 == b2 and random.random() < 0.5) else c2
            path.append(node)

        reward = yield node.to_point()

        for node in path:
            e = tree[node]
            e.cnt += 1
            e.sum += reward

        for node in reversed(path):
            if node.depth == max_depth:
                b_vals[node] = u_val(node)
            else:
                c1, c2 = node.children()
                b_vals[node] = min(u_val(node), max(b_vals[c1], b_vals[c2]))

    return best_x()


class TimedOptimizer:
    """Wraps a generator-based optimizer into a suggest/observe interface."""

    def __init__(self, gen: Generator | Callable, *args, **kwargs):
        if callable(gen):
            self.gen = gen(*args, **kwargs)
        else:
            self.gen = gen
        yielded = next(self.gen)
        if callable(yielded):
            self._best_x = yielded
            self.next_x = next(self.gen)
        else:
            self.next_x = yielded
        self.done = False

    def suggest(self):
        assert not self.done, "Optimization is already finished"
        return self.next_x

    def observe(self, x, y) -> None:
        assert not self.done, "Optimization is already finished"
        assert np.all(x == self.next_x), "You should only observe the suggested x"
        try:
            self.next_x = self.gen.send(y)
        except StopIteration as e:
            if e.value is not None:
                val = e.value
                self._best_x = lambda: val
            self.done = True

    def suggest_best(self):
        return self._best_x()
