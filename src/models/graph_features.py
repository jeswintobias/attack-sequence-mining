"""Part A: technique transition graph, Node2Vec embeddings, and travel paths.

The graph is built from a fold's fit rows only: an edge a -> b for every adjacent
pair in a (prefix + next) sequence, weighted by duplicate-weighted frequency.

Three kinds of signal come out of it:

- **Node2Vec embeddings** - biased second-order random walks fed to skip-gram
  word2vec. These give *topological closeness*: techniques that sit in similar
  positions in the attack graph land near each other, which is the similarity
  prior that raw transition counts cannot express.
- **Hop distance** - unweighted shortest path, "how many steps away is this
  candidate".
- **Path cost** - shortest path under -log(transition probability) edge weights,
  i.e. the cost of the most likely route from the prefix to the candidate. This
  is the weighted analogue of hop distance and the one with real dynamic range.

Embedding dimension defaults to 16. With roughly 107 nodes, the 128 dimensions
that Node2Vec papers use would be parameter expansion rather than compression.
"""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
for p in [str(ROOT), str(SRC), str(SRC / "core"), str(SRC / "models")]:
    if p not in sys.path:
        sys.path.insert(0, p)

import config

UNREACHABLE_HOPS = 8.0        # cap so "unreachable" is a usable number, not inf
UNREACHABLE_COST = 20.0


class TransitionGraph:
    """Directed, weighted technique transition graph with derived geometry."""

    def __init__(self, dim: int = 16, walk_length: int = 20, num_walks: int = 20,
                 p: float = 1.0, q: float = 0.5, window: int = 5, seed: int | None = None):
        self.dim = dim
        self.walk_length = walk_length
        self.num_walks = num_walks
        self.p = p
        self.q = q
        self.window = window
        self.seed = config.RANDOM_SEED if seed is None else seed

    # ---- construction ------------------------------------------------------
    def fit(self, examples: pd.DataFrame):
        weights = defaultdict(float)
        nodes = set()
        for prefix, nxt, dup in zip(examples["prefix"], examples["next"],
                                    examples["dup_count"].to_numpy(dtype=float)):
            sequence = list(prefix) + [str(nxt)]
            nodes.update(sequence)
            for a, b in zip(sequence, sequence[1:]):
                weights[(a, b)] += dup
        self.nodes = sorted(nodes)
        self.index = {t: i for i, t in enumerate(self.nodes)}
        n = len(self.nodes)
        self.weight = np.zeros((n, n), dtype=np.float64)
        for (a, b), w in weights.items():
            self.weight[self.index[a], self.index[b]] = w
        row_sums = self.weight.sum(axis=1, keepdims=True)
        safe = np.where(row_sums > 0, row_sums, 1.0)
        self.probability = self.weight / safe
        self._fit_geometry()
        self._fit_embeddings()
        return self

    def _fit_geometry(self):
        """Hop distances and -log-probability shortest-path costs."""
        from scipy.sparse import csr_matrix
        from scipy.sparse.csgraph import shortest_path

        adjacency = csr_matrix((self.weight > 0).astype(np.float64))
        hops = shortest_path(adjacency, method="D", directed=True, unweighted=True)
        self.hops = np.where(np.isinf(hops), UNREACHABLE_HOPS,
                             np.minimum(hops, UNREACHABLE_HOPS))

        with np.errstate(divide="ignore"):
            cost = -np.log(np.where(self.probability > 0, self.probability, np.nan))
        cost = np.where(np.isnan(cost), 0.0, cost)
        weighted = csr_matrix(np.where(self.weight > 0, np.maximum(cost, 1e-9), 0.0))
        path = shortest_path(weighted, method="D", directed=True)
        self.path_cost = np.where(np.isinf(path), UNREACHABLE_COST,
                                  np.minimum(path, UNREACHABLE_COST))

    # ---- Node2Vec ----------------------------------------------------------
    def _alias_neighbours(self):
        self.neighbours = [np.flatnonzero(self.weight[i] > 0) for i in range(len(self.nodes))]
        self.neighbour_p = []
        for i, nbrs in enumerate(self.neighbours):
            if len(nbrs) == 0:
                self.neighbour_p.append(np.array([]))
                continue
            w = self.weight[i, nbrs]
            self.neighbour_p.append(w / w.sum())

    def _walks(self) -> list[list[str]]:
        """Second-order biased random walks (the p/q Node2Vec bias).

        p controls returning to the previous node, q controls moving outward
        versus staying in the local neighbourhood. q < 1 biases outward, which
        suits attack progression where we care about where a campaign travels.
        """
        rng = np.random.default_rng(self.seed)
        self._alias_neighbours()
        walks = []
        order = np.arange(len(self.nodes))
        for _ in range(self.num_walks):
            rng.shuffle(order)
            for start in order:
                if len(self.neighbours[start]) == 0:
                    continue
                walk = [start]
                while len(walk) < self.walk_length:
                    current = walk[-1]
                    nbrs = self.neighbours[current]
                    if len(nbrs) == 0:
                        break
                    base = self.neighbour_p[current]
                    if len(walk) == 1:
                        probabilities = base
                    else:
                        previous = walk[-2]
                        bias = np.ones(len(nbrs), dtype=np.float64)
                        for k, nxt in enumerate(nbrs):
                            if nxt == previous:
                                bias[k] = 1.0 / self.p
                            elif self.weight[previous, nxt] > 0:
                                bias[k] = 1.0
                            else:
                                bias[k] = 1.0 / self.q
                        probabilities = base * bias
                        total = probabilities.sum()
                        probabilities = base if total <= 0 else probabilities / total
                    walk.append(int(rng.choice(nbrs, p=probabilities)))
                walks.append([self.nodes[i] for i in walk])
        return walks

    def _fit_embeddings(self):
        from gensim.models import Word2Vec

        walks = self._walks()
        model = Word2Vec(
            sentences=walks, vector_size=self.dim, window=self.window,
            min_count=0, sg=1, negative=5, epochs=20,
            workers=1, seed=self.seed,
        )
        self.embedding = np.zeros((len(self.nodes), self.dim), dtype=np.float32)
        for technique, i in self.index.items():
            if technique in model.wv:
                self.embedding[i] = model.wv[technique]
        norms = np.linalg.norm(self.embedding, axis=1, keepdims=True)
        self.unit = self.embedding / np.where(norms > 0, norms, 1.0)
        self.n_walks = len(walks)

    # ---- lookups -----------------------------------------------------------
    def candidate_rows(self, candidates) -> np.ndarray:
        """Row index per candidate, -1 when the candidate is not in the graph."""
        return np.array([self.index.get(str(c), -1) for c in candidates], dtype=int)
