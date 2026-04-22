"""Simple Good-Turing with Katz-style backoff.

Implementation notes
--------------------

This implementation follows Gale & Sampson's *Simple Good-Turing* for the
count-rescaling step and then uses Katz-style backoff to turn the rescaled
counts into a conditional probability distribution. Two changes relative to
the previous implementation eliminate the pathological ``inf`` perplexities
that were observed for every Good-Turing run:

1. ``c*`` for any count ``c >= 1`` is taken from the log-linear regression
   rather than the raw frequency-of-frequency, so that sparse high-c bins
   (which often have ``N_c = 0`` in low-resource corpora) cannot drive the
   rescaled count to zero.

2. Unseen n-grams receive Katz-style backoff mass that is distributed via
   a lower-order Good-Turing distribution, never via an uninitialised
   ``c*[0] = 0``. A non-zero probability floor guarantees finite perplexity.

The ``unstable`` flag remains: when ``N_1 / N_2 < 2`` the regression is
ill-conditioned and the caller should be warned that Good-Turing is
unreliable at this corpus size.
"""

import math

import numpy as np

from src.ngram import CountTable
from src.smoothing.base import Smoother


class GoodTuring(Smoother):
    """Simple Good-Turing with Katz-style backoff to lower-order models."""

    name = "GoodTuring"

    def __init__(self) -> None:
        self.ct: CountTable | None = None
        self.unstable: bool = False
        self._regression: dict[int, tuple[float, float]] = {}
        self._n_c: dict[int, dict[int, int]] = {}
        # For each order, the total adjusted count and the missing-mass P0.
        self._total: dict[int, float] = {}
        self._p0: dict[int, float] = {}
        # Precomputed per-context totals (avoids sum() per vocab word call).
        self._bigram_ctx_total: dict[tuple, int] = {}
        self._trigram_ctx_total: dict[tuple, int] = {}
        # Cache alpha and normaliser per (context, order) to avoid O(vocab) recomputation.
        self._alpha_norm_cache: dict[tuple, tuple[float, float]] = {}

    def fit(self, count_table: CountTable) -> None:
        self.ct = count_table
        self._alpha_norm_cache.clear()
        self._bigram_ctx_total = {ctx: sum(w.values()) for ctx, w in count_table.bigram_counts.items()}
        self._trigram_ctx_total = {ctx: sum(w.values()) for ctx, w in count_table.trigram_counts.items()}
        for order in (1, 2, 3):
            n_c = self._compute_n_c(order)
            self._n_c[order] = n_c
            if not n_c:
                self._regression[order] = (0.0, -1.0)
                self._total[order] = 1.0
                self._p0[order] = 0.0
                continue

            # Stability flag (bigram level is the canonical diagnostic).
            if order == 2:
                n1 = n_c.get(1, 0)
                n2 = n_c.get(2, 0)
                if n2 == 0 or n1 / n2 < 2:
                    self.unstable = True

            a, b = self._fit_regression(n_c)
            self._regression[order] = (a, b)

            # Total observed count at this order.
            total_seen = sum(c * nc for c, nc in n_c.items())
            # Probability mass reserved for unseen events = N_1 / N.
            n1 = n_c.get(1, 0)
            p0 = n1 / total_seen if total_seen > 0 else 0.0

            self._total[order] = float(total_seen)
            self._p0[order] = float(p0)

    def prob(self, word: str, context: tuple[str, ...]) -> float:
        assert self.ct is not None
        order = len(context) + 1
        p = self._prob(word, context, order)
        if p <= 0:
            return float("-inf")
        return math.log(p)

    def _prob(self, word: str, context: tuple[str, ...], order: int) -> float:
        """Return the Good-Turing probability P(word | context) at this order."""
        assert self.ct is not None

        if order == 1:
            c = self.ct.unigram_counts.get(word, 0)
            total = self._total.get(1, 1.0)
            if c > 0:
                c_star = self._c_star(c, 1)
                return c_star / total
            # Unseen unigram: distribute P0 uniformly over unseen types.
            V = max(self.ct.vocab_size, 1)
            seen = len(self.ct.unigram_counts)
            n_unseen = max(V - seen, 1)
            p0 = self._p0.get(1, 0.0)
            return max(p0 / n_unseen, 1.0 / (V * total))

        if order == 2:
            count_ctx = self._bigram_ctx_total.get(context, 0)
            if count_ctx == 0:
                return self._prob(word, context[1:], order - 1)
            ctx_counts = self.ct.bigram_counts[context]
        else:
            count_ctx = self._trigram_ctx_total.get(context, 0)
            if count_ctx == 0:
                return self._prob(word, context[1:], order - 1)
            ctx_counts = self.ct.trigram_counts[context]

        c = ctx_counts.get(word, 0)
        if c > 0:
            c_star = self._c_star(c, order)
            # Katz discount: use the rescaled count as the numerator.
            return c_star / count_ctx

        # Seen context but unseen word: alpha and norm are the same for all
        # unseen words given this context, so cache them to avoid recomputing
        # O(|ctx_counts|) lower-order probs for every vocabulary word.
        cache_key = (context, order)
        if cache_key not in self._alpha_norm_cache:
            discounted_mass = 0.0
            for cw in ctx_counts.values():
                discounted_mass += self._c_star(cw, order) / count_ctx
            alpha = max(1.0 - discounted_mass, 0.0)
            norm = self._backoff_normaliser(ctx_counts, context, order) if alpha > 0.0 else 1.0
            self._alpha_norm_cache[cache_key] = (alpha, norm)
        alpha, norm = self._alpha_norm_cache[cache_key]

        if alpha == 0.0:
            # Floor: at least 1 / (V * count_ctx) so perplexity stays finite.
            return 1.0 / (max(self.ct.vocab_size, 1) * count_ctx)

        lower = self._prob(word, context[1:], order - 1)
        return alpha * lower / norm

    def _backoff_normaliser(
        self,
        ctx_counts: dict[str, int],
        context: tuple[str, ...],
        order: int,
    ) -> float:
        """Sum of lower-order probabilities over words unseen in ctx_counts."""
        total = 0.0
        for w in ctx_counts:
            total += self._prob(w, context[1:], order - 1)
        return max(1.0 - total, 1e-6)

    def _c_star(self, c: int, order: int) -> float:
        """Rescaled Good-Turing count c* = (c+1) * N_{c+1} / N_c.

        Uses the log-linear regression estimate of N_c for c >= 1 so that
        sparse high-c bins with N_c = 0 cannot produce c* = 0.
        """
        if c <= 0:
            return 0.0
        a, b = self._regression.get(order, (0.0, -1.0))
        nc = self._regression_nc(c, a, b)
        nc1 = self._regression_nc(c + 1, a, b)
        if nc <= 0:
            return float(c)
        return (c + 1) * nc1 / nc

    def _compute_n_c(self, order: int) -> dict[int, int]:
        assert self.ct is not None
        freq: dict[int, int] = {}
        if order == 1:
            for c in self.ct.unigram_counts.values():
                freq[c] = freq.get(c, 0) + 1
        elif order == 2:
            for ctx_words in self.ct.bigram_counts.values():
                for c in ctx_words.values():
                    freq[c] = freq.get(c, 0) + 1
        else:
            for ctx_words in self.ct.trigram_counts.values():
                for c in ctx_words.values():
                    freq[c] = freq.get(c, 0) + 1
        return freq

    def _fit_regression(self, n_c: dict[int, int]) -> tuple[float, float]:
        """Fit log(N_c) = a + b * log(c) on nonzero bins."""
        points = [(c, nc) for c, nc in n_c.items() if c > 0 and nc > 0]
        if len(points) < 2:
            return (0.0, -1.0)

        log_c = np.array([math.log(c) for c, _ in points])
        log_nc = np.array([math.log(nc) for _, nc in points])
        A = np.vstack([np.ones_like(log_c), log_c]).T
        a, b = np.linalg.lstsq(A, log_nc, rcond=None)[0]
        return (float(a), float(b))

    def _regression_nc(self, c: int, a: float, b: float) -> float:
        if c <= 0:
            return 0.0
        return math.exp(a + b * math.log(c))
