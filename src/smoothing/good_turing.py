"""Simple Good-Turing smoothing with log-linear regression on frequency-of-frequency counts."""

import math

import numpy as np

from src.ngram import CountTable, frequency_of_frequency, build_counts
from src.smoothing.base import Smoother


class GoodTuring(Smoother):
    """Simple Good-Turing smoothing for n-gram models."""

    name = "GoodTuring"
    K = 5  # threshold: smooth counts 0..K, use raw counts above K

    def __init__(self) -> None:
        self.ct: CountTable | None = None
        self.unstable: bool = False
        self._c_star: dict[int, dict[int, float]] = {}  # order -> {c: c*}
        self._total_adjusted: dict[int, float] = {}  # order -> sum of adjusted counts
        self._n_c: dict[int, dict[int, int]] = {}  # order -> {c: N_c}
        self._regression_params: dict[int, tuple[float, float]] = {}  # order -> (a, b)
        self._n_unseen: dict[int, int] = {}  # order -> number of unseen n-grams

    def fit(self, count_table: CountTable) -> None:
        """Compute Good-Turing adjusted counts for each order."""
        self.ct = count_table
        all_counts = {
            1: count_table.unigram_counts,
            2: count_table.bigram_counts,
            3: count_table.trigram_counts,
        }

        for order in [1, 2, 3]:
            n_c = self._compute_n_c(all_counts, order)
            self._n_c[order] = n_c

            if not n_c:
                continue

            # Check stability: N1/N2 < 2 indicates unreliable estimates
            n1 = n_c.get(1, 0)
            n2 = n_c.get(2, 0)
            if n2 > 0 and n1 / n2 < 2:
                self.unstable = True

            # Log-linear regression on log(c) vs log(N_c)
            a, b = self._fit_regression(n_c)
            self._regression_params[order] = (a, b)

            # Compute c* for c = 0..K
            c_star: dict[int, float] = {}
            for c in range(0, self.K + 1):
                nc = n_c.get(c, 0)
                nc1 = n_c.get(c + 1, 0)

                # Use regression estimate if actual N_c+1 is zero
                if nc1 == 0:
                    nc1 = self._regression_n_c(c + 1, a, b)
                if nc == 0 and c > 0:
                    nc = self._regression_n_c(c, a, b)

                if nc > 0:
                    c_star[c] = (c + 1) * nc1 / nc
                else:
                    c_star[c] = 0.0

            self._c_star[order] = c_star

            # Total adjusted mass for normalization
            total = 0.0
            seen_counts = self._get_all_counts(all_counts, order)
            for c, num in Counter_from_counts(seen_counts).items():
                if c <= self.K:
                    total += c_star.get(c, c) * num
                else:
                    total += c * num
            # Add mass for unseen n-grams
            n_unseen = self._count_unseen(order)
            self._n_unseen[order] = n_unseen
            total += c_star.get(0, 0) * n_unseen
            self._total_adjusted[order] = total if total > 0 else 1.0

    def prob(self, word: str, context: tuple[str, ...]) -> float:
        """Return log P(word | context) using Good-Turing smoothed counts."""
        assert self.ct is not None
        order = len(context) + 1

        if order == 1:
            raw_count = self.ct.unigram_counts.get(word, 0)
        elif order == 2:
            raw_count = self.ct.bigram_counts.get(context, {}).get(word, 0)
        elif order == 3:
            raw_count = self.ct.trigram_counts.get(context, {}).get(word, 0)
        else:
            return self.prob(word, context[-2:])

        c_star_map = self._c_star.get(order, {})
        total = self._total_adjusted.get(order, 1.0)

        if raw_count <= self.K:
            adjusted = c_star_map.get(raw_count, raw_count)
        else:
            adjusted = float(raw_count)

        if adjusted <= 0:
            return float("-inf")

        return math.log(adjusted / total)

    def _compute_n_c(
        self, all_counts: dict, order: int
    ) -> dict[int, int]:
        """Compute frequency-of-frequency counts for the given order."""
        freq: dict[int, int] = {}
        if order == 1:
            for c in all_counts[1].values():
                freq[c] = freq.get(c, 0) + 1
        else:
            for ctx_words in all_counts[order].values():
                for c in ctx_words.values():
                    freq[c] = freq.get(c, 0) + 1
        return freq

    def _fit_regression(self, n_c: dict[int, int]) -> tuple[float, float]:
        """Fit log-linear regression: log(N_c) = a + b * log(c).

        Returns (a, b).
        """
        points = [(c, nc) for c, nc in n_c.items() if c > 0 and nc > 0]
        if len(points) < 2:
            return (0.0, -1.0)

        log_c = np.array([math.log(c) for c, _ in points])
        log_nc = np.array([math.log(nc) for _, nc in points])

        # Linear regression: log_nc = a + b * log_c
        A = np.vstack([np.ones_like(log_c), log_c]).T
        result = np.linalg.lstsq(A, log_nc, rcond=None)
        a, b = result[0]
        return (float(a), float(b))

    def _regression_n_c(self, c: int, a: float, b: float) -> float:
        """Estimate N_c from the log-linear regression."""
        if c <= 0:
            return 0.0
        return math.exp(a + b * math.log(c))

    def _get_all_counts(self, all_counts: dict, order: int) -> list[int]:
        """Return a flat list of all n-gram counts for the given order."""
        if order == 1:
            return list(all_counts[1].values())
        counts_list = []
        for ctx_words in all_counts[order].values():
            counts_list.extend(ctx_words.values())
        return counts_list

    def _count_unseen(self, order: int) -> int:
        """Estimate the number of unseen n-grams."""
        assert self.ct is not None
        V = self.ct.vocab_size
        if order == 1:
            seen = len(self.ct.unigram_counts)
            return max(V - seen, 0)
        elif order == 2:
            seen = sum(len(words) for words in self.ct.bigram_counts.values())
            possible = V * V
            return max(possible - seen, 1)
        else:
            seen = sum(len(words) for words in self.ct.trigram_counts.values())
            possible = V * V * V
            return max(possible - seen, 1)


def Counter_from_counts(counts: list[int]) -> dict[int, int]:
    """Count how many n-grams have each count value."""
    result: dict[int, int] = {}
    for c in counts:
        result[c] = result.get(c, 0) + 1
    return result
