"""Modified Kneser-Ney smoothing (Chen & Goodman 1998)."""

import math
from collections import defaultdict

from src.ngram import CountTable
from src.smoothing.base import Smoother


class KneserNey(Smoother):
    """Modified Kneser-Ney with three discount parameters and recursive interpolation."""

    name = "KneserNey"

    def __init__(self) -> None:
        self.ct: CountTable | None = None
        self.d1: float = 0.0
        self.d2: float = 0.0
        self.d3: float = 0.0
        # Precomputed continuation counts
        self._continuation_count: dict[str, int] = {}  # word -> |{w': c(w',w)>0}|
        self._total_bigram_types: int = 0  # |{(w',w): c(w',w)>0}|
        # Trigram continuation counts: (w1,) -> word -> count of distinct left contexts
        self._tri_continuation: dict[tuple[str,], dict[str, int]] = {}
        self._total_trigram_types: int = 0

    def fit(self, count_table: CountTable) -> None:
        """Estimate discount parameters and precompute continuation counts."""
        self.ct = count_table
        self._estimate_discounts()
        self._build_continuation_counts()

    def _estimate_discounts(self) -> None:
        """Estimate d1, d2, d3+ using the standard Y formula from Chen & Goodman."""
        assert self.ct is not None

        # Compute N1, N2, N3, N4 from bigram counts
        freq: dict[int, int] = defaultdict(int)
        for ctx_words in self.ct.bigram_counts.values():
            for c in ctx_words.values():
                freq[c] += 1

        n1 = freq.get(1, 0)
        n2 = freq.get(2, 0)
        n3 = freq.get(3, 0)
        n4 = freq.get(4, 0)

        if n1 == 0 or n2 == 0:
            self.d1 = 0.1
            self.d2 = 0.2
            self.d3 = 0.3
            return

        Y = n1 / (n1 + 2 * n2)
        self.d1 = 1.0 - 2.0 * Y * n2 / n1 if n1 > 0 else 0.0
        self.d2 = 2.0 - 3.0 * Y * n3 / n2 if n2 > 0 else 0.0
        self.d3 = 3.0 - 4.0 * Y * n4 / n3 if n3 > 0 else 0.0

        # Clamp to valid range
        self.d1 = max(0.01, min(self.d1, 0.99))
        self.d2 = max(0.01, min(self.d2, 1.99))
        self.d3 = max(0.01, min(self.d3, 2.99))

    def _build_continuation_counts(self) -> None:
        """Precompute continuation counts for lower-order distributions."""
        assert self.ct is not None

        # Bigram continuation: for each word w, count distinct left contexts w'
        # such that c(w', w) > 0
        cont: dict[str, int] = defaultdict(int)
        total_types = 0
        for ctx, words in self.ct.bigram_counts.items():
            for w in words:
                cont[w] += 1
                total_types += 1
        self._continuation_count = dict(cont)
        self._total_bigram_types = total_types

        # Trigram continuation: for each (w2,), count distinct w1 such that
        # c(w1, w2, w) > 0 for each w
        tri_cont: dict[tuple[str,], dict[str, int]] = defaultdict(lambda: defaultdict(int))
        total_tri = 0
        for ctx, words in self.ct.trigram_counts.items():
            # ctx is (w1, w2), we want continuation count for w given (w2,)
            w2 = ctx[1]
            for w in words:
                tri_cont[(w2,)][w] += 1
                total_tri += 1
        self._tri_continuation = {k: dict(v) for k, v in tri_cont.items()}
        self._total_trigram_types = total_tri

    def _discount(self, count: int) -> float:
        """Return the appropriate discount for a given count."""
        if count == 1:
            return self.d1
        elif count == 2:
            return self.d2
        else:
            return self.d3

    def prob(self, word: str, context: tuple[str, ...]) -> float:
        """Return log P(word | context) using Modified Kneser-Ney interpolation.

        Supports trigram (context len 2), bigram (context len 1), and unigram (context len 0).
        """
        assert self.ct is not None
        p = self._prob_recursive(word, context)
        if p <= 0:
            return float("-inf")
        return math.log(p)

    def _prob_recursive(self, word: str, context: tuple[str, ...]) -> float:
        """Recursive interpolated probability (linear, not log space)."""
        assert self.ct is not None

        if len(context) == 0:
            # Unigram: use continuation counts p_cont(w) = N1+(*, w) / N1+(*, *)
            cont_w = self._continuation_count.get(word, 0)
            if self._total_bigram_types > 0:
                p = cont_w / self._total_bigram_types
            else:
                p = 1.0 / max(self.ct.vocab_size, 1)
            # Floor to prevent zero
            return max(p, 1e-10)

        if len(context) == 1:
            return self._interpolated_bigram(word, context)

        # Trigram: context is (w1, w2)
        return self._interpolated_trigram(word, context[-2:])

    def _interpolated_bigram(self, word: str, context: tuple[str, ...]) -> float:
        """Interpolated bigram probability with KN continuation counts as lower order."""
        assert self.ct is not None
        ctx_counts = self.ct.bigram_counts.get(context, {})
        count_ctx = sum(ctx_counts.values())

        if count_ctx == 0:
            return self._prob_recursive(word, ())

        count_cw = ctx_counts.get(word, 0)
        d = self._discount(count_cw) if count_cw > 0 else 0.0

        first_term = max(count_cw - d, 0.0) / count_ctx

        # Interpolation weight: gamma
        n1_ctx = sum(1 for c in ctx_counts.values() if c == 1)
        n2_ctx = sum(1 for c in ctx_counts.values() if c == 2)
        n3p_ctx = sum(1 for c in ctx_counts.values() if c >= 3)

        gamma = (self.d1 * n1_ctx + self.d2 * n2_ctx + self.d3 * n3p_ctx) / count_ctx

        lower = self._prob_recursive(word, ())
        return first_term + gamma * lower

    def _interpolated_trigram(self, word: str, context: tuple[str, ...]) -> float:
        """Interpolated trigram probability."""
        assert self.ct is not None
        ctx_counts = self.ct.trigram_counts.get(context, {})
        count_ctx = sum(ctx_counts.values())

        if count_ctx == 0:
            return self._prob_recursive(word, context[-1:])

        count_cw = ctx_counts.get(word, 0)
        d = self._discount(count_cw) if count_cw > 0 else 0.0

        first_term = max(count_cw - d, 0.0) / count_ctx

        n1_ctx = sum(1 for c in ctx_counts.values() if c == 1)
        n2_ctx = sum(1 for c in ctx_counts.values() if c == 2)
        n3p_ctx = sum(1 for c in ctx_counts.values() if c >= 3)

        gamma = (self.d1 * n1_ctx + self.d2 * n2_ctx + self.d3 * n3p_ctx) / count_ctx

        lower = self._prob_recursive(word, context[-1:])
        return first_term + gamma * lower
