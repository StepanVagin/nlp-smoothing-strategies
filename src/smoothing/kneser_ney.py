"""Modified Kneser-Ney smoothing (Chen & Goodman 1998).

The key property of Modified Kneser-Ney that distinguishes it from plain
absolute discounting with interpolation is that lower-order distributions
are estimated from *continuation counts* rather than raw n-gram counts. A
word that has been seen in many distinct contexts is preferred in the
lower-order backoff, not a word that happens to have a high raw frequency
in one or two contexts (the classic "San Francisco" / "Francisco" example).

Precomputed quantities:

    N1+(*, w)          number of distinct left contexts w' such that c(w', w) > 0
    N1+(*, *)          number of distinct bigram types
    N1+(*, w2, w)      number of distinct w1 such that c(w1, w2, w) > 0
    N1+(*, w2, *)      number of distinct (w1, w) pairs such that c(w1, w2, w) > 0

Used as follows in a trigram model:

    P_kn(w | w1, w2)  = max(c(w1,w2,w) - d, 0)/c(w1,w2,*)
                        + gamma(w1,w2) * P_cont(w | w2)
    P_cont(w | w2)    = max(N1+(*, w2, w) - d, 0)/N1+(*, w2, *)
                        + gamma_cont(w2) * P_cont(w)
    P_cont(w)         = N1+(*, w) / N1+(*, *)
"""

import math
from collections import defaultdict

from src.ngram import CountTable
from src.smoothing.base import Smoother


class KneserNey(Smoother):
    """Modified Kneser-Ney with three discount parameters."""

    name = "KneserNey"

    def __init__(self) -> None:
        self.ct: CountTable | None = None
        self.d1: float = 0.0
        self.d2: float = 0.0
        self.d3: float = 0.0

        # N1+(*, w) and N1+(*, *)
        self._cont_unigram: dict[str, int] = {}
        self._total_bigram_types: int = 0

        # N1+(*, w2, w) keyed as _tri_cont[(w2,)][w] and totals N1+(*, w2, *)
        self._tri_cont: dict[tuple[str, ...], dict[str, int]] = {}
        self._tri_cont_total: dict[tuple[str, ...], int] = {}

    def fit(self, count_table: CountTable) -> None:
        self.ct = count_table
        self._estimate_discounts()
        self._build_continuation_counts()

    def _estimate_discounts(self) -> None:
        """Estimate d1, d2, d3+ using the Chen & Goodman Y formula."""
        assert self.ct is not None

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
        self.d1 = 1.0 - 2.0 * Y * n2 / n1
        self.d2 = 2.0 - 3.0 * Y * n3 / n2 if n2 > 0 else 0.2
        self.d3 = 3.0 - 4.0 * Y * n4 / n3 if n3 > 0 else 0.3

        self.d1 = max(0.01, min(self.d1, 0.99))
        self.d2 = max(0.01, min(self.d2, 1.99))
        self.d3 = max(0.01, min(self.d3, 2.99))

    def _build_continuation_counts(self) -> None:
        assert self.ct is not None

        cont: dict[str, int] = defaultdict(int)
        total_types = 0
        for ctx, words in self.ct.bigram_counts.items():
            for w in words:
                cont[w] += 1
                total_types += 1
        self._cont_unigram = dict(cont)
        self._total_bigram_types = total_types

        tri_cont: dict[tuple[str, ...], dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        for ctx, words in self.ct.trigram_counts.items():
            # ctx is (w1, w2); continuation is over w1 for a given (w2, w).
            w2 = ctx[1]
            for w in words:
                tri_cont[(w2,)][w] += 1
        self._tri_cont = {k: dict(v) for k, v in tri_cont.items()}
        self._tri_cont_total = {k: sum(v.values()) for k, v in self._tri_cont.items()}

    def _discount(self, count: int) -> float:
        if count == 1:
            return self.d1
        if count == 2:
            return self.d2
        return self.d3

    def prob(self, word: str, context: tuple[str, ...]) -> float:
        assert self.ct is not None
        if len(context) == 0:
            p = self._p_cont_unigram(word)
        elif len(context) == 1:
            # Top-level bigram: raw counts + interpolation with continuation unigram.
            p = self._bigram_top(word, context)
        else:
            ctx = context[-2:]
            p = self._trigram_top(word, ctx)
        if p <= 0:
            return float("-inf")
        return math.log(p)

    def _p_cont_unigram(self, word: str) -> float:
        """Continuation unigram P_cont(w) = N1+(*, w) / N1+(*, *)."""
        assert self.ct is not None
        if self._total_bigram_types == 0:
            return 1.0 / max(self.ct.vocab_size, 1)
        cont_w = self._cont_unigram.get(word, 0)
        if cont_w == 0:
            # Small non-zero floor to handle test tokens whose continuation count
            # is 0 (e.g. tokens mapped to a rarely-appearing UNK). Using 1/V
            # keeps the model proper and avoids the 1e-10 "dead floor" of the
            # previous implementation.
            return 1.0 / max(self._total_bigram_types + self.ct.vocab_size, 1)
        return cont_w / self._total_bigram_types

    def _bigram_top(self, word: str, context: tuple[str, ...]) -> float:
        """Highest-order bigram P_kn(w | w1) using raw bigram counts."""
        assert self.ct is not None
        ctx_counts = self.ct.bigram_counts.get(context, {})
        count_ctx = sum(ctx_counts.values())
        if count_ctx == 0:
            return self._p_cont_unigram(word)

        count_cw = ctx_counts.get(word, 0)
        d = self._discount(count_cw) if count_cw > 0 else 0.0
        first = max(count_cw - d, 0.0) / count_ctx

        n1_c = sum(1 for c in ctx_counts.values() if c == 1)
        n2_c = sum(1 for c in ctx_counts.values() if c == 2)
        n3p_c = sum(1 for c in ctx_counts.values() if c >= 3)
        gamma = (self.d1 * n1_c + self.d2 * n2_c + self.d3 * n3p_c) / count_ctx

        return first + gamma * self._p_cont_unigram(word)

    def _p_cont_bigram(self, word: str, context: tuple[str, ...]) -> float:
        """Continuation bigram P_cont(w | w2) using continuation counts.

        Used when backing off from a trigram model, not at the top level.
        """
        total = self._tri_cont_total.get(context, 0)
        if total == 0:
            return self._p_cont_unigram(word)

        cont_map = self._tri_cont.get(context, {})
        cont_cw = cont_map.get(word, 0)
        d = self._discount(cont_cw) if cont_cw > 0 else 0.0
        first = max(cont_cw - d, 0.0) / total

        n1_c = sum(1 for c in cont_map.values() if c == 1)
        n2_c = sum(1 for c in cont_map.values() if c == 2)
        n3p_c = sum(1 for c in cont_map.values() if c >= 3)
        gamma = (self.d1 * n1_c + self.d2 * n2_c + self.d3 * n3p_c) / total

        return first + gamma * self._p_cont_unigram(word)

    def _trigram_top(self, word: str, context: tuple[str, ...]) -> float:
        """Highest-order trigram with continuation bigram as lower order."""
        assert self.ct is not None
        ctx_counts = self.ct.trigram_counts.get(context, {})
        count_ctx = sum(ctx_counts.values())
        if count_ctx == 0:
            return self._p_cont_bigram(word, context[-1:])

        count_cw = ctx_counts.get(word, 0)
        d = self._discount(count_cw) if count_cw > 0 else 0.0
        first = max(count_cw - d, 0.0) / count_ctx

        n1_c = sum(1 for c in ctx_counts.values() if c == 1)
        n2_c = sum(1 for c in ctx_counts.values() if c == 2)
        n3p_c = sum(1 for c in ctx_counts.values() if c >= 3)
        gamma = (self.d1 * n1_c + self.d2 * n2_c + self.d3 * n3p_c) / count_ctx

        return first + gamma * self._p_cont_bigram(word, context[-1:])
