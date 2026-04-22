"""Absolute discounting with recursive backoff."""

import math

from src.ngram import CountTable
from src.smoothing.base import Smoother


class AbsoluteDiscounting(Smoother):
    """Absolute discounting smoothing with fixed discount d=0.75 and recursive backoff."""

    name = "AbsoluteDiscounting"

    D = 0.75

    def __init__(self) -> None:
        self.ct: CountTable | None = None
        self._bigram_ctx_total: dict[tuple, int] = {}
        self._trigram_ctx_total: dict[tuple, int] = {}

    def fit(self, count_table: CountTable) -> None:
        self.ct = count_table
        self._bigram_ctx_total = {ctx: sum(w.values()) for ctx, w in count_table.bigram_counts.items()}
        self._trigram_ctx_total = {ctx: sum(w.values()) for ctx, w in count_table.trigram_counts.items()}

    def prob(self, word: str, context: tuple[str, ...]) -> float:
        assert self.ct is not None

        if len(context) == 0:
            return self._unigram_laplace(word)

        if len(context) >= 2:
            ctx = context[-2:]
            count_ctx = self._trigram_ctx_total.get(ctx, 0)
            if count_ctx == 0:
                return self.prob(word, context[-1:])
            ctx_counts = self.ct.trigram_counts[ctx]
            return self._discounted_prob(word, ctx, ctx_counts, count_ctx, context[-1:])

        # Bigram context (len == 1)
        count_ctx = self._bigram_ctx_total.get(context, 0)
        if count_ctx == 0:
            return self._unigram_laplace(word)
        ctx_counts = self.ct.bigram_counts[context]
        return self._discounted_prob(word, context, ctx_counts, count_ctx, ())

    def _discounted_prob(
        self,
        word: str,
        context: tuple[str, ...],
        ctx_counts: dict[str, int],
        count_ctx: int,
        backoff_context: tuple[str, ...],
    ) -> float:
        """Compute absolute discounted probability with backoff interpolation."""
        count_cw = ctx_counts.get(word, 0)
        d = self.D

        first_term = max(count_cw - d, 0.0) / count_ctx

        # Number of unique words following this context
        n_types = len(ctx_counts)
        backoff_weight = (d / count_ctx) * n_types

        # Recursive lower-order probability (in log space -> convert)
        lower_log_prob = self.prob(word, backoff_context)
        if math.isinf(lower_log_prob):
            lower_prob = 0.0
        else:
            lower_prob = math.exp(lower_log_prob)

        p = first_term + backoff_weight * lower_prob
        if p <= 0:
            return float("-inf")
        return math.log(p)

    def _unigram_laplace(self, word: str) -> float:
        """Laplace-smoothed unigram as the final fallback."""
        assert self.ct is not None
        count_w = self.ct.unigram_counts.get(word, 0)
        V = self.ct.vocab_size
        total = self.ct.total_tokens
        return math.log((count_w + 1) / (total + V))
