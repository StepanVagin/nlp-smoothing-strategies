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

    def fit(self, count_table: CountTable) -> None:
        """Store the count table."""
        self.ct = count_table

    def prob(self, word: str, context: tuple[str, ...]) -> float:
        """Return log P(word | context) with absolute discounting.

        Recursively backs off from trigram -> bigram -> unigram (Laplace).
        """
        assert self.ct is not None

        if len(context) == 0:
            return self._unigram_laplace(word)

        if len(context) >= 2:
            ctx = context[-2:]
            ctx_counts = self.ct.trigram_counts.get(ctx, {})
            count_ctx = sum(ctx_counts.values())
            if count_ctx == 0:
                return self.prob(word, context[-1:])
            return self._discounted_prob(word, ctx, ctx_counts, count_ctx, context[-1:])

        # Bigram context (len == 1)
        ctx_counts = self.ct.bigram_counts.get(context, {})
        count_ctx = sum(ctx_counts.values())
        if count_ctx == 0:
            return self._unigram_laplace(word)
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
