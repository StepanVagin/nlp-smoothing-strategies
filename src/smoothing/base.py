"""Abstract base class for smoothing strategies."""

import math
from abc import ABC, abstractmethod

from src.ngram import CountTable


class Smoother(ABC):
    """Base class for n-gram smoothing methods."""

    name: str = "base"

    @abstractmethod
    def fit(self, count_table: CountTable) -> None:
        """Fit the smoother to precomputed counts."""

    @abstractmethod
    def prob(self, word: str, context: tuple[str, ...]) -> float:
        """Return the log probability of word given context.

        Returns -inf for zero-probability events.
        """

    def perplexity(self, tokens: list[str], order: int) -> float:
        """Compute perplexity of the token sequence under this model."""
        log_prob_sum = 0.0
        n_tokens = 0

        for i in range(order - 1, len(tokens)):
            context = tuple(tokens[i - order + 1 : i])
            lp = self.prob(tokens[i], context)
            if math.isinf(lp) and lp < 0:
                return float("inf")
            log_prob_sum += lp
            n_tokens += 1

        if n_tokens == 0:
            return float("inf")

        avg_log_prob = log_prob_sum / n_tokens
        return math.exp(-avg_log_prob)

    def zero_prob_rate(self, ngrams: list[tuple], order: int) -> float:
        """Return the fraction of n-grams assigned -inf log probability."""
        if not ngrams:
            return 0.0
        zero_count = 0
        for ngram in ngrams:
            context = ngram[:-1]
            word = ngram[-1]
            lp = self.prob(word, context)
            if math.isinf(lp) and lp < 0:
                zero_count += 1
        return zero_count / len(ngrams)
