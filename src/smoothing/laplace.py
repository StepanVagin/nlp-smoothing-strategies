"""Add-1 (Laplace) smoothing."""

import math

from src.ngram import CountTable
from src.smoothing.base import Smoother


class Laplace(Smoother):
    """Add-1 smoothing for n-gram models."""

    name = "Laplace"

    def __init__(self) -> None:
        self.ct: CountTable | None = None

    def fit(self, count_table: CountTable) -> None:
        """Store the count table."""
        self.ct = count_table

    def prob(self, word: str, context: tuple[str, ...]) -> float:
        """Return log P(word | context) with add-1 smoothing.

        Falls back to unigram if context order does not match available counts.
        """
        assert self.ct is not None
        V = self.ct.vocab_size

        if len(context) == 0:
            # Unigram
            count_w = self.ct.unigram_counts.get(word, 0)
            total = self.ct.total_tokens
            return math.log((count_w + 1) / (total + V))

        if len(context) == 1:
            ctx_counts = self.ct.bigram_counts.get(context, {})
        elif len(context) == 2:
            ctx_counts = self.ct.trigram_counts.get(context, {})
        else:
            # Fall back to last two context words for trigram
            context = context[-2:]
            ctx_counts = self.ct.trigram_counts.get(context, {})

        count_context_word = ctx_counts.get(word, 0)
        count_context = sum(ctx_counts.values()) if ctx_counts else 0

        return math.log((count_context_word + 1) / (count_context + V))
