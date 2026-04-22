"""Add-1 (Laplace) smoothing."""

import math

from src.ngram import CountTable
from src.smoothing.base import Smoother


class Laplace(Smoother):
    """Add-1 smoothing for n-gram models."""

    name = "Laplace"

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
        V = self.ct.vocab_size

        if len(context) == 0:
            count_w = self.ct.unigram_counts.get(word, 0)
            total = self.ct.total_tokens
            return math.log((count_w + 1) / (total + V))

        if len(context) == 1:
            ctx_counts = self.ct.bigram_counts.get(context, {})
            count_context = self._bigram_ctx_total.get(context, 0)
        elif len(context) == 2:
            ctx_counts = self.ct.trigram_counts.get(context, {})
            count_context = self._trigram_ctx_total.get(context, 0)
        else:
            context = context[-2:]
            ctx_counts = self.ct.trigram_counts.get(context, {})
            count_context = self._trigram_ctx_total.get(context, 0)

        count_context_word = ctx_counts.get(word, 0)
        return math.log((count_context_word + 1) / (count_context + V))
