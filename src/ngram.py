"""N-gram count building utilities."""

from collections import Counter, defaultdict
from dataclasses import dataclass


@dataclass
class CountTable:
    """Holds precomputed n-gram counts and corpus statistics."""

    unigram_counts: dict[str, int]
    bigram_counts: dict[tuple[str, ...], dict[str, int]]
    trigram_counts: dict[tuple[str, ...], dict[str, int]]
    vocab_size: int
    total_tokens: int


def build_counts(tokens: list[str], order: int) -> dict[int, dict]:
    """Return nested count dicts for all n-grams up to the given order.

    Returns a dict keyed by n (1, 2, 3, ...) where each value is:
      - For n=1: Counter {word: count}
      - For n>=2: defaultdict {context_tuple: {word: count}}
    """
    counts: dict[int, dict] = {}

    # Unigrams
    counts[1] = dict(Counter(tokens))

    # Higher-order n-grams
    for n in range(2, order + 1):
        ngram_counts: dict[tuple[str, ...], dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        for i in range(len(tokens) - n + 1):
            context = tuple(tokens[i : i + n - 1])
            word = tokens[i + n - 1]
            ngram_counts[context][word] += 1
        # Convert inner defaultdicts to plain dicts
        counts[n] = {ctx: dict(words) for ctx, words in ngram_counts.items()}

    return counts


def frequency_of_frequency(counts: dict[int, dict], order: int) -> dict[int, int]:
    """Return N_c mapping: count -> how many n-grams have that count.

    For the specified order level.
    """
    freq_counter: Counter = Counter()

    if order == 1:
        for count in counts[1].values():
            freq_counter[count] += 1
    else:
        for context_words in counts[order].values():
            for count in context_words.values():
                freq_counter[count] += 1

    return dict(freq_counter)


def make_count_table(tokens: list[str], order: int) -> CountTable:
    """Build a CountTable from a token list up to the given order (max 3)."""
    all_counts = build_counts(tokens, max(order, 3))

    unigram = all_counts.get(1, {})
    bigram = all_counts.get(2, {})
    trigram = all_counts.get(3, {})

    vocab_size = len(unigram)
    total_tokens = sum(unigram.values())

    return CountTable(
        unigram_counts=unigram,
        bigram_counts=bigram,
        trigram_counts=trigram,
        vocab_size=vocab_size,
        total_tokens=total_tokens,
    )
