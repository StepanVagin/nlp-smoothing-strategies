"""Corpus loading and vocabulary utilities."""

from collections import Counter
from pathlib import Path

UNK = "<unk>"


def load_tokens(path: Path) -> list[str]:
    """Read a token file (one token per line) and return a list of tokens."""
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def build_vocab(tokens: list[str], min_freq: int = 1) -> set[str]:
    """Return a vocabulary set, excluding tokens that appear fewer than min_freq times."""
    counts = Counter(tokens)
    vocab = {tok for tok, c in counts.items() if c >= min_freq}
    vocab.add(UNK)
    return vocab


def apply_unk(tokens: list[str], vocab: set[str]) -> list[str]:
    """Replace out-of-vocabulary tokens with <unk>."""
    return [tok if tok in vocab else UNK for tok in tokens]


def get_subset(tokens: list[str], fraction: float) -> list[str]:
    """Return the first fraction * len(tokens) tokens (contiguous slice)."""
    n = max(1, int(len(tokens) * fraction))
    return tokens[:n]


def oov_rate(test_tokens: list[str], vocab: set[str]) -> float:
    """Return the fraction of test tokens not in vocab."""
    if not test_tokens:
        return 0.0
    oov_count = sum(1 for tok in test_tokens if tok not in vocab)
    return oov_count / len(test_tokens)
