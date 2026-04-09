"""Evaluation utilities for smoothed n-gram models."""

from src.corpus import UNK
from src.smoothing.base import Smoother


def run_evaluation(
    smoother: Smoother,
    test_tokens: list[str],
    order: int,
    vocab: set[str],
) -> dict:
    """Run full evaluation of a smoother on test tokens.

    Returns a dict with perplexity, zero_prob_rate, and oov_rate.
    """
    from src.corpus import oov_rate as compute_oov_rate

    ppl = smoother.perplexity(test_tokens, order)

    # Build test n-grams for zero-prob measurement
    test_ngrams = []
    for i in range(order - 1, len(test_tokens)):
        ngram = tuple(test_tokens[i - order + 1 : i + 1])
        test_ngrams.append(ngram)

    zpr = smoother.zero_prob_rate(test_ngrams, order)
    oov = compute_oov_rate(test_tokens, vocab)

    return {
        "perplexity": ppl,
        "zero_prob_rate": zpr,
        "oov_rate": oov,
    }


def get_rare_word_sentences(
    test_tokens: list[str],
    train_counts: dict[str, int],
    threshold: int = 1,
) -> list[list[str]]:
    """Return sentences where >50% of tokens are singletons (count <= threshold) in training.

    Sentences are delimited by <eos> tokens.
    """
    sentences: list[list[str]] = []
    current: list[str] = []

    for tok in test_tokens:
        if tok == "<eos>":
            if current:
                sentences.append(current)
            current = []
        else:
            current.append(tok)
    if current:
        sentences.append(current)

    rare_sentences = []
    for sent in sentences:
        if len(sent) == 0:
            continue
        rare_count = sum(
            1 for tok in sent if train_counts.get(tok, 0) <= threshold
        )
        if rare_count / len(sent) > 0.5:
            rare_sentences.append(sent)

    return rare_sentences


def perplexity_on_rare(
    smoother: Smoother,
    rare_sentences: list[list[str]],
    order: int,
) -> float:
    """Compute perplexity of the smoother on rare-word sentences.

    Concatenates all rare sentences with <eos> boundaries and computes perplexity.
    """
    if not rare_sentences:
        return float("inf")

    tokens: list[str] = []
    for sent in rare_sentences:
        tokens.extend(sent)
        tokens.append("<eos>")

    return smoother.perplexity(tokens, order)
