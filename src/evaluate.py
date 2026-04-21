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


def next_word_prediction_metrics(
    smoother: Smoother,
    test_tokens: list[str],
    order: int,
    vocab: set[str],
    sample_size: int = 100,
    seed: int = 42,
) -> dict:
    """
    Sample positions from test_tokens and rank every vocabulary word by
    log-probability given the preceding context.

    For each sampled position the function calls smoother.prob() for all
    vocabulary words, sorts them, and records the rank of the actual next word.

    Returns top1_accuracy, top5_accuracy, and mrr (mean reciprocal rank).
    Running time scales as sample_size × |vocab| × cost(prob), so expect
    ~1-3 minutes for the default parameters on a 30 K vocabulary.
    """
    import random

    random.seed(seed)

    positions = list(range(order - 1, len(test_tokens)))
    if len(positions) > sample_size:
        positions = random.sample(positions, sample_size)
    positions.sort()

    vocab_list = sorted(vocab)
    hits1: list[int] = []
    hits5: list[int] = []
    rranks: list[float] = []

    for i in positions:
        context = tuple(test_tokens[i - order + 1 : i])
        actual = test_tokens[i]
        ranked = sorted(vocab_list, key=lambda w: smoother.prob(w, context), reverse=True)
        rank = next(
            (r + 1 for r, w in enumerate(ranked) if w == actual), len(vocab_list) + 1
        )
        hits1.append(1 if rank == 1 else 0)
        hits5.append(1 if rank <= 5 else 0)
        rranks.append(1.0 / rank)

    n = len(rranks)
    if n == 0:
        return {"top1_accuracy": 0.0, "top5_accuracy": 0.0, "mrr": 0.0}
    return {
        "top1_accuracy": sum(hits1) / n,
        "top5_accuracy": sum(hits5) / n,
        "mrr": sum(rranks) / n,
    }


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
