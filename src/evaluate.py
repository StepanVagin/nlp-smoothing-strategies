"""Evaluation utilities for smoothed n-gram models."""

import multiprocessing
import os
import threading

from src.corpus import UNK
from src.smoothing.base import Smoother

# ---------------------------------------------------------------------------
# Module-level state shared with forked worker processes.
# Set by next_word_prediction_metrics() before creating the pool.
# Guarded by _fork_lock so concurrent callers don't clobber each other.
# ---------------------------------------------------------------------------
_fork_lock = threading.Lock()
_worker_smoother = None
_worker_vocab_list: list[str] = []
_worker_test_tokens: list[str] = []
_worker_order: int = 2


def _rank_position(i: int) -> int:
    """Compute 1-based rank of the true next word at position i.

    Rank = (# vocab words with strictly higher score) + 1.
    Avoids a full sort: O(|vocab|) comparisons instead of O(|vocab| log |vocab|).
    """
    context = tuple(_worker_test_tokens[i - _worker_order + 1 : i])
    actual = _worker_test_tokens[i]
    actual_score = _worker_smoother.prob(actual, context)
    n_better = sum(
        1 for w in _worker_vocab_list
        if w != actual and _worker_smoother.prob(w, context) > actual_score
    )
    return n_better + 1


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
    n_jobs: int | None = None,
) -> dict:
    """
    Sample positions from test_tokens and rank every vocabulary word by
    log-probability given the preceding context.

    Positions are evaluated in parallel across CPU cores (fork-based, no
    pickling overhead). Each position scores |vocab| words and records the
    rank of the actual next word via counting rather than sorting.

    Returns top1_accuracy, top5_accuracy, and mrr (mean reciprocal rank).
    """
    import random

    global _worker_smoother, _worker_vocab_list, _worker_test_tokens, _worker_order

    random.seed(seed)

    positions = list(range(order - 1, len(test_tokens)))
    if len(positions) > sample_size:
        positions = random.sample(positions, sample_size)
    positions.sort()

    if not positions:
        return {"top1_accuracy": 0.0, "top5_accuracy": 0.0, "mrr": 0.0}

    workers = min(n_jobs or (os.cpu_count() or 1), len(positions))
    ctx = multiprocessing.get_context("fork")

    # Lock ensures concurrent callers don't race on the module globals that
    # forked workers inherit.  Sequential callers pay zero contention cost.
    with _fork_lock:
        _worker_smoother = smoother
        _worker_vocab_list = sorted(vocab)
        _worker_test_tokens = test_tokens
        _worker_order = order

        with ctx.Pool(workers) as pool:
            ranks = pool.map(_rank_position, positions)

    hits1 = [1 if r == 1 else 0 for r in ranks]
    hits5 = [1 if r <= 5 else 0 for r in ranks]
    rranks = [1.0 / r for r in ranks]

    n = len(ranks)
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
