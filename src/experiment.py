"""Main experiment sweep: evaluate all smoothing methods across corpus sizes and n-gram orders."""

import csv
import json
import sys
from pathlib import Path

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.corpus import UNK, apply_unk, build_vocab, get_subset, load_tokens, oov_rate
from src.evaluate import get_rare_word_sentences, perplexity_on_rare, run_evaluation
from src.ngram import build_counts, frequency_of_frequency, make_count_table
from src.smoothing import AbsoluteDiscounting, GoodTuring, KneserNey, Laplace

SUBSET_FRACS = [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01]
ORDERS = [2, 3]
METHODS = [Laplace, GoodTuring, AbsoluteDiscounting, KneserNey]

# Fixed vocabulary is built from the full training corpus with this min_freq
# cutoff. Tokens below the cutoff are rewritten to <unk> so that UNK appears
# in training and acquires a real probability mass. The same vocab is applied
# to every subset and to the test set, which preserves OOV as an honest
# signal across the sweep.
VOCAB_MIN_FREQ = 3

DATA_DIR = Path("data")
RESULTS_DIR = Path("results")


def main() -> None:
    """Run the full experiment sweep with a fixed vocabulary."""
    RESULTS_DIR.mkdir(exist_ok=True)

    print("Loading tokens...")
    train_tokens_full = load_tokens(DATA_DIR / "train.txt")
    test_tokens_raw = load_tokens(DATA_DIR / "test.txt")
    print(f"  Train: {len(train_tokens_full):,} tokens")
    print(f"  Test:  {len(test_tokens_raw):,} tokens")

    print(f"Building fixed vocabulary from full train (min_freq={VOCAB_MIN_FREQ})...")
    fixed_vocab = build_vocab(train_tokens_full, min_freq=VOCAB_MIN_FREQ)
    print(f"  Fixed vocab size: {len(fixed_vocab):,}")

    # Apply UNK once to the full training corpus and to the test set.
    train_tokens_unk = apply_unk(train_tokens_full, fixed_vocab)
    test_unk = apply_unk(test_tokens_raw, fixed_vocab)

    # "Real" OOV rate against the fixed vocabulary (same for every run).
    test_oov_rate = oov_rate(test_tokens_raw, fixed_vocab)
    print(f"  Test OOV rate (fixed vocab): {test_oov_rate:.4f}")

    fieldnames = [
        "fraction",
        "n_train_tokens",
        "vocab_size",
        "n_train_types",
        "order",
        "method",
        "perplexity",
        "zero_prob_rate",
        "oov_rate",
        "perplexity_rare",
        "unseen_context_rate",
        "singleton_frac",
        "n1_over_n2",
        "kn_d1",
        "kn_d2",
        "kn_d3",
        "gt_unstable",
    ]

    results_path = RESULTS_DIR / "results.csv"
    freq_of_freq_data: dict[str, dict[str, int]] = {}
    rows: list[dict] = []
    total_runs = len(SUBSET_FRACS) * len(ORDERS) * len(METHODS)

    with tqdm(total=total_runs, desc="Experiment sweep") as pbar:
        for frac in SUBSET_FRACS:
            train_subset = get_subset(train_tokens_unk, frac)
            n_train = len(train_subset)
            n_train_types = len(set(train_subset))
            v_size = len(fixed_vocab)

            if frac in [1.0, 0.1, 0.05, 0.01]:
                all_counts = build_counts(train_subset, 3)
                for order in [2, 3]:
                    fof = frequency_of_frequency(all_counts, order)
                    key = f"frac={frac}_order={order}"
                    freq_of_freq_data[key] = {str(k): v for k, v in sorted(fof.items())}

            unigram_counts: dict[str, int] = {}
            for tok in train_subset:
                unigram_counts[tok] = unigram_counts.get(tok, 0) + 1
            rare_sentences = get_rare_word_sentences(test_unk, unigram_counts)

            for order in ORDERS:
                ct = make_count_table(train_subset, order)

                # Sparsity diagnostics for this (frac, order) cell.
                ngram_level = ct.bigram_counts if order == 2 else ct.trigram_counts
                all_counts_flat = [
                    c for ctx in ngram_level.values() for c in ctx.values()
                ]
                n1 = sum(1 for c in all_counts_flat if c == 1)
                n2 = sum(1 for c in all_counts_flat if c == 2)
                singleton_frac = n1 / len(all_counts_flat) if all_counts_flat else 0.0
                n1_over_n2 = n1 / n2 if n2 > 0 else float("inf")

                # Fraction of test n-grams whose context was never seen in training.
                seen_contexts = set(ngram_level.keys())
                test_contexts = [
                    tuple(test_unk[i - order + 1 : i])
                    for i in range(order - 1, len(test_unk))
                ]
                if test_contexts:
                    unseen_context_rate = sum(
                        1 for c in test_contexts if c not in seen_contexts
                    ) / len(test_contexts)
                else:
                    unseen_context_rate = 0.0

                for MethodClass in METHODS:
                    smoother = MethodClass()
                    smoother.fit(ct)

                    metrics = run_evaluation(smoother, test_unk, order, fixed_vocab)
                    ppl_rare = perplexity_on_rare(smoother, rare_sentences, order)

                    row = {
                        "fraction": frac,
                        "n_train_tokens": n_train,
                        "vocab_size": v_size,
                        "n_train_types": n_train_types,
                        "order": order,
                        "method": smoother.name,
                        "perplexity": metrics["perplexity"],
                        "zero_prob_rate": metrics["zero_prob_rate"],
                        "oov_rate": test_oov_rate,
                        "perplexity_rare": ppl_rare,
                        "unseen_context_rate": unseen_context_rate,
                        "singleton_frac": singleton_frac,
                        "n1_over_n2": n1_over_n2,
                        "kn_d1": "",
                        "kn_d2": "",
                        "kn_d3": "",
                        "gt_unstable": "",
                    }

                    if isinstance(smoother, KneserNey):
                        row["kn_d1"] = smoother.d1
                        row["kn_d2"] = smoother.d2
                        row["kn_d3"] = smoother.d3

                    if isinstance(smoother, GoodTuring):
                        row["gt_unstable"] = smoother.unstable

                    rows.append(row)

                    ppl_str = (
                        f"{metrics['perplexity']:.1f}"
                        if metrics["perplexity"] != float("inf")
                        else "inf"
                    )
                    pbar.set_postfix_str(
                        f"frac={frac} order={order} {smoother.name} ppl={ppl_str}"
                    )
                    pbar.update(1)

    with open(results_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nResults written to {results_path}")

    fof_path = RESULTS_DIR / "freq_of_freq.json"
    with open(fof_path, "w", encoding="utf-8") as f:
        json.dump(freq_of_freq_data, f, indent=2)
    print(f"Frequency-of-frequency data written to {fof_path}")


if __name__ == "__main__":
    main()
