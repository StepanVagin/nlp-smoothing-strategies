"""Main experiment sweep: evaluate all smoothing methods across corpus sizes and n-gram orders."""

import csv
import json
import sys
from pathlib import Path

from tqdm import tqdm

# Add project root to path so imports work when run from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.corpus import apply_unk, build_vocab, get_subset, load_tokens
from src.evaluate import get_rare_word_sentences, perplexity_on_rare, run_evaluation
from src.ngram import build_counts, frequency_of_frequency, make_count_table
from src.smoothing import AbsoluteDiscounting, GoodTuring, KneserNey, Laplace

SUBSET_FRACS = [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01]
ORDERS = [2, 3]
METHODS = [Laplace, GoodTuring, AbsoluteDiscounting, KneserNey]

DATA_DIR = Path("data")
RESULTS_DIR = Path("results")


def main() -> None:
    """Run the full experiment sweep."""
    RESULTS_DIR.mkdir(exist_ok=True)

    # Load raw tokens
    print("Loading tokens...")
    train_tokens_full = load_tokens(DATA_DIR / "train.txt")
    test_tokens_raw = load_tokens(DATA_DIR / "test.txt")
    print(f"  Train: {len(train_tokens_full):,} tokens")
    print(f"  Test:  {len(test_tokens_raw):,} tokens")

    # CSV output
    fieldnames = [
        "fraction",
        "n_train_tokens",
        "vocab_size",
        "order",
        "method",
        "perplexity",
        "zero_prob_rate",
        "oov_rate",
        "perplexity_rare",
        "kn_d1",
        "kn_d2",
        "kn_d3",
        "gt_unstable",
    ]

    results_path = RESULTS_DIR / "results.csv"
    freq_of_freq_data: dict[str, dict[str, int]] = {}

    total_runs = len(SUBSET_FRACS) * len(ORDERS) * len(METHODS)
    rows: list[dict] = []

    with tqdm(total=total_runs, desc="Experiment sweep") as pbar:
        for frac in SUBSET_FRACS:
            # Slice training subset
            train_subset = get_subset(train_tokens_full, frac)
            n_train = len(train_subset)

            # Build vocab and apply unk
            vocab = build_vocab(train_subset, min_freq=1)
            train_unk = apply_unk(train_subset, vocab)
            test_unk = apply_unk(test_tokens_raw, vocab)
            v_size = len(vocab)

            # Save frequency-of-frequency data for selected fractions
            if frac in [1.0, 0.1, 0.05, 0.01]:
                all_counts = build_counts(train_unk, 3)
                for order in [2, 3]:
                    fof = frequency_of_frequency(all_counts, order)
                    key = f"frac={frac}_order={order}"
                    freq_of_freq_data[key] = {str(k): v for k, v in sorted(fof.items())}

            # Get rare word sentences
            unigram_counts = {}
            for tok in train_unk:
                unigram_counts[tok] = unigram_counts.get(tok, 0) + 1
            rare_sentences = get_rare_word_sentences(test_unk, unigram_counts)

            for order in ORDERS:
                # Build count table
                ct = make_count_table(train_unk, order)

                for MethodClass in METHODS:
                    smoother = MethodClass()
                    smoother.fit(ct)

                    # Evaluate
                    metrics = run_evaluation(smoother, test_unk, order, vocab)
                    ppl_rare = perplexity_on_rare(smoother, rare_sentences, order)

                    row = {
                        "fraction": frac,
                        "n_train_tokens": n_train,
                        "vocab_size": v_size,
                        "order": order,
                        "method": smoother.name,
                        "perplexity": metrics["perplexity"],
                        "zero_prob_rate": metrics["zero_prob_rate"],
                        "oov_rate": metrics["oov_rate"],
                        "perplexity_rare": ppl_rare,
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

    # Write CSV
    with open(results_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nResults written to {results_path}")

    # Write freq-of-freq data
    fof_path = RESULTS_DIR / "freq_of_freq.json"
    with open(fof_path, "w", encoding="utf-8") as f:
        json.dump(freq_of_freq_data, f, indent=2)
    print(f"Frequency-of-frequency data written to {fof_path}")


if __name__ == "__main__":
    main()
