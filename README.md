# Smoothing Study: N-gram Smoothing Strategy Comparison

A pure statistical NLP case study that implements and compares four n-gram smoothing strategies on shrinking subsets of the WikiText-2 corpus to document when each technique breaks down.

## Methods

- **Laplace (Add-1)**: Simple additive smoothing
- **Good-Turing**: Frequency re-estimation with log-linear regression on frequency-of-frequency counts
- **Absolute Discounting**: Fixed discount (d=0.75) with recursive backoff
- **Modified Kneser-Ney**: Three discount parameters with continuation counts and recursive interpolation (Chen & Goodman 1998)

## Setup

```bash
pip install -r requirements.txt
python setup_data.py          # downloads WikiText-2, writes data/{train,valid,test}.txt
python src/experiment.py      # runs the full sweep, writes results/results.csv
python plots.py               # generates all 8 figures into plots/
```

## Experiment Design

Each smoothing method is evaluated across:
- **7 corpus fractions**: 1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01
- **2 n-gram orders**: bigram (n=2), trigram (n=3)

Metrics: perplexity, zero-probability rate, OOV rate, perplexity on rare-word sentences.

## Figures

| Figure | File | Description |
|--------|------|-------------|
| 1 | `perplexity_vs_corpus_size.png` | Perplexity vs training tokens (bigram/trigram panels) |
| 2 | `zero_prob_rate_vs_corpus_size.png` | Zero-probability rate vs training tokens |
| 3 | `frequency_of_frequency.png` | N_c distributions at 4 corpus sizes with GT regression fit |
| 4 | `vocabulary_growth.png` | Vocabulary size vs training tokens (log-log) |
| 5 | `kn_discounts_vs_corpus_size.png` | Kneser-Ney d1, d2, d3+ discount evolution |
| 6 | `perplexity_heatmap.png` | Perplexity heatmaps (method x corpus fraction) |
| 7 | `perplexity_rare_words.png` | Perplexity on rare-word sentences (grouped bars) |
| 8 | `decision_guide.png` | Smoothing method selection flowchart |

## Decision Guide

| Corpus Size | Bigram | Trigram |
|-------------|--------|---------|
| < 50k tokens | Laplace | Absolute Discounting |
| 50k - 500k tokens | Absolute Discounting or Kneser-Ney (compare both) | Absolute Discounting or Kneser-Ney (compare both) |
| > 500k tokens | Modified Kneser-Ney | Modified Kneser-Ney |

**Note**: If N1/N2 < 2, avoid Good-Turing regardless of corpus size (unstable frequency-of-frequency estimates).

## Project Structure

```
smoothing_study/
├── data/                          # created by setup_data.py, gitignored
├── src/
│   ├── corpus.py                  # token loading, vocab, OOV utilities
│   ├── ngram.py                   # n-gram count building, frequency-of-frequency
│   ├── smoothing/
│   │   ├── base.py                # abstract Smoother base class
│   │   ├── laplace.py             # Add-1 smoothing
│   │   ├── good_turing.py         # Simple Good-Turing with regression
│   │   ├── absolute_discounting.py # Fixed discount with recursive backoff
│   │   └── kneser_ney.py          # Modified Kneser-Ney (Chen & Goodman 1998)
│   ├── evaluate.py                # evaluation metrics and rare-word analysis
│   └── experiment.py              # main experiment sweep
├── notebooks/
│   └── analysis.ipynb             # interactive result exploration
├── plots/                         # output directory, gitignored
├── results/
│   └── results.csv                # written by experiment.py
├── setup_data.py                  # data download and preprocessing
├── plots.py                       # standalone figure generation
├── requirements.txt               # datasets, numpy, matplotlib, tqdm
└── README.md
```
