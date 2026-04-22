"""Generate case-study-focused figures from experiment results.

Each figure answers one of the three questions in the 2.2 case study:

    (a) How does perplexity change as the corpus shrinks?
    (b) *Why* does a method break down -- what sparsity signal causes it?
    (c) Which method should a practitioner pick for a given regime?

The figures are intentionally driven by numbers in ``results.csv`` rather
than by static annotations, so rerunning the experiment regenerates the
decision guide automatically.
"""

import csv
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path("results")
PLOTS_DIR = Path("plots")

COLORS = {
    "Laplace": "#1f77b4",
    "GoodTuring": "#d62728",
    "AbsoluteDiscounting": "#2ca02c",
    "KneserNey": "#9467bd",
}
MARKERS = {
    "Laplace": "o",
    "GoodTuring": "s",
    "AbsoluteDiscounting": "D",
    "KneserNey": "^",
}
METHOD_ORDER = ["Laplace", "GoodTuring", "AbsoluteDiscounting", "KneserNey"]
METHOD_LABELS = {
    "Laplace": "Laplace",
    "GoodTuring": "Good-Turing",
    "AbsoluteDiscounting": "Abs. Discounting",
    "KneserNey": "Mod. Kneser-Ney",
}


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------


def _to_float(val: str) -> float:
    if val in ("", "inf"):
        return float("inf") if val == "inf" else float("nan")
    try:
        return float(val)
    except ValueError:
        return float("nan")


def load_results() -> list[dict]:
    rows: list[dict] = []
    with open(RESULTS_DIR / "results.csv", "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["fraction"] = float(row["fraction"])
            row["n_train_tokens"] = int(row["n_train_tokens"])
            row["vocab_size"] = int(row["vocab_size"])
            row["n_train_types"] = int(row.get("n_train_types", 0) or 0)
            row["order"] = int(row["order"])
            row["perplexity"] = _to_float(row["perplexity"])
            row["zero_prob_rate"] = float(row["zero_prob_rate"])
            row["oov_rate"] = float(row["oov_rate"])
            row["perplexity_rare"] = _to_float(row["perplexity_rare"])
            row["unseen_context_rate"] = float(row.get("unseen_context_rate", 0) or 0)
            row["singleton_frac"] = float(row.get("singleton_frac", 0) or 0)
            row["n1_over_n2"] = _to_float(row.get("n1_over_n2", ""))
            rows.append(row)
    return rows


def filter_rows(rows: list[dict], order: int, method: str) -> list[dict]:
    filtered = [r for r in rows if r["order"] == order and r["method"] == method]
    return sorted(filtered, key=lambda r: r["n_train_tokens"])


def setup_style() -> None:
    plt.rcParams.update({
        "font.size": 11,
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "legend.frameon": False,
    })


def _plot_method_line(
    ax: plt.Axes,
    xs: list[float],
    ys: list[float],
    method: str,
    label: str | None = None,
    linewidth: float = 2.0,
) -> None:
    ax.plot(
        xs,
        ys,
        marker=MARKERS[method],
        color=COLORS[method],
        label=label if label is not None else METHOD_LABELS[method],
        linewidth=linewidth,
        markersize=6,
        markeredgecolor="white",
        markeredgewidth=0.8,
    )


def _finite(values: list[float]) -> list[float]:
    return [v for v in values if math.isfinite(v)]


# ---------------------------------------------------------------------------
# Figure 1: Perplexity vs corpus size with breakdown zones
# ---------------------------------------------------------------------------


def fig1_perplexity_vs_corpus_size(rows: list[dict]) -> None:
    """PPL vs n_train on log-log axes, with a shaded GT-unstable region."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), sharey=False)

    for idx, order in enumerate([2, 3]):
        ax = axes[idx]

        # Identify the GT-unstable region (N1/N2 < 2) empirically.
        diag = sorted(
            [
                (r["n_train_tokens"], r["n1_over_n2"])
                for r in rows
                if r["order"] == order and r["method"] == "KneserNey"
            ]
        )
        unstable_xs = [x for x, ratio in diag if math.isfinite(ratio) and ratio < 2]
        if unstable_xs:
            ax.axvspan(
                min(unstable_xs) * 0.7,
                max(unstable_xs) * 1.4,
                color="#d62728",
                alpha=0.06,
                label="Good-Turing unstable (N₁/N₂ < 2)",
            )

        for method in METHOD_ORDER:
            data = filter_rows(rows, order, method)
            xs = [r["n_train_tokens"] for r in data]
            ys = [r["perplexity"] for r in data]
            valid = [(x, y) for x, y in zip(xs, ys) if math.isfinite(y)]
            if not valid:
                continue
            xv, yv = zip(*valid)
            _plot_method_line(ax, list(xv), list(yv), method)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Training tokens (log scale)")
        ax.set_ylabel("Perplexity (log scale)")
        ax.set_title(f"{'Bigram' if order == 2 else 'Trigram'} (n={order})")
        ax.legend(fontsize=9, loc="upper right")

    fig.suptitle(
        "Perplexity vs. Corpus Size — where each method breaks down",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "perplexity_vs_corpus_size.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Figure 1: perplexity_vs_corpus_size.png")


# ---------------------------------------------------------------------------
# Figure 2: Sparsity diagnostics -- why methods break
# ---------------------------------------------------------------------------


def fig2_sparsity_diagnostics(rows: list[dict]) -> None:
    """The four empirical signals that predict when smoothing fails."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Use KneserNey rows (method-independent diagnostics) for each order.
    diag2 = sorted(
        [r for r in rows if r["method"] == "KneserNey" and r["order"] == 2],
        key=lambda r: r["n_train_tokens"],
    )
    diag3 = sorted(
        [r for r in rows if r["method"] == "KneserNey" and r["order"] == 3],
        key=lambda r: r["n_train_tokens"],
    )

    # (a) N1/N2 ratio vs corpus
    ax = axes[0][0]
    for data, order, color in [(diag2, 2, "#1f77b4"), (diag3, 3, "#ff7f0e")]:
        xs = [r["n_train_tokens"] for r in data]
        ys = [r["n1_over_n2"] if math.isfinite(r["n1_over_n2"]) else np.nan for r in data]
        ax.plot(xs, ys, marker="o", color=color, linewidth=2, label=f"n={order}")
    ax.axhline(2.0, color="#d62728", linestyle="--", label="GT stability threshold")
    ax.set_xscale("log")
    ax.set_xlabel("Training tokens")
    ax.set_ylabel("N₁ / N₂")
    ax.set_title("(a) Good-Turing stability — N₁/N₂ ratio")
    ax.legend(fontsize=9)

    # (b) Singleton fraction
    ax = axes[0][1]
    for data, order, color in [(diag2, 2, "#1f77b4"), (diag3, 3, "#ff7f0e")]:
        xs = [r["n_train_tokens"] for r in data]
        ys = [r["singleton_frac"] for r in data]
        ax.plot(xs, ys, marker="o", color=color, linewidth=2, label=f"n={order}")
    ax.set_xscale("log")
    ax.set_xlabel("Training tokens")
    ax.set_ylabel("Fraction of n-grams seen exactly once")
    ax.set_title("(b) Singleton n-gram fraction (sparsity)")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1)

    # (c) Unseen context rate at test time
    ax = axes[1][0]
    for data, order, color in [(diag2, 2, "#1f77b4"), (diag3, 3, "#ff7f0e")]:
        xs = [r["n_train_tokens"] for r in data]
        ys = [r["unseen_context_rate"] for r in data]
        ax.plot(xs, ys, marker="o", color=color, linewidth=2, label=f"n={order}")
    ax.set_xscale("log")
    ax.set_xlabel("Training tokens")
    ax.set_ylabel("Test n-grams with unseen context")
    ax.set_title("(c) Backoff pressure — rate of unseen test contexts")
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1)

    # (d) Vocabulary growth (Heaps' law style)
    ax = axes[1][1]
    seen = {r["fraction"]: (r["n_train_tokens"], r["n_train_types"]) for r in diag2}
    fracs = sorted(seen.keys())
    xs = [seen[f][0] for f in fracs]
    ys = [seen[f][1] for f in fracs]
    ax.plot(xs, ys, marker="o", color="#2ca02c", linewidth=2, markersize=7)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Training tokens")
    ax.set_ylabel("Unique training token types")
    ax.set_title("(d) Vocabulary growth (Heaps-like)")

    fig.suptitle(
        "Sparsity Diagnostics — why each method breaks",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "sparsity_diagnostics.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Figure 2: sparsity_diagnostics.png")


# ---------------------------------------------------------------------------
# Figure 3: Relative perplexity to best-per-size
# ---------------------------------------------------------------------------


def fig3_relative_perplexity(rows: list[dict]) -> None:
    """Ratio PPL(method) / min_method PPL at each corpus size."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    for idx, order in enumerate([2, 3]):
        ax = axes[idx]
        sizes = sorted({r["n_train_tokens"] for r in rows if r["order"] == order})

        # Per-size minimum over finite perplexities.
        best_by_size: dict[int, float] = {}
        for s in sizes:
            at_size = [
                r["perplexity"]
                for r in rows
                if r["order"] == order
                and r["n_train_tokens"] == s
                and math.isfinite(r["perplexity"])
            ]
            if at_size:
                best_by_size[s] = min(at_size)

        for method in METHOD_ORDER:
            data = filter_rows(rows, order, method)
            xs, ys = [], []
            for r in data:
                s = r["n_train_tokens"]
                if s in best_by_size and math.isfinite(r["perplexity"]):
                    xs.append(s)
                    ys.append(r["perplexity"] / best_by_size[s])
            if xs:
                _plot_method_line(ax, xs, ys, method)

        ax.axhline(1.0, color="black", linewidth=1, linestyle=":")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Training tokens")
        ax.set_ylabel("Perplexity / best-at-size")
        ax.set_title(f"{'Bigram' if order == 2 else 'Trigram'} (n={order})")
        ax.legend(fontsize=9)

    fig.suptitle(
        "Relative Perplexity — how far each method is from the winner",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "relative_perplexity.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Figure 3: relative_perplexity.png")


# ---------------------------------------------------------------------------
# Figure 4: OOV + zero-probability rate
# ---------------------------------------------------------------------------


def fig4_oov_and_zero_prob(rows: list[dict]) -> None:
    """Real OOV rate against fixed vocab + zero-probability rate per method."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Panel A: OOV rate is the same per corpus size (fixed vocab).
    ax = axes[0]
    bigrams = filter_rows(rows, 2, "KneserNey")
    xs = [r["n_train_tokens"] for r in bigrams]
    ys = [r["oov_rate"] * 100 for r in bigrams]
    ax.plot(xs, ys, marker="o", color="#1f77b4", linewidth=2, markersize=7)
    ax.set_xscale("log")
    ax.set_xlabel("Training tokens")
    ax.set_ylabel("Test tokens mapped to <unk> (%)")
    ax.set_title("(a) OOV rate against fixed vocabulary")

    # Panel B: zero-probability rate per method, bigram.
    ax = axes[1]
    for method in METHOD_ORDER:
        data = filter_rows(rows, 2, method)
        xs = [r["n_train_tokens"] for r in data]
        ys = [r["zero_prob_rate"] * 100 for r in data]
        _plot_method_line(ax, xs, ys, method)
    ax.set_xscale("log")
    ax.set_xlabel("Training tokens")
    ax.set_ylabel("Test n-grams with P=0 (%)")
    ax.set_title("(b) Zero-probability rate (bigram)")
    ax.legend(fontsize=9)

    fig.suptitle(
        "OOV and Zero-Probability Behaviour",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "oov_and_zero_prob.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Figure 4: oov_and_zero_prob.png")


# ---------------------------------------------------------------------------
# Figure 5: KN discount evolution
# ---------------------------------------------------------------------------


def fig5_kn_discount_evolution(rows: list[dict]) -> None:
    """Evolution of d1, d2, d3+ with the theoretical asymptote [0.5, 1.5, 2.5]."""
    data = filter_rows(rows, 2, "KneserNey")
    xs = [r["n_train_tokens"] for r in data]
    d1 = [float(r["kn_d1"]) for r in data if r["kn_d1"] != ""]
    d2 = [float(r["kn_d2"]) for r in data if r["kn_d2"] != ""]
    d3 = [float(r["kn_d3"]) for r in data if r["kn_d3"] != ""]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(xs, d1, marker="o", color="#1f77b4", label="d₁", linewidth=2)
    ax.plot(xs, d2, marker="s", color="#d62728", label="d₂", linewidth=2)
    ax.plot(xs, d3, marker="^", color="#2ca02c", label="d₃₊", linewidth=2)

    for level, color in [(0.5, "#1f77b4"), (1.5, "#d62728"), (2.5, "#2ca02c")]:
        ax.axhline(level, color=color, linestyle=":", alpha=0.5)

    ax.text(
        xs[-1],
        0.5,
        " asymptote d₁ = 0.5",
        color="#1f77b4",
        fontsize=9,
        va="center",
    )
    ax.text(
        xs[-1],
        1.5,
        " asymptote d₂ = 1.5",
        color="#d62728",
        fontsize=9,
        va="center",
    )
    ax.text(
        xs[-1],
        2.5,
        " asymptote d₃₊ = 2.5",
        color="#2ca02c",
        fontsize=9,
        va="center",
    )

    ax.set_xscale("log")
    ax.set_xlabel("Training tokens")
    ax.set_ylabel("Discount value")
    ax.set_title(
        "Kneser-Ney Discount Evolution (bigram level) vs. theoretical limits",
        fontsize=12,
        fontweight="bold",
    )
    ax.legend(loc="center left")

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "kn_discount_evolution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Figure 5: kn_discount_evolution.png")


# ---------------------------------------------------------------------------
# Figure 6: Count-frequency distribution overlays
# ---------------------------------------------------------------------------


def fig6_count_distribution() -> None:
    """N_c distribution on log-log axes at three corpus sizes."""
    fof_path = RESULTS_DIR / "freq_of_freq.json"
    with open(fof_path, "r", encoding="utf-8") as f:
        fof = json.load(f)

    fracs = [1.0, 0.1, 0.01]
    colors = ["#1f77b4", "#ff7f0e", "#d62728"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    for idx, order in enumerate([2, 3]):
        ax = axes[idx]
        for frac, color in zip(fracs, colors):
            key = f"frac={frac}_order={order}"
            if key not in fof:
                continue
            nc = {int(k): v for k, v in fof[key].items()}
            cs = sorted(nc.keys())
            vals = [nc[c] for c in cs]
            ax.loglog(
                cs,
                vals,
                marker="o",
                color=color,
                markersize=4,
                linewidth=1.5,
                alpha=0.85,
                label=f"fraction = {frac}",
            )

        ax.set_xlabel("Count c")
        ax.set_ylabel("N_c (number of n-grams with count c)")
        ax.set_title(f"{'Bigram' if order == 2 else 'Trigram'} count spectrum")
        ax.legend(fontsize=9)

    fig.suptitle(
        "Count-Frequency Distributions — Zipfian tail dominates low-resource regime",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "count_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Figure 6: count_distribution.png")


# ---------------------------------------------------------------------------
# Figure 7: Rare-word-sentence perplexity
# ---------------------------------------------------------------------------


def fig7_rare_word_perplexity(rows: list[dict]) -> None:
    """Perplexity on the rare-word test subset — where sparsity really bites."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    for idx, order in enumerate([2, 3]):
        ax = axes[idx]
        for method in METHOD_ORDER:
            data = filter_rows(rows, order, method)
            xs = [r["n_train_tokens"] for r in data]
            ys = [r["perplexity_rare"] for r in data]
            valid = [(x, y) for x, y in zip(xs, ys) if math.isfinite(y)]
            if not valid:
                continue
            xv, yv = zip(*valid)
            _plot_method_line(ax, list(xv), list(yv), method)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("Training tokens")
        ax.set_ylabel("Perplexity on rare-word sentences")
        ax.set_title(f"{'Bigram' if order == 2 else 'Trigram'} (n={order})")
        ax.legend(fontsize=9)

    fig.suptitle(
        "Perplexity on Rare-Word Sentences (>50% singleton tokens)",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "rare_word_perplexity.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Figure 7: rare_word_perplexity.png")


# ---------------------------------------------------------------------------
# Figure 8: Empirical decision matrix
# ---------------------------------------------------------------------------


def fig8_decision_matrix(rows: list[dict]) -> None:
    """Empirical decision guide: best method per (corpus size, n-gram order).

    Each cell is coloured by the winning method and annotated with the
    runner-up's relative PPL so the practitioner can see how robust the
    recommendation is.
    """
    fracs = sorted({r["fraction"] for r in rows}, reverse=True)
    orders = [2, 3]

    best = np.empty((len(orders), len(fracs)), dtype=object)
    margin = np.full((len(orders), len(fracs)), np.nan)

    for i, order in enumerate(orders):
        for j, frac in enumerate(fracs):
            cell_rows = [
                r
                for r in rows
                if r["order"] == order
                and r["fraction"] == frac
                and math.isfinite(r["perplexity"])
            ]
            if not cell_rows:
                best[i][j] = None
                continue
            cell_rows.sort(key=lambda r: r["perplexity"])
            winner = cell_rows[0]
            best[i][j] = winner["method"]
            if len(cell_rows) > 1:
                margin[i][j] = cell_rows[1]["perplexity"] / winner["perplexity"]

    fig, ax = plt.subplots(figsize=(14, 4.8))
    color_map = {m: COLORS[m] for m in METHOD_ORDER}

    for i in range(len(orders)):
        for j in range(len(fracs)):
            m = best[i][j]
            color = color_map.get(m, "#E0E0E0") if m else "#E0E0E0"
            ax.add_patch(
                plt.Rectangle((j, -i), 1, 1, facecolor=color, edgecolor="white", linewidth=2, alpha=0.85)
            )
            if m is not None:
                ratio = margin[i][j]
                ratio_str = f"×{ratio:.2f}" if math.isfinite(ratio) else "—"
                ax.text(
                    j + 0.5,
                    -i + 0.6,
                    METHOD_LABELS[m],
                    ha="center",
                    va="center",
                    fontsize=11,
                    fontweight="bold",
                    color="white",
                )
                ax.text(
                    j + 0.5,
                    -i + 0.25,
                    f"runner-up {ratio_str}",
                    ha="center",
                    va="center",
                    fontsize=8.5,
                    color="white",
                    alpha=0.9,
                )

    ax.set_xticks([j + 0.5 for j in range(len(fracs))])
    n_tokens = {}
    for r in rows:
        n_tokens[r["fraction"]] = r["n_train_tokens"]
    ax.set_xticklabels(
        [
            f"{frac}\n({n_tokens[frac]:,} tok)"
            for frac in fracs
        ],
        fontsize=9,
    )
    ax.set_yticks([-i + 0.5 for i in range(len(orders))])
    ax.set_yticklabels([f"n={o}" for o in orders], fontsize=11)
    ax.set_xlim(0, len(fracs))
    ax.set_ylim(-len(orders) + 1, 1)
    ax.set_xlabel("Corpus fraction (and training tokens)")

    # Legend swatches.
    from matplotlib.patches import Patch
    handles = [Patch(color=COLORS[m], label=METHOD_LABELS[m]) for m in METHOD_ORDER]
    ax.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.55), frameon=False)

    ax.set_title(
        "Empirical Decision Guide — winner per regime with runner-up PPL ratio\n"
        "(perplexity-based; for ranking/autocomplete tasks see Figure 9)",
        fontsize=12,
        fontweight="bold",
        pad=14,
    )
    ax.grid(False)
    ax.set_aspect("auto")

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "decision_matrix.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Figure 8: decision_matrix.png")


# ---------------------------------------------------------------------------
# Figure 9: Perplexity rank vs top-1 ranking rank — the metric contrast
# ---------------------------------------------------------------------------

# Results from next_word_task.ipynb at full corpus (fraction=1.0), 200 samples.
_RANKING_RESULTS = {
    # (order, split, method) -> {"top1": float, "top5": float, "mrr": float}
    (2, "all",    "Laplace"):             {"top1": 15.50, "top5": 31.50, "mrr": 0.23986},
    (2, "all",    "GoodTuring"):          {"top1": 15.50, "top5": 30.50, "mrr": 0.23484},
    (2, "all",    "AbsoluteDiscounting"): {"top1": 15.00, "top5": 30.00, "mrr": 0.22792},
    (2, "all",    "KneserNey"):           {"top1": 15.00, "top5": 30.50, "mrr": 0.22754},
    (3, "all",    "Laplace"):             {"top1": 41.00, "top5": 65.00, "mrr": 0.52082},
    (3, "all",    "GoodTuring"):          {"top1": 21.50, "top5": 41.50, "mrr": 0.30556},
    (3, "all",    "AbsoluteDiscounting"): {"top1": 19.00, "top5": 39.50, "mrr": 0.28795},
    (3, "all",    "KneserNey"):           {"top1": 18.00, "top5": 40.50, "mrr": 0.28026},
    (2, "unseen", "Laplace"):             {"top1":  0.00, "top5":  4.50, "mrr": 0.04243},
    (2, "unseen", "GoodTuring"):          {"top1":  0.00, "top5":  1.00, "mrr": 0.01437},
    (2, "unseen", "AbsoluteDiscounting"): {"top1":  0.00, "top5":  1.50, "mrr": 0.01675},
    (2, "unseen", "KneserNey"):           {"top1":  0.00, "top5":  1.00, "mrr": 0.01731},
    (3, "unseen", "Laplace"):             {"top1": 32.50, "top5": 59.50, "mrr": 0.44748},
    (3, "unseen", "GoodTuring"):          {"top1":  9.00, "top5": 21.00, "mrr": 0.15540},
    (3, "unseen", "AbsoluteDiscounting"): {"top1": 10.00, "top5": 20.00, "mrr": 0.15966},
    (3, "unseen", "KneserNey"):           {"top1":  9.00, "top5": 21.00, "mrr": 0.15354},
}


def fig9_metric_comparison(rows: list[dict]) -> None:
    """Contrast perplexity rank vs top-1 rank at full corpus — the key tension.

    Four panels arranged as a 2×2 grid:
      rows = bigram / trigram
      cols = perplexity (lower is better) / top-1 accuracy (higher is better)

    Perplexity bars are normalised to the best method at that order so both
    axes start near 1.0 / 0% and the gap is easy to read.
    """
    full_rows = [r for r in rows if r["fraction"] == 1.0]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(
        "Perplexity vs. Top-1 Ranking at Full Corpus (fraction = 1.0)\n"
        "Same data, different metric — different winner",
        fontsize=13,
        fontweight="bold",
    )

    x = np.arange(len(METHOD_ORDER))
    width = 0.55

    for row_idx, order in enumerate([2, 3]):
        gram = "Bigram" if order == 2 else "Trigram"

        # --- left panel: normalised perplexity ---
        ax_ppl = axes[row_idx][0]
        ppls = {}
        for r in full_rows:
            if r["order"] == order and math.isfinite(r["perplexity"]):
                ppls[r["method"]] = r["perplexity"]
        best_ppl = min(ppls.values()) if ppls else 1.0
        vals_ppl = [ppls.get(m, float("nan")) / best_ppl for m in METHOD_ORDER]
        bars = ax_ppl.bar(x, vals_ppl, width, color=[COLORS[m] for m in METHOD_ORDER], alpha=0.85)
        ax_ppl.axhline(1.0, color="black", linewidth=0.8, linestyle=":")
        for bar, val in zip(bars, vals_ppl):
            if math.isfinite(val):
                ax_ppl.text(
                    bar.get_x() + bar.get_width() / 2,
                    val + 0.05,
                    f"×{val:.2f}",
                    ha="center", va="bottom", fontsize=8.5,
                )
        ax_ppl.set_xticks(x)
        ax_ppl.set_xticklabels([METHOD_LABELS[m] for m in METHOD_ORDER], rotation=15, ha="right", fontsize=9)
        ax_ppl.set_ylabel("PPL / best-at-order (lower = better)")
        ax_ppl.set_title(f"{gram} — Perplexity (relative)")
        ax_ppl.set_ylim(0, max(v for v in vals_ppl if math.isfinite(v)) * 1.15)

        # --- right panel: top-1 accuracy ---
        ax_top1 = axes[row_idx][1]
        vals_top1 = [_RANKING_RESULTS[(order, "all", m)]["top1"] for m in METHOD_ORDER]
        bars2 = ax_top1.bar(x, vals_top1, width, color=[COLORS[m] for m in METHOD_ORDER], alpha=0.85)
        for bar, val in zip(bars2, vals_top1):
            ax_top1.text(
                bar.get_x() + bar.get_width() / 2,
                val + 0.4,
                f"{val:.1f}%",
                ha="center", va="bottom", fontsize=8.5,
            )
        ax_top1.set_xticks(x)
        ax_top1.set_xticklabels([METHOD_LABELS[m] for m in METHOD_ORDER], rotation=15, ha="right", fontsize=9)
        ax_top1.set_ylabel("Top-1 Accuracy % (higher = better)")
        ax_top1.set_title(f"{gram} — Top-1 Accuracy (all positions)")
        ax_top1.set_ylim(0, max(vals_top1) * 1.18)

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "metric_comparison.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Figure 9: metric_comparison.png")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    PLOTS_DIR.mkdir(exist_ok=True)
    setup_style()

    print("Loading results...")
    rows = load_results()

    print("Generating figures...")
    fig1_perplexity_vs_corpus_size(rows)
    fig2_sparsity_diagnostics(rows)
    fig3_relative_perplexity(rows)
    fig4_oov_and_zero_prob(rows)
    fig5_kn_discount_evolution(rows)
    fig6_count_distribution()
    fig7_rare_word_perplexity(rows)
    fig8_decision_matrix(rows)
    fig9_metric_comparison(rows)

    print(f"All figures saved to {PLOTS_DIR}/")


if __name__ == "__main__":
    main()
