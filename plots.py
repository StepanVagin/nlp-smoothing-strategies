"""Generate all 8 figures from experiment results."""

import csv
import json
import math
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path("results")
PLOTS_DIR = Path("plots")

# Consistent color palette
COLORS = {
    "Laplace": "steelblue",
    "GoodTuring": "tomato",
    "AbsoluteDiscounting": "mediumseagreen",
    "KneserNey": "darkorchid",
}

METHOD_ORDER = ["Laplace", "GoodTuring", "AbsoluteDiscounting", "KneserNey"]
METHOD_LABELS = {
    "Laplace": "Laplace",
    "GoodTuring": "Good-Turing",
    "AbsoluteDiscounting": "Abs. Discounting",
    "KneserNey": "Kneser-Ney",
}


def load_results() -> list[dict]:
    """Load results.csv into a list of dicts with numeric conversions."""
    rows = []
    with open(RESULTS_DIR / "results.csv", "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["fraction"] = float(row["fraction"])
            row["n_train_tokens"] = int(row["n_train_tokens"])
            row["vocab_size"] = int(row["vocab_size"])
            row["order"] = int(row["order"])
            row["perplexity"] = float(row["perplexity"]) if row["perplexity"] != "inf" else float("inf")
            row["zero_prob_rate"] = float(row["zero_prob_rate"])
            row["oov_rate"] = float(row["oov_rate"])
            row["perplexity_rare"] = float(row["perplexity_rare"]) if row["perplexity_rare"] != "inf" else float("inf")
            rows.append(row)
    return rows


def filter_rows(rows: list[dict], order: int, method: str) -> list[dict]:
    """Filter rows by order and method, sorted by n_train_tokens."""
    filtered = [r for r in rows if r["order"] == order and r["method"] == method]
    return sorted(filtered, key=lambda r: r["n_train_tokens"])


def setup_style() -> None:
    """Set matplotlib defaults for clean, readable plots."""
    plt.rcParams.update({
        "font.size": 11,
        "axes.facecolor": "white",
        "figure.facecolor": "white",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
    })


def fig1_perplexity_vs_corpus_size(rows: list[dict]) -> None:
    """Figure 1: Perplexity vs corpus size, two-panel (bigram/trigram)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

    for idx, order in enumerate([2, 3]):
        ax = axes[idx]
        for method in METHOD_ORDER:
            data = filter_rows(rows, order, method)
            x = [r["n_train_tokens"] for r in data]
            y = [r["perplexity"] if not math.isinf(r["perplexity"]) else None for r in data]
            # Filter out inf values
            valid = [(xi, yi) for xi, yi in zip(x, y) if yi is not None]
            if valid:
                xv, yv = zip(*valid)
                ax.plot(xv, yv, "o-", color=COLORS[method], label=METHOD_LABELS[method], linewidth=2, markersize=5)

        ax.set_xscale("log")
        ax.set_xlabel("Training tokens")
        ax.set_ylabel("Perplexity")
        ax.set_title(f"{'Bigram' if order == 2 else 'Trigram'} (n={order})")
        ax.axvline(50_000, color="gray", linestyle="--", alpha=0.5, label="50k tokens" if idx == 0 else None)
        ax.axvline(500_000, color="gray", linestyle=":", alpha=0.5, label="500k tokens" if idx == 0 else None)
        ax.legend(fontsize=9)

    fig.suptitle("Perplexity vs. Corpus Size", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "perplexity_vs_corpus_size.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Figure 1: perplexity_vs_corpus_size.png")


def fig2_zero_prob_rate(rows: list[dict]) -> None:
    """Figure 2: Zero-probability rate vs corpus size."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

    for idx, order in enumerate([2, 3]):
        ax = axes[idx]
        for method in METHOD_ORDER:
            data = filter_rows(rows, order, method)
            x = [r["n_train_tokens"] for r in data]
            y = [r["zero_prob_rate"] for r in data]
            ax.plot(x, y, "o-", color=COLORS[method], label=METHOD_LABELS[method], linewidth=2, markersize=5)

        ax.set_xscale("log")
        ax.set_xlabel("Training tokens")
        ax.set_ylabel("Zero-probability rate")
        ax.set_title(f"{'Bigram' if order == 2 else 'Trigram'} (n={order})")
        ax.axvline(50_000, color="gray", linestyle="--", alpha=0.5)
        ax.axvline(500_000, color="gray", linestyle=":", alpha=0.5)
        ax.legend(fontsize=9)

    fig.suptitle("Zero-Probability Rate vs. Corpus Size", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "zero_prob_rate_vs_corpus_size.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Figure 2: zero_prob_rate_vs_corpus_size.png")


def fig3_frequency_of_frequency() -> None:
    """Figure 3: Frequency-of-frequency distributions with GT regression line."""
    fof_path = RESULTS_DIR / "freq_of_freq.json"
    with open(fof_path, "r", encoding="utf-8") as f:
        fof_data = json.load(f)

    fracs = [1.0, 0.1, 0.05, 0.01]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    for idx, frac in enumerate(fracs):
        ax = axes[idx // 2][idx % 2]
        key = f"frac={frac}_order=2"
        if key not in fof_data:
            ax.set_title(f"Fraction {frac} (no data)")
            continue

        n_c = {int(k): v for k, v in fof_data[key].items()}
        cs = list(range(1, 11))
        vals = [n_c.get(c, 0) for c in cs]

        ax.bar(cs, vals, color="steelblue", alpha=0.7, label="Observed N_c")

        # Log-linear GT regression
        points = [(c, n_c[c]) for c in n_c if c > 0 and n_c[c] > 0 and c <= 20]
        if len(points) >= 2:
            log_c = np.array([math.log(c) for c, _ in points])
            log_nc = np.array([math.log(nc) for _, nc in points])
            A = np.vstack([np.ones_like(log_c), log_c]).T
            params = np.linalg.lstsq(A, log_nc, rcond=None)[0]
            a, b = params

            x_fit = np.linspace(1, 10, 50)
            y_fit = np.exp(a + b * np.log(x_fit))
            ax.plot(x_fit, y_fit, "r--", linewidth=2, label="GT regression")

        ax.set_xlabel("Count (c)")
        ax.set_ylabel("N_c")
        ax.set_title(f"Corpus fraction = {frac}")
        ax.set_xticks(cs)
        ax.legend(fontsize=9)

    fig.suptitle("Frequency of Frequency (Bigram)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "frequency_of_frequency.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Figure 3: frequency_of_frequency.png")


def fig4_vocabulary_growth(rows: list[dict]) -> None:
    """Figure 4: Vocabulary growth curve."""
    # Use unique (fraction, n_train_tokens, vocab_size) from order=2
    seen = {}
    for r in rows:
        if r["order"] == 2 and r["method"] == "Laplace":
            seen[r["fraction"]] = (r["n_train_tokens"], r["vocab_size"])

    fracs = sorted(seen.keys())
    x = [seen[f][0] for f in fracs]
    y = [seen[f][1] for f in fracs]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, y, "o-", color="steelblue", linewidth=2, markersize=8)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Training tokens")
    ax.set_ylabel("Vocabulary size")
    ax.set_title("Vocabulary Growth vs. Corpus Size", fontsize=13, fontweight="bold")

    for f, xi, yi in zip(fracs, x, y):
        ax.annotate(f"{f}", (xi, yi), textcoords="offset points", xytext=(5, 8), fontsize=9)

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "vocabulary_growth.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Figure 4: vocabulary_growth.png")


def fig5_kn_discounts(rows: list[dict]) -> None:
    """Figure 5: Kneser-Ney discount parameters vs corpus size."""
    kn_rows = [r for r in rows if r["method"] == "KneserNey" and r["order"] == 2]
    kn_rows.sort(key=lambda r: r["n_train_tokens"])

    x = [r["n_train_tokens"] for r in kn_rows]
    d1 = [float(r["kn_d1"]) for r in kn_rows]
    d2 = [float(r["kn_d2"]) for r in kn_rows]
    d3 = [float(r["kn_d3"]) for r in kn_rows]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x, d1, "o-", color="steelblue", label="d1", linewidth=2, markersize=6)
    ax.plot(x, d2, "s-", color="tomato", label="d2", linewidth=2, markersize=6)
    ax.plot(x, d3, "^-", color="mediumseagreen", label="d3+", linewidth=2, markersize=6)
    ax.set_xscale("log")
    ax.set_xlabel("Training tokens")
    ax.set_ylabel("Discount value")
    ax.set_title("Kneser-Ney Discounts vs. Corpus Size (Bigram)", fontsize=13, fontweight="bold")
    ax.legend()

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "kn_discounts_vs_corpus_size.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Figure 5: kn_discounts_vs_corpus_size.png")


def fig6_perplexity_heatmap(rows: list[dict]) -> None:
    """Figure 6: Perplexity heatmaps (bigram/trigram)."""
    fracs = sorted(set(r["fraction"] for r in rows))

    fig, axes = plt.subplots(1, 2, figsize=(16, 5))

    for idx, order in enumerate([2, 3]):
        ax = axes[idx]
        matrix = []
        for method in METHOD_ORDER:
            row_vals = []
            for frac in fracs:
                match = [r for r in rows if r["order"] == order and r["method"] == method and r["fraction"] == frac]
                if match:
                    val = match[0]["perplexity"]
                    row_vals.append(val if not math.isinf(val) else float("nan"))
                else:
                    row_vals.append(float("nan"))
            matrix.append(row_vals)

        data = np.array(matrix)
        # Use RdYlGn reversed
        im = ax.imshow(data, cmap="RdYlGn_r", aspect="auto")

        ax.set_xticks(range(len(fracs)))
        ax.set_xticklabels([str(f) for f in fracs], fontsize=9)
        ax.set_yticks(range(len(METHOD_ORDER)))
        ax.set_yticklabels([METHOD_LABELS[m] for m in METHOD_ORDER], fontsize=10)
        ax.set_xlabel("Corpus fraction")
        ax.set_title(f"{'Bigram' if order == 2 else 'Trigram'} (n={order})")

        # Annotate cells
        for i in range(len(METHOD_ORDER)):
            for j in range(len(fracs)):
                val = data[i, j]
                if not np.isnan(val):
                    text = f"{val:.1f}" if val < 1e6 else "inf"
                    ax.text(j, i, text, ha="center", va="center", fontsize=8,
                            color="white" if val > np.nanmedian(data) else "black")

        fig.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle("Perplexity Heatmap", fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "perplexity_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Figure 6: perplexity_heatmap.png")


def fig7_perplexity_rare(rows: list[dict]) -> None:
    """Figure 7: Perplexity on rare-word sentences, grouped bar chart."""
    target_fracs = {1.0: "large", 0.1: "medium", 0.01: "small"}
    # Use bigram order
    order = 2

    fig, ax = plt.subplots(figsize=(10, 6))
    x_positions = np.arange(len(target_fracs))
    bar_width = 0.18
    frac_labels = list(target_fracs.values())
    frac_keys = list(target_fracs.keys())

    for i, method in enumerate(METHOD_ORDER):
        vals = []
        for frac in frac_keys:
            match = [r for r in rows if r["order"] == order and r["method"] == method and r["fraction"] == frac]
            if match:
                v = match[0]["perplexity_rare"]
                vals.append(v if not math.isinf(v) else 0)
            else:
                vals.append(0)
        offset = (i - 1.5) * bar_width
        ax.bar(x_positions + offset, vals, bar_width, color=COLORS[method], label=METHOD_LABELS[method])

    # Reference lines: full-test perplexity for each size
    for j, frac in enumerate(frac_keys):
        match = [r for r in rows if r["order"] == order and r["method"] == "KneserNey" and r["fraction"] == frac]
        if match:
            ref_ppl = match[0]["perplexity"]
            if not math.isinf(ref_ppl):
                ax.axhline(ref_ppl, color="gray", linestyle=":", alpha=0.4)

    ax.set_xticks(x_positions)
    ax.set_xticklabels([f"{label}\n(frac={frac})" for label, frac in zip(frac_labels, frac_keys)])
    ax.set_ylabel("Perplexity (rare-word sentences)")
    ax.set_title("Perplexity on Rare-Word Sentences (Bigram)", fontsize=13, fontweight="bold")
    ax.legend()

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "perplexity_rare_words.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Figure 7: perplexity_rare_words.png")


def fig8_decision_guide() -> None:
    """Figure 8: Decision flowchart using matplotlib patches."""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis("off")

    def draw_box(x: float, y: float, w: float, h: float, text: str,
                 boxstyle: str = "round,pad=0.3", facecolor: str = "#E8E8E8",
                 fontsize: int = 10, fontweight: str = "normal") -> None:
        bbox = mpatches.FancyBboxPatch(
            (x - w / 2, y - h / 2), w, h,
            boxstyle=boxstyle, facecolor=facecolor, edgecolor="black", linewidth=1.5
        )
        ax.add_patch(bbox)
        ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
                fontweight=fontweight, wrap=True,
                bbox=dict(facecolor="none", edgecolor="none"))

    def draw_arrow(x1: float, y1: float, x2: float, y2: float, label: str = "") -> None:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color="black", lw=1.5))
        if label:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mx + 0.15, my, label, fontsize=9, fontstyle="italic", color="gray")

    # Root question
    draw_box(7, 9, 5, 0.9, "How large is your\ntraining corpus?",
             boxstyle="round,pad=0.3", facecolor="#FFE4B5", fontsize=12, fontweight="bold")

    # Three branches
    draw_box(2.5, 7, 3, 0.8, "< 50k tokens", facecolor="#FFDAB9", fontsize=11)
    draw_box(7, 7, 3.5, 0.8, "50k - 500k tokens", facecolor="#FFDAB9", fontsize=11)
    draw_box(11.5, 7, 3, 0.8, "> 500k tokens", facecolor="#FFDAB9", fontsize=11)

    draw_arrow(5.5, 8.55, 2.5, 7.4, "small")
    draw_arrow(7, 8.55, 7, 7.4, "medium")
    draw_arrow(8.5, 8.55, 11.5, 7.4, "large")

    # Small branch: bigram vs trigram
    draw_box(1.5, 5.3, 2.5, 0.7, "Bigram?", facecolor="#E0E0E0", fontsize=10)
    draw_box(4, 5.3, 2.5, 0.7, "Trigram?", facecolor="#E0E0E0", fontsize=10)
    draw_arrow(1.8, 6.6, 1.5, 5.7)
    draw_arrow(3.2, 6.6, 4, 5.7)

    draw_box(1.5, 3.8, 2.3, 0.7, "Laplace", facecolor="#B0C4DE", fontsize=11, fontweight="bold")
    draw_box(4, 3.8, 2.8, 0.7, "Abs. Discounting", facecolor="#90EE90", fontsize=10, fontweight="bold")
    draw_arrow(1.5, 4.95, 1.5, 4.15)
    draw_arrow(4, 4.95, 4, 4.15)

    # Medium branch
    draw_box(7, 5.3, 4, 0.8, "Abs. Discounting\nor Kneser-Ney\n(compare both)", facecolor="#98FB98", fontsize=10, fontweight="bold")
    draw_arrow(7, 6.6, 7, 5.7)

    # Large branch
    draw_box(11.5, 5.3, 3.2, 0.7, "Modified\nKneser-Ney", facecolor="#DDA0DD", fontsize=11, fontweight="bold")
    draw_arrow(11.5, 6.6, 11.5, 5.7)

    # Side note
    draw_box(10.5, 2.5, 5, 1.2, "If N1/N2 < 2:\nAvoid Good-Turing\nregardless of corpus size",
             boxstyle="round,pad=0.4", facecolor="#FFB6B6", fontsize=10, fontweight="bold")

    ax.set_title("Smoothing Method Decision Guide", fontsize=14, fontweight="bold", pad=20)

    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "decision_guide.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Figure 8: decision_guide.png")


def main() -> None:
    """Generate all plots."""
    PLOTS_DIR.mkdir(exist_ok=True)
    setup_style()

    print("Loading results...")
    rows = load_results()

    print("Generating figures...")
    fig1_perplexity_vs_corpus_size(rows)
    fig2_zero_prob_rate(rows)
    fig3_frequency_of_frequency()
    fig4_vocabulary_growth(rows)
    fig5_kn_discounts(rows)
    fig6_perplexity_heatmap(rows)
    fig7_perplexity_rare(rows)
    fig8_decision_guide()

    print("All figures saved to plots/")


if __name__ == "__main__":
    main()
